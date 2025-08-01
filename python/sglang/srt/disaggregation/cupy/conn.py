from __future__ import annotations

import dataclasses
import logging
import struct
import threading
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set
import ctypes

import cupy as cp
import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import BaseKVSender, KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)

GUARD = "CupyMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_ptrs: List[int]
    dst_aux_ptrs: List[int]
    gpu_id: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            gpu_id=int(msg[5].decode("ascii")),
            dst_kv_ptrs=[cp.cuda.runtime.ipcOpenMemHandle(handle) for handle in msg[6:]],
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunk IDs that have been received.
    received_kvs: Set[int] = dataclasses.field(default_factory=set)
    # Number of kv chunks to expect, will know this after last chunk is received.
    num_kvs_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False

    def is_done(self):
        if self.num_kvs_expected is None:
            return False
        return self.num_kvs_expected == len(self.received_kvs) and self.received_aux


class CupyKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "Please install CuPy to run SGLang with CuPyTransferEngine."
            ) from e
        self.agent_name = str(uuid.uuid4())
        self.device = cp.cuda.Device(self.kv_args.gpu_id)
        self.server_socket = zmq.Context().socket(zmq.PULL)

        self._start_communication_thread()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.request_status: Dict[int, KVPoll] = {}
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: The prefill engine could recv bootstrapping first
            self.request_status[bootstrap_room] = max(
                self.request_status[bootstrap_room], status
            )

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self._ensure_peer_access(self.kv_args.gpu_id, decode_kv_args.gpu_id)

    def _ensure_peer_access(self, src_gpu: int, dst_gpu: int):
        assert src_gpu != dst_gpu
        can_access = cp.cuda.runtime.deviceCanAccessPeer(src_gpu, dst_gpu)
        assert can_access, f"Peer access not supported between GPU {src_gpu} and {dst_gpu}"
        with cp.cuda.Device(src_gpu):
            cp.cuda.runtime.deviceEnablePeerAccess(dst_gpu)

    def _copy_peer_memory(self, src_addr: int, src_gpu: int, dst_addr: int, dst_gpu: int, length: int):
        with cp.cuda.Device(src_gpu):
            src_mem = cp.cuda.UnownedMemory(src_addr, length, None)
            src_ptr = cp.cuda.MemoryPointer(src_mem, 0)
            src_array = cp.ndarray(shape=(length,), dtype=cp.uint8, memptr=src_ptr)

        with cp.cuda.Device(dst_gpu):
            dst_mem = cp.cuda.UnownedMemory(dst_addr, length, None)
            dst_ptr = cp.cuda.MemoryPointer(dst_mem, 0)
            dst_array = cp.ndarray(shape=(length,), dtype=cp.uint8, memptr=dst_ptr)

        cp.copyto(dst_array, src_array)

        logger.debug(f"Successfully copied {length} bytes from GPU {src_gpu} to GPU {dst_gpu}")

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")

        event = cp.cuda.Event()
        num_layers = len(self.kv_args.kv_data_ptrs)

        with self.device:
            for layer_id in range(num_layers):
                src_ptr = self.kv_args.kv_data_ptrs[layer_id]
                dst_ptr = dst_kv_ptrs[layer_id]
                item_len = self.kv_args.kv_item_lens[layer_id]

                for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    length = item_len * len(prefill_index)
                    self._copy_peer_memory(src_addr, self.kv_args.gpu_id, dst_addr, dst_gpu_id, length)

            event.record()

        logger.debug(
            f"len(transfers): before group: {len(prefill_kv_indices)},"
            f"after group: {len(prefill_kv_blocks) * num_layers}"
        )

        return event

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        events = []
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue

            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            notif = "_".join([str(req.room), "kv", str(chunk_id), str(int(is_last))])
            kv_event = self.send_kvcache(
                req.agent_name,
                kv_indices,
                self.decode_kv_args_table[req.agent_name].dst_kv_ptrs,
                chunked_dst_kv_indice,
                self.decode_kv_args_table[req.agent_name].gpu_id,
                notif,
            )
            events.append(kv_event)
            # Only the last chunk we need to send the aux data.
            if is_last:
                assert aux_index is not None
                self._send_aux_data_to_peer(req, bootstrap_room, aux_index)
        if is_last:
            del self.transfer_infos[bootstrap_room]
        return events

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _send_aux_data_to_peer(self, req: TransferInfo, bootstrap_room: int, aux_index: int):
        if req.is_dummy():
            return

        aux_item_len = self.kv_args.aux_item_lens[0]
        aux_addr = self.kv_args.aux_data_ptrs[0] + aux_index * aux_item_len
        aux_data = ctypes.string_at(aux_addr, aux_item_len)

        # Send aux data to decode worker
        decode_url = f"tcp://{req.endpoint}:{req.dst_port}"
        sock = zmq.Context().socket(zmq.PUSH)
        try:
            sock.connect(decode_url)
            sock.send_multipart([
                GUARD,
                str(bootstrap_room).encode("ascii"),
                str(req.dst_aux_index).encode("ascii"),
                aux_data
            ])
        finally:
            sock.close()

    def _start_communication_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def communication_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in msg)}"
                )

                assert msg[0] == GUARD, f"First message should be {GUARD}. Foreign traffic?"
                if self.disaggregation_mode == DisaggregationMode.PREFILL:
                    self._handle_bootstrap_message(msg[1:])
                else:
                    self._handle_aux_data_message(msg[1:])

        threading.Thread(target=communication_thread).start()

    def _handle_bootstrap_message(self, msg_parts):
        room = msg_parts[0].decode("ascii")
        agent_name = msg_parts[3].decode("ascii")
        if room == "None":
            # Register new peer and save KV base pointers.
            self._add_remote_peer(
                KVArgsRegisterInfo.from_zmq(msg_parts)
            )
            logger.debug(f"Register KVArgs from {agent_name} successfully")
            return

        room = int(room)
        if room not in self.transfer_infos:
            self.transfer_infos[room] = {}
        self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(msg_parts)
        required_dst_info_num = self.transfer_infos[room][agent_name].required_dst_info_num
        logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
        if len(self.transfer_infos[room]) == required_dst_info_num:
            logger.debug(f"{room=} is bootstrapped")
            self.update_status(room, KVPoll.WaitingForInput)

    def _handle_aux_data_message(self, msg_parts):
        room = int(msg_parts[0].decode("ascii"))
        aux_index = int(msg_parts[1].decode("ascii"))
        aux_data = msg_parts[2]

        aux_item_len = self.kv_args.aux_item_lens[0]
        aux_addr = self.kv_args.aux_data_ptrs[0] + aux_index * aux_item_len
        ctypes.memmove(aux_addr, aux_data, len(aux_data))

        self.transfer_statuses[room].received_aux = True
        logger.debug(f"Received aux data for room {room}, index {aux_index}")


class CupyKVSender(BaseKVSender):

    def __init__(
        self,
        mgr: CupyKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.events = []
        self.has_sent = False
        self.chunk_id = 0
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        # inner state
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        new_events = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
        )
        self.events.extend(new_events)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        return KVPoll.Success if all(event.done for event in self.events) else KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class CupyKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: CupyKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, data_parallel_rank)

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to {self.prefill_server_url} with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent_name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

        self.started_transfer = True

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore
        logger.debug(f"Receiver polling for room {self.bootstrap_room}")
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            logger.debug(f"Receiver concluded for room {self.bootstrap_room}")
            return KVPoll.Success  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            kv_handles = [
                cp.cuda.runtime.ipcGetMemHandle(ptr)
                for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            ]
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent_name.encode("ascii"),
                        packed_aux_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        *kv_handles,
                    ]
                )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class CupyKVBootstrapServer(CommonKVBootstrapServer):
    pass
