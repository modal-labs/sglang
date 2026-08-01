from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Union,
)

import zmq
from torch.distributed import barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.mem_cache.hybrid_cache.hicache_authority import HiCacheRecordBatch
from sglang.srt.utils import (
    broadcast_pyobj,
    point_to_point_pyobj,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy]
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: ParallelState
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_batch: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None
    # HiCache V3 authority: the per-step rank0->peers request broadcast below
    # is the publication carrier for cache control-plane records (bytes on an
    # existing collective; never a new collective).
    hicache_bus: Any = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_batch()):
                # Flush rule (docs/HICACHE_V3.md): pending records must ride
                # a broadcast within N=16 steps. allow_skip() is a pure
                # function of a local counter that advances identically on
                # every rank (the skipper decision itself is rank-symmetric),
                # so every rank enters the forced broadcast together.
                if self.hicache_bus is None or self.hicache_bus.allow_skip():
                    return []

        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _pull_raw_reqs(self) -> Optional[List]:
        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = sock_recv(self.recv_from_rpc, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                dp_offset = (
                    self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
                )
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                )
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
        if self.server_args.enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.ps.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.ps.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                self.server_args.enable_dp_attention_local_control_broadcast
                or self.server_args.is_ep_scale_joiner
            )
            if _local_ctrl:
                if self.ps.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.ps.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            elif self.ps.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.ps.tp_size != 1:
            bus = self.hicache_bus
            if bus is not None and bus.is_rank0:
                # V3 publication: piggyback the pending cache records on this
                # step's request broadcast. build_publish_batch() also
                # self-applies the records to rank 0's tree, so every rank
                # mutates mirrored state at the same loop position.
                record_batch = bus.build_publish_batch()
                if record_batch is not None:
                    recv_reqs = list(recv_reqs or []) + [record_batch]
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
            if bus is not None:
                bus.note_broadcast()
                record_batches = [
                    r for r in recv_reqs if isinstance(r, HiCacheRecordBatch)
                ]
                if record_batches:
                    recv_reqs = [
                        r for r in recv_reqs if not isinstance(r, HiCacheRecordBatch)
                    ]
                    if not bus.is_rank0:
                        for record_batch in record_batches:
                            bus.apply_batch(record_batch)
        return recv_reqs

    def unwrap_pickle_wrapper(self, recv_reqs: Optional[List]) -> None:
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                req.unwrap_pickle_fields()
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                for sub_req in req:
                    sub_req.unwrap_pickle_fields()

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        if (
            self.ps.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                if error_code is None:
                    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                elif isinstance(error_code, HTTPStatus):
                    status_code = error_code
                else:
                    status_code = HTTPStatus(int(error_code))
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        if recv_reqs:
            if self.model_config.is_multimodal and has_shm_features(recv_reqs):
                # The broadcast source returns with its original objects while
                # peer ranks may still be unpickling ShmPointerMMData
                # (-> shm_open).  Synchronize the same CPU groups that carried
                # SHM-backed work requests before materialize() unlinks them.
                if self.server_args.enable_dp_attention:
                    if self.ps.attn_tp_size > 1:
                        barrier(group=self.attn_tp_cpu_group)
                    if self.ps.attn_cp_size > 1:
                        barrier(group=self.attn_cp_cpu_group)
                elif self.ps.tp_size > 1:
                    barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs
