from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.managers.cache_controller import (
    make_timing_event_pair,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupPrebuild,
    storage_supports_host_dedup,
)
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


@dataclasses.dataclass(frozen=True)
class _ReservedPoolIndices:
    transfer: PoolTransfer
    attribute: str
    previous_indices: Optional[torch.Tensor]
    indices: Optional[torch.Tensor]
    free_fn: Optional[Callable[[torch.Tensor], Any]] = None


@dataclasses.dataclass(frozen=True)
class _PoolTransferReservation:
    transfers: Optional[list[PoolTransfer]]
    reserved_indices: tuple[_ReservedPoolIndices, ...] = ()


@dataclasses.dataclass(frozen=True)
class HybridWriteReservation:
    """Reversible host allocations for one hybrid D→H transfer."""

    host_indices: torch.Tensor
    device_indices: torch.Tensor
    node_id: int
    priority: Optional[int]
    pool_reservation: _PoolTransferReservation


@dataclasses.dataclass(frozen=True)
class HybridLoadReservation:
    """Reversible device allocations for one hybrid H→D transfer."""

    host_indices: torch.Tensor
    device_indices: torch.Tensor
    node_id: int
    priority: Optional[int]
    pool_reservation: _PoolTransferReservation
    primary_free_fn: Optional[Callable[[torch.Tensor], Any]]


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.storage_hit_count = 0
        self.start_time = time.monotonic()
        super().__init__(
            None,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
        mla_dedup_prebuild: Optional[MLAHostDedupPrebuild] = None,
        enable_mla_hicache_host_dedup: bool = False,
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
            mla_dedup_prebuild=mla_dedup_prebuild,
            enable_mla_hicache_host_dedup=enable_mla_hicache_host_dedup,
        )
        # The base gate ran with storage_backend=None; re-apply it with the
        # real startup backend so broadcast never runs against full
        # (non-dummy) pools.
        if self.mla_broadcast_enabled and not storage_supports_host_dedup(
            startup_storage_backend
        ):
            self._destroy_mla_broadcast_group()
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)
            # The broadcast indexes the MLA KV buffer by layer_num; an
            # expanded transfer layer count (e.g. +Mamba state) would index
            # out of bounds and the extra pools aren't deduped — disable.
            if self.mla_broadcast_enabled:
                logger.info(
                    "Disabling MLA host-dedup broadcast: transfer layer count "
                    "(%d) exceeds the MLA KV layers, so extra hybrid pools "
                    "(e.g. Mamba) are not deduplicated.",
                    self.layer_num,
                )
                self._destroy_mla_broadcast_group()

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            # Dummy pool: nothing to register; this rank never reads L3.
            if getattr(entry.host_pool, "_is_dummy", False):
                continue
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def reserve_write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
        *,
        allow_evict: bool = False,
    ) -> Optional[HybridWriteReservation]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        try:
            pool_reservation = self._reserve_pool_transfers(
                extra_pools,
                alloc_host=True,
                kv_device_indices=device_indices,
                kv_host_indices=host_indices,
                allow_evict=allow_evict,
            )
        except Exception:
            self.mem_pool_host.free(host_indices)
            raise
        if pool_reservation is None:
            self.mem_pool_host.free(host_indices)
            return None
        return HybridWriteReservation(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
            priority=priority,
            pool_reservation=pool_reservation,
        )

    def commit_write(self, reservation: HybridWriteReservation) -> torch.Tensor:
        self.write_queue.append(
            CacheOperation(
                reservation.host_indices,
                reservation.device_indices,
                reservation.node_id,
                reservation.priority,
                pool_transfers=reservation.pool_reservation.transfers,
            )
        )
        self.start_writing()
        return reservation.host_indices

    def abort_write(self, reservation: HybridWriteReservation) -> None:
        try:
            self._rollback_pool_reservation(reservation.pool_reservation)
        finally:
            self.mem_pool_host.free(reservation.host_indices)

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        reservation = self.reserve_write(
            device_indices,
            priority,
            node_id,
            extra_pools,
            allow_evict=True,
        )
        if reservation is None:
            return None
        return self.commit_write(reservation)

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        self.write_queue.clear()
        start_event = device_module.Event()
        finish_event = device_module.Event()
        if self._mla_skip_host_io and not self.has_draft:
            # MLA/DSA dedup: dummy target host pools on this rank; skip D2H.
            start_event.record()
            finish_event.record()
            self.ack_write_queue.append(
                HiCacheAck(start_event, finish_event, op.node_ids)
            )
            return

        # Page-first write-back JIT kernels can keep destination host indices on CPU.
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices = op.host_indices
            device_indices = op.device_indices
            resolved_pool_transfers = op.pool_transfers
        else:
            host_indices, device_indices, resolved_pool_transfers = (
                self.move_hybrid_indices(op)
            )
        if self.has_draft:
            # See HiCacheController._resolve_write_indices: non-src MLA ranks
            # resolve target indices for a dummy pool, while the local draft
            # staged write-back still requires CPU destination indices.
            draft_host_indices, draft_device_indices = self._resolve_write_indices(
                op, self.mem_pool_host_draft
            )
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            # Only the source rank owns deduplicated target pools. Draft KV is
            # rank-local and must still be copied on every rank.
            if not self._mla_skip_host_io:
                self.mem_pool_host.backup_from_device_all_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
            if self.has_draft and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    draft_host_indices,
                    draft_device_indices,
                    self.io_backend,
                )
            finish_event.record()
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def reserve_load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
        *,
        allow_evict: bool = False,
    ) -> Optional[HybridLoadReservation]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
            primary_free_fn = None
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None
            primary_free_fn = full_allocator.free

        try:
            pool_reservation = self._reserve_pool_transfers(
                extra_pools,
                alloc_host=False,
                kv_device_indices=device_indices,
                kv_host_indices=host_indices,
                allow_evict=allow_evict,
            )
        except Exception:
            if primary_free_fn is not None:
                primary_free_fn(device_indices)
            raise
        if pool_reservation is None:
            if primary_free_fn is not None:
                primary_free_fn(device_indices)
            return None
        return HybridLoadReservation(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
            priority=priority,
            pool_reservation=pool_reservation,
            primary_free_fn=primary_free_fn,
        )

    def commit_load(self, reservation: HybridLoadReservation) -> torch.Tensor:
        self.load_queue.append(
            CacheOperation(
                reservation.host_indices,
                reservation.device_indices,
                reservation.node_id,
                reservation.priority,
                pool_transfers=reservation.pool_reservation.transfers,
            )
        )
        return reservation.device_indices

    def abort_load(self, reservation: HybridLoadReservation) -> None:
        try:
            self._rollback_pool_reservation(reservation.pool_reservation)
        finally:
            if reservation.primary_free_fn is not None:
                reservation.primary_free_fn(reservation.device_indices)

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        reservation = self.reserve_load(
            host_indices,
            priority,
            node_id,
            extra_pools,
            allow_evict=True,
        )
        if reservation is None:
            return None
        return self.commit_load(reservation)

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        self.load_queue.clear()
        if self.mla_broadcast_enabled:
            return self._start_loading_mla(producer_id, op)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        ack_start_event, ack_finish_event, timing_enabled = make_timing_event_pair()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            ack_start_event.record()
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                if (
                    self.has_draft
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            ack_finish_event.record()
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                ack_start_event,
                ack_finish_event,
                op.node_ids,
                num_tokens=len(op.device_indices),
                timing_enabled=timing_enabled,
            )
        )
        return producer_id

    def _prepare_mla_source_load(self, op: CacheOperation):
        """Resolve target and extra-pool indices once for layerwise H2D."""
        return self.move_hybrid_indices(op)

    def _load_mla_source_layer(self, source_state, layer_id: int) -> None:
        """Load one target layer and its corresponding extra-pool data."""
        host_indices, device_indices, resolved_pool_transfers = source_state
        self.mem_pool_host.load_to_device_per_layer(
            self.mem_pool_device,
            host_indices,
            device_indices,
            layer_id,
            self.io_backend,
            pool_transfers=resolved_pool_transfers,
        )

    def _record_mla_source_load(self, source_state) -> None:
        host_indices, device_indices, resolved_pool_transfers = source_state
        self._record_transfer_indices_on_stream(
            self.load_stream,
            host_indices,
            device_indices,
            resolved_pool_transfers,
        )

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # Dummy host pools (KV and indexer): no L3 reads on this rank. Must
        # precede super()._page_transfer and the sidecar batch_get below;
        # pool_transfers_done lets this rank pass the all-reduced termination
        # check.
        if self._mla_skip_host_io:
            operation.completed_tokens += len(operation.hash_value) * self.page_size
            operation.pool_transfers_done = True
            return

        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        if (
            operation.pool_transfers
            and not operation.is_terminated()
            and kv_completed_pages == len(operation.hash_value)
        ):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        # MLA KV is replicated across TP ranks and should still be written only
        # by TP0. On follower ranks, only the rank-sharded Mamba/KDA pool is
        # owned by the rank and must be written here. Do not replicate other
        # sidecar pools (for example SWA or indexer state) accidentally.
        backup_transfers = operation.pool_transfers
        if self.backup_skip:
            backup_transfers = [
                transfer
                for transfer in operation.pool_transfers or []
                if transfer.name == PoolName.MAMBA
            ]

        if backup_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(backup_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        if not self.backup_skip:
            super()._page_backup(operation)
        else:
            sidecar_ok = bool(backup_transfers)
            if sidecar_ok:
                for transfer in backup_transfers:
                    result = results.get(transfer.name)
                    if result is None:
                        result = results.get(transfer.name.value)
                    expected = len(transfer.keys or [])
                    if expected == 0 and transfer.host_indices is not None:
                        expected = int(transfer.host_indices.numel())
                    if (
                        not isinstance(result, (list, tuple))
                        or len(result) != expected
                        or not all(bool(ok) for ok in result)
                    ):
                        sidecar_ok = False
                        break
            operation.completed_tokens = (
                len(operation.hash_value) * self.page_size if sidecar_ok else 0
            )

    def backup_thread_func(self):
        """Back up rank-sharded sidecars on every TP rank.

        The base implementation skips the entire operation on non-zero MLA TP
        ranks. That optimization is valid for replicated MLA KV, but not for
        hybrid rank-sharded pools such as Kimi-K3 Mamba state.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_backup(operation)
                self.ack_backup_queue.put(operation)
            except Empty:
                continue

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    @staticmethod
    def _rollback_pool_reservation(
        reservation: _PoolTransferReservation,
    ) -> None:
        first_error = None
        for reserved in reversed(reservation.reserved_indices):
            try:
                if reserved.free_fn is not None:
                    assert reserved.indices is not None
                    reserved.free_fn(reserved.indices)
            except Exception as error:
                if first_error is None:
                    first_error = error
            finally:
                setattr(
                    reserved.transfer,
                    reserved.attribute,
                    reserved.previous_indices,
                )
        if first_error is not None:
            raise first_error

    def _reserve_pool_transfers(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
        *,
        allow_evict: bool,
    ) -> Optional[_PoolTransferReservation]:
        """Reserve missing pool indices without queueing any transfer."""
        if not extra_pools:
            return _PoolTransferReservation(transfers=None)
        reserved_indices: list[_ReservedPoolIndices] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            rollback_indices = tuple(reserved_indices)
            reserved_indices.clear()
            self._rollback_pool_reservation(
                _PoolTransferReservation(
                    transfers=extra_pools,
                    reserved_indices=rollback_indices,
                )
            )

        try:
            for pool in extra_pools:
                entry = self.mem_pool_host.entry_map.get(pool.name)
                if entry is None:
                    rollback_allocated()
                    return None
                if pool.indices_from_pool is not None:
                    derived_transfers.append(pool)
                    continue
                if alloc_host:
                    if pool.host_indices is not None or pool.device_indices is None:
                        continue
                    alloc_fn = entry.host_pool.alloc
                    free_fn = entry.host_pool.free
                    evict_fn = entry.host_evict_fn
                    size = len(pool.device_indices)
                else:
                    if pool.device_indices is not None or pool.host_indices is None:
                        continue
                    # Overrides support raw device pools such as SWA.
                    alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                    free_fn = entry.device_free_fn or entry.device_pool.free
                    evict_fn = entry.device_evict_fn
                    size = len(pool.host_indices)
                indices = alloc_fn(size)
                if indices is None and allow_evict and evict_fn:
                    evict_fn(size)
                    indices = alloc_fn(size)
                if indices is None:
                    rollback_allocated()
                    return None
                attribute = "host_indices" if alloc_host else "device_indices"
                previous_indices = getattr(pool, attribute)
                setattr(pool, attribute, indices)
                reserved_indices.append(
                    _ReservedPoolIndices(
                        transfer=pool,
                        attribute=attribute,
                        previous_indices=previous_indices,
                        indices=indices,
                        free_fn=free_fn,
                    )
                )

            # Assign indices to deferred pools from their source.
            for pool in derived_transfers:
                if pool.indices_from_pool == PoolName.KV:
                    source_indices = (
                        ("host_indices", kv_host_indices),
                        ("device_indices", kv_device_indices),
                    )
                else:
                    source = next(
                        (
                            transfer
                            for transfer in extra_pools
                            if transfer.indices_from_pool is None
                            and transfer.name == pool.indices_from_pool
                        ),
                        None,
                    )
                    if source is None:
                        rollback_allocated()
                        return None
                    source_indices = (
                        ("host_indices", source.host_indices),
                        ("device_indices", source.device_indices),
                    )
                for attribute, indices in source_indices:
                    previous_indices = getattr(pool, attribute)
                    setattr(pool, attribute, indices)
                    reserved_indices.append(
                        _ReservedPoolIndices(
                            transfer=pool,
                            attribute=attribute,
                            previous_indices=previous_indices,
                            indices=indices,
                        )
                    )
        except Exception:
            try:
                rollback_allocated()
            except Exception:
                logger.exception("Failed to fully roll back pool reservation")
            raise

        return _PoolTransferReservation(
            transfers=extra_pools,
            reserved_indices=tuple(reserved_indices),
        )
