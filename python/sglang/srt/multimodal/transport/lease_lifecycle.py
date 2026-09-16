"""Request-scoped ownership helpers for the lease pool on the producer side."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Iterator

import torch

from sglang.srt.multimodal.transport.cuda_ipc import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)
from sglang.srt.multimodal.transport.memory_pool import CONTROL_WORD_BYTES

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs

logger = logging.getLogger(__name__)

TRANSPORT_FIELDS = ("feature", "precomputed_embeddings")


def iter_transport_proxies(
    mm_items,
) -> Iterator[tuple[MultimodalDataItem, str, CudaIpcTensorTransportProxy]]:
    for item in mm_items:
        for field in TRANSPORT_FIELDS:
            value = getattr(item, field, None)
            if isinstance(value, CudaIpcTensorTransportProxy):
                yield item, field, value


def pool_owns_proxy(pool: MmItemMemoryPool, proxy) -> bool:
    return tuple(proxy.proxy_state["ipc_extra"]["pool_handle"]) == tuple(
        pool._pool_ipc_handle
    )


def cancel_proxies(pool, proxies, *, context: str) -> int:
    cancelled = 0
    seen = set()
    for proxy in proxies:
        if id(proxy) in seen:
            continue
        seen.add(id(proxy))
        if not pool_owns_proxy(pool, proxy):
            continue
        try:
            pool.cancel_proxy(proxy)
        except Exception as exc:
            logger.warning(
                f"[lease_lifecycle] cancel_proxy failed ({context}): {exc!r}"
            )
            continue
        cancelled += 1
    return cancelled


def cancel_undispatched_proxies(pool, mm_items, *, context: str) -> int:
    cancelled = 0
    fields_by_proxy = {}
    for item, field, proxy in iter_transport_proxies(mm_items):
        fields_by_proxy.setdefault(id(proxy), (proxy, []))[1].append((item, field))

    for proxy, fields in fields_by_proxy.values():
        if cancel_proxies(pool, [proxy], context=context):
            for item, field in fields:
                setattr(item, field, None)
            cancelled += 1
    return cancelled


def copy_lease_to_cpu(pool, proxy) -> torch.Tensor:
    ipc_extra = proxy.proxy_state["ipc_extra"]
    inner = pool._pool
    slot_stride = inner.control_words_per_slot * CONTROL_WORD_BYTES
    slot = proxy.ready_byte_offset // slot_stride
    with inner._lock:
        lease = inner._occupied.get(slot)
        if (
            lease is None
            or lease.generation != proxy.generation
            or lease.start != ipc_extra["pool_byte_offset"]
        ):
            raise RuntimeError(
                f"CUDA IPC lease slot={slot} gen={proxy.generation} "
                "is no longer active; cannot clone"
            )
        with torch.cuda.device(pool.device_id):
            raw = pool.memory_pool[lease.start : lease.start + lease.nbytes].clone()
    return raw.view(ipc_extra["recons_dtype"]).reshape(ipc_extra["recons_shape"]).cpu()


def detach_proxies_for_clones(pool, mm_inputs: MultimodalInputs) -> MultimodalInputs:
    clone_inputs = copy.copy(mm_inputs)
    clone_inputs.mm_items = [copy.copy(item) for item in mm_inputs.mm_items]
    copied = {}
    for item in clone_inputs.mm_items:
        for field in TRANSPORT_FIELDS:
            proxy = getattr(item, field, None)
            if not isinstance(proxy, CudaIpcTensorTransportProxy):
                continue
            if not pool_owns_proxy(pool, proxy):
                continue
            proxy_id = id(proxy)
            if proxy_id not in copied:
                copied[proxy_id] = copy_lease_to_cpu(pool, proxy)
            setattr(item, field, copied[proxy_id])
    return clone_inputs
