"""Bounded per-slot generation guards for stream-ordered CUDA IPC leases."""

import logging
from collections.abc import Hashable

from sglang.srt.multimodal.transport.memory_pool import CONTROL_WORD_BYTES

logger = logging.getLogger(__name__)

_max_acked_gen: dict[tuple[Hashable, int], int] = {}
stats = {"stale_read_refused": 0, "stale_write_refused": 0}
_warned_reads: dict[tuple[Hashable, int], int] = {}


def _pool_key(proxy) -> Hashable:
    pool_handle = proxy.proxy_state["ipc_extra"]["pool_handle"]
    try:
        return tuple(pool_handle)
    except TypeError:
        return pool_handle


def _key(proxy) -> tuple[Hashable, int]:
    return (_pool_key(proxy), proxy.ready_byte_offset)


def _slot(proxy) -> int:
    return proxy.ready_byte_offset // (
        (1 + proxy.total_consumer_count) * CONTROL_WORD_BYTES
    )


def check_read(proxy, *, rank) -> None:
    key = _key(proxy)
    recorded = _max_acked_gen.get(key)
    if recorded is None or recorded < proxy.generation:
        return

    stats["stale_read_refused"] += 1
    if _warned_reads.get(key) != proxy.generation:
        _warned_reads[key] = proxy.generation
        logger.warning(
            "Refused stale CUDA IPC lease read: slot=%s generation=%s "
            "recorded=%s rank=%s",
            _slot(proxy),
            proxy.generation,
            recorded,
            rank,
        )
    raise StaleLeaseError(
        f"CUDA IPC lease slot {_slot(proxy)} generation {proxy.generation} "
        f"already acknowledged (max acked {recorded}) on rank {rank}"
    )


def check_write(proxy, *, rank) -> bool:
    key = _key(proxy)
    recorded = _max_acked_gen.get(key)
    if recorded is not None and recorded >= proxy.generation:
        stats["stale_write_refused"] += 1
        logger.debug(
            "Refused stale CUDA IPC lease write: slot=%s generation=%s "
            "recorded=%s rank=%s",
            _slot(proxy),
            proxy.generation,
            recorded,
            rank,
        )
        return False

    return True


def record_write(proxy, *, rank) -> None:
    key = _key(proxy)
    current = _max_acked_gen.get(key)
    if current is None or current < proxy.generation:
        _max_acked_gen[key] = proxy.generation


def check_and_record_write(proxy, *, rank) -> bool:
    if not check_write(proxy, rank=rank):
        return False
    record_write(proxy, rank=rank)
    return True


def reset() -> None:
    _max_acked_gen.clear()
    _warned_reads.clear()
    stats["stale_read_refused"] = 0
    stats["stale_write_refused"] = 0


class StaleLeaseError(RuntimeError):
    """Raised when a lease generation was already acknowledged locally."""
