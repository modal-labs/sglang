"""Failed HiCache write reservations must not mutate host allocator state.

Regression test for the HiCache V3 divergence fail-stop seen in production
("peer mirror allocation does not match published placement"): the host free
lists are order-sensitive FIFOs (alloc pops the front, free rejoins at the
back via a deferred release list), so ``reserve_write``'s old
alloc-then-rollback on a failed extra-pool allocation rotated rank 0's KV
free list with no record on the wire. Peers replay only successful PLACEs,
so the next successful reservation on rank 0 drew different slots than every
peer's mirror replay and the kv_crc check fail-stopped all ranks.

The fix prechecks availability (exact under the authority lock, which
serializes every allocator mutation) and returns None without allocating.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.test_utils import CustomTestCase


class _FifoHostPool:
    """Order-sensitive FIFO free list matching HostKVCache alloc/free semantics."""

    def __init__(self, slots):
        self.free_slots = list(slots)
        self.release_slots = []
        self.alloc_calls = 0

    def available_size(self):
        return len(self.free_slots) + len(self.release_slots)

    def alloc(self, need_size):
        self.alloc_calls += 1
        if need_size > len(self.free_slots):
            self.free_slots += self.release_slots
            self.release_slots = []
        if need_size > len(self.free_slots):
            return None
        out = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, indices):
        self.release_slots += indices.tolist()

    def order_signature(self):
        return tuple(self.free_slots), tuple(self.release_slots)


def _controller(kv_slots, mamba_slots):
    controller = object.__new__(HybridCacheController)
    kv_pool = _FifoHostPool(kv_slots)
    mamba_pool = _FifoHostPool(mamba_slots)
    kv_pool.entry_map = {
        "mamba": SimpleNamespace(
            host_pool=mamba_pool,
            host_evict_fn=None,
            device_pool=None,
            device_alloc_fn=None,
            device_free_fn=None,
            device_evict_fn=None,
        )
    }
    controller.mem_pool_host = kv_pool
    return controller, kv_pool, mamba_pool


def _mamba_transfer(n=1):
    return PoolTransfer(name="mamba", device_indices=torch.arange(n))


class TestReservationPrecheck(CustomTestCase):
    def test_failed_reservation_is_side_effect_free(self):
        # KV has room, mamba pool is exhausted: the pre-fix path allocated KV
        # first and rolled it back (front slots rotated to the back).
        controller, kv_pool, mamba_pool = _controller(range(16), [])
        before = kv_pool.order_signature()

        reservation = controller.reserve_write(
            torch.arange(4), extra_pools=[_mamba_transfer()], allow_evict=False
        )

        self.assertIsNone(reservation)
        self.assertEqual(kv_pool.order_signature(), before)
        self.assertEqual(kv_pool.alloc_calls, 0)
        self.assertEqual(mamba_pool.alloc_calls, 0)

    def test_kv_exhaustion_is_side_effect_free(self):
        controller, kv_pool, _ = _controller(range(2), range(4))
        reservation = controller.reserve_write(
            torch.arange(4), extra_pools=[_mamba_transfer()], allow_evict=False
        )
        self.assertIsNone(reservation)
        self.assertEqual(kv_pool.alloc_calls, 0)

    def test_successful_reservation_allocates_front_slots(self):
        controller, kv_pool, mamba_pool = _controller(range(16), range(4))
        reservation = controller.reserve_write(
            torch.arange(4), extra_pools=[_mamba_transfer()], allow_evict=False
        )
        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.host_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(kv_pool.alloc_calls, 1)
        self.assertEqual(mamba_pool.alloc_calls, 1)

    def test_same_pool_demands_are_summed(self):
        # Two transfers of 1 against a single free mamba slot must be
        # predicted as a failure (per-transfer checks would each pass).
        controller, kv_pool, _ = _controller(range(16), [0])
        reservation = controller.reserve_write(
            torch.arange(4),
            extra_pools=[_mamba_transfer(), _mamba_transfer()],
            allow_evict=False,
        )
        self.assertIsNone(reservation)
        self.assertEqual(kv_pool.alloc_calls, 0)

    def test_unknown_pool_is_predicted(self):
        controller, kv_pool, _ = _controller(range(16), range(4))
        reservation = controller.reserve_write(
            torch.arange(4),
            extra_pools=[PoolTransfer(name="nope", device_indices=torch.arange(1))],
            allow_evict=False,
        )
        self.assertIsNone(reservation)
        self.assertEqual(kv_pool.alloc_calls, 0)


if __name__ == "__main__":
    unittest.main()
