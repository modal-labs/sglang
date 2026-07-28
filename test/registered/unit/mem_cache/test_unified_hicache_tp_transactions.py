"""CPU tests for transactional Unified HiCache TP/PP transfers."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
    HybridLoadReservation,
    HybridWriteReservation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _indices(start: int, count: int) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=torch.int64)


class _TrackingPool:
    def __init__(self, start: int, *, fail: bool = False):
        self.next_index = start
        self.fail = fail
        self.alloc_calls: list[int] = []
        self.freed: list[torch.Tensor] = []

    def alloc(self, size: int):
        self.alloc_calls.append(size)
        if self.fail:
            return None
        result = _indices(self.next_index, size)
        self.next_index += size
        return result

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())


def _pool_entry(host_pool, device_pool):
    return SimpleNamespace(
        host_pool=host_pool,
        device_pool=device_pool,
        device_alloc_fn=None,
        device_free_fn=None,
        host_evict_fn=mock.Mock(),
        device_evict_fn=mock.Mock(),
    )


def _make_controller(*, fail_host_pool=None, fail_device_pool=None):
    pools = SimpleNamespace(
        primary_host=_TrackingPool(100),
        swa_host=_TrackingPool(200, fail=fail_host_pool == PoolName.SWA),
        mamba_host=_TrackingPool(300, fail=fail_host_pool == PoolName.MAMBA),
        primary_device=_TrackingPool(400),
        swa_device=_TrackingPool(500, fail=fail_device_pool == PoolName.SWA),
        mamba_device=_TrackingPool(600, fail=fail_device_pool == PoolName.MAMBA),
    )
    entries = {
        PoolName.SWA: _pool_entry(pools.swa_host, pools.swa_device),
        PoolName.MAMBA: _pool_entry(pools.mamba_host, pools.mamba_device),
    }
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        alloc=pools.primary_host.alloc,
        free=pools.primary_host.free,
        entry_map=entries,
    )
    controller.mem_pool_device_allocator = SimpleNamespace(
        full_attn_allocator=pools.primary_device
    )
    controller.device = torch.device("cpu")
    controller.write_queue = []
    controller.load_queue = []
    controller.start_writing = mock.Mock()
    return controller, pools, entries


def _write_transfers():
    return [
        PoolTransfer(name=PoolName.SWA, device_indices=_indices(10, 4)),
        PoolTransfer(name=PoolName.MAMBA, device_indices=_indices(20, 1)),
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            indices_from_pool=PoolName.KV,
        ),
    ]


def _load_transfers():
    mamba_host = _indices(30, 1)
    return [
        PoolTransfer(name=PoolName.SWA, host_indices=_indices(10, 4)),
        PoolTransfer(name=PoolName.MAMBA, host_indices=mamba_host),
        # This request-owned CoW slot is preallocated by MambaComponent and is
        # therefore not owned by the controller transaction.
        PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=mamba_host,
            device_indices=_indices(900, 1),
        ),
        PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            indices_from_pool=PoolName.KV,
        ),
    ]


def _assert_one_free(test, pool: _TrackingPool, expected: torch.Tensor):
    test.assertEqual(len(pool.freed), 1)
    torch.testing.assert_close(pool.freed[0], expected)


class TestHybridControllerWriteTransaction(unittest.TestCase):
    def test_reserve_then_abort_releases_primary_swa_mamba_and_bindings(self):
        controller, pools, entries = _make_controller()
        transfers = _write_transfers()

        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        self.assertIsInstance(reservation, HybridWriteReservation)
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        for entry in entries.values():
            entry.host_evict_fn.assert_not_called()
        self.assertIsNotNone(transfers[0].host_indices)
        self.assertIsNotNone(transfers[1].host_indices)
        self.assertIs(transfers[2].host_indices, reservation.host_indices)

        controller.abort_write(reservation)

        _assert_one_free(self, pools.primary_host, reservation.host_indices)
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertIsNone(transfers[2].host_indices)
        self.assertIsNone(transfers[2].device_indices)
        self.assertEqual(controller.write_queue, [])

    def test_commit_populates_real_queue_without_releasing_reservations(self):
        controller, pools, _ = _make_controller()
        transfers = _write_transfers()
        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        host_indices = controller.commit_write(reservation)

        self.assertIs(host_indices, reservation.host_indices)
        self.assertEqual(len(controller.write_queue), 1)
        operation = controller.write_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [7])
        self.assertIs(operation.pool_transfers, transfers)
        controller.start_writing.assert_called_once()
        self.assertEqual(pools.primary_host.freed, [])
        self.assertEqual(pools.swa_host.freed, [])
        self.assertEqual(pools.mamba_host.freed, [])

    def test_partial_aux_failure_rolls_back_without_eviction_or_queueing(self):
        controller, pools, entries = _make_controller(fail_host_pool=PoolName.MAMBA)
        transfers = _write_transfers()

        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        self.assertIsNone(reservation)
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        for entry in entries.values():
            entry.host_evict_fn.assert_not_called()


class TestHybridControllerLoadTransaction(unittest.TestCase):
    def test_reserve_then_abort_releases_owned_primary_swa_mamba_only(self):
        controller, pools, entries = _make_controller()
        transfers = _load_transfers()
        request_mamba_indices = transfers[2].device_indices

        reservation = controller.reserve_load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        self.assertIsInstance(reservation, HybridLoadReservation)
        self.assertEqual(controller.load_queue, [])
        for entry in entries.values():
            entry.device_evict_fn.assert_not_called()
        self.assertIsNotNone(transfers[0].device_indices)
        self.assertIsNotNone(transfers[1].device_indices)
        self.assertIs(transfers[3].device_indices, reservation.device_indices)

        controller.abort_load(reservation)

        _assert_one_free(self, pools.primary_device, reservation.device_indices)
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        _assert_one_free(self, pools.mamba_device, _indices(600, 1))
        self.assertIsNone(transfers[0].device_indices)
        self.assertIsNone(transfers[1].device_indices)
        self.assertIs(transfers[2].device_indices, request_mamba_indices)
        self.assertIsNone(transfers[3].host_indices)
        self.assertIsNone(transfers[3].device_indices)
        self.assertEqual(controller.load_queue, [])

    def test_commit_populates_real_queue_without_releasing_reservations(self):
        controller, pools, _ = _make_controller()
        transfers = _load_transfers()
        reservation = controller.reserve_load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        device_indices = controller.commit_load(reservation)

        self.assertIs(device_indices, reservation.device_indices)
        self.assertEqual(len(controller.load_queue), 1)
        operation = controller.load_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [9])
        self.assertIs(operation.pool_transfers, transfers)
        self.assertEqual(pools.primary_device.freed, [])
        self.assertEqual(pools.swa_device.freed, [])
        self.assertEqual(pools.mamba_device.freed, [])

    def test_partial_aux_failure_rolls_back_without_eviction_or_queueing(self):
        controller, pools, entries = _make_controller(fail_device_pool=PoolName.MAMBA)
        transfers = _load_transfers()

        reservation = controller.reserve_load(
            _indices(0, 4), node_id=9, extra_pools=transfers
        )

        self.assertIsNone(reservation)
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertIsNone(transfers[0].device_indices)
        self.assertIsNone(transfers[1].device_indices)
        self.assertEqual(controller.load_queue, [])
        for entry in entries.values():
            entry.device_evict_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
