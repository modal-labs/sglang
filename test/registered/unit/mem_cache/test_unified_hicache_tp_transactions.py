"""CPU tests for transactional Unified HiCache TP/PP transfers."""

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictResult, IncLockRefResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
    HybridLoadReservation,
    HybridWriteReservation,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    MambaComponent,
    PrepareLoadBackResult,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _indices(start: int, count: int) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=torch.int64)


class _TrackingPool:
    def __init__(
        self,
        start: int,
        *,
        fail: bool = False,
        raise_on_alloc: bool = False,
        raise_on_free: bool = False,
    ):
        self.next_index = start
        self.fail = fail
        self.raise_on_alloc = raise_on_alloc
        self.raise_on_free = raise_on_free
        self.alloc_calls: list[int] = []
        self.freed: list[torch.Tensor] = []

    def alloc(self, size: int):
        self.alloc_calls.append(size)
        if self.raise_on_alloc:
            raise RuntimeError("injected allocation failure")
        if self.fail:
            return None
        result = _indices(self.next_index, size)
        self.next_index += size
        return result

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())
        if self.raise_on_free:
            raise RuntimeError("injected free failure")


def _pool_entry(host_pool, device_pool):
    return SimpleNamespace(
        host_pool=host_pool,
        device_pool=device_pool,
        device_alloc_fn=None,
        device_free_fn=None,
        host_evict_fn=mock.Mock(),
        device_evict_fn=mock.Mock(),
    )


def _make_controller(
    *,
    fail_host_pool=None,
    fail_device_pool=None,
    raise_host_pool=None,
    raise_device_pool=None,
    raise_free_host_pool=None,
    raise_free_device_pool=None,
):
    pools = SimpleNamespace(
        primary_host=_TrackingPool(100),
        swa_host=_TrackingPool(
            200,
            fail=fail_host_pool == PoolName.SWA,
            raise_on_alloc=raise_host_pool == PoolName.SWA,
            raise_on_free=raise_free_host_pool == PoolName.SWA,
        ),
        mamba_host=_TrackingPool(
            300,
            fail=fail_host_pool == PoolName.MAMBA,
            raise_on_alloc=raise_host_pool == PoolName.MAMBA,
            raise_on_free=raise_free_host_pool == PoolName.MAMBA,
        ),
        primary_device=_TrackingPool(400),
        swa_device=_TrackingPool(
            500,
            fail=fail_device_pool == PoolName.SWA,
            raise_on_alloc=raise_device_pool == PoolName.SWA,
            raise_on_free=raise_free_device_pool == PoolName.SWA,
        ),
        mamba_device=_TrackingPool(
            600,
            fail=fail_device_pool == PoolName.MAMBA,
            raise_on_alloc=raise_device_pool == PoolName.MAMBA,
            raise_on_free=raise_free_device_pool == PoolName.MAMBA,
        ),
    )
    entries = {
        PoolName.SWA: _pool_entry(pools.swa_host, pools.swa_device),
        PoolName.MAMBA: _pool_entry(pools.mamba_host, pools.mamba_device),
        PoolName.DEEPSEEK_V4_C4: _pool_entry(None, None),
    }
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        alloc=pools.primary_host.alloc,
        free=pools.primary_host.free,
        available_size=mock.Mock(return_value=1_000),
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


def _component(component_type: ComponentType, transfers=None):
    component = mock.Mock()
    component.component_type = component_type
    component.build_hicache_transfers.return_value = transfers
    component.prepare_load_back.return_value = PrepareLoadBackResult()
    return component


def _make_unified_cache(controller, components):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = controller
    cache.tp_world_size = 2
    cache.pp_rank = 0
    cache.pp_size = 1
    cache.pp_group = None
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.tp_group = object()
    cache._components_tuple = tuple(components)
    cache.components = {component.component_type: component for component in components}
    cache.sidecar_pool_specs = []
    cache.root_node = UnifiedTreeNode(tuple(cache.components))
    return cache


def _make_node(cache):
    node = UnifiedTreeNode(tuple(cache.components))
    node.parent = cache.root_node
    return node


def _install_consensus(cache, *, peer_succeeded: bool, before_reduce):
    def reduce(tensor, op):
        before_reduce()
        if not peer_succeeded:
            tensor.zero_()

    cache._all_reduce_attn_groups = mock.Mock(side_effect=reduce)


def _assert_min_reduce(test, cache):
    cache._all_reduce_attn_groups.assert_called_once()
    _, op = cache._all_reduce_attn_groups.call_args.args
    test.assertEqual(op, torch.distributed.ReduceOp.MIN)


def _make_mamba_component(cache, node, req, allocator):
    component = SimpleNamespace(
        component_type=ComponentType.MAMBA,
        cache=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=allocator),
            evict=mock.Mock(),
        ),
        commit_hicache_transfer=mock.Mock(),
    )
    component.prepare_load_back = MethodType(
        MambaComponent.prepare_load_back, component
    )
    component.finalize_load_back = MethodType(
        MambaComponent.finalize_load_back, component
    )
    tree_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=node.component_data[ComponentType.MAMBA].host_value,
        nodes_to_load=[node],
    )
    request_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=node.component_data[ComponentType.MAMBA].host_value,
    )

    def build_hicache_transfers(*_args, **_kwargs):
        if req.mamba_pool_idx is not None:
            request_transfer.device_indices = req.mamba_pool_idx.reshape(1)
        return [tree_transfer, request_transfer]

    component.build_hicache_transfers = mock.Mock(side_effect=build_hicache_transfers)
    return component, (tree_transfer, request_transfer)


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

    def test_raising_aux_allocator_rolls_back_primary_and_prior_aux(self):
        controller, pools, _ = _make_controller(raise_host_pool=PoolName.MAMBA)
        transfers = _write_transfers()

        with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
            controller.reserve_write(_indices(0, 4), node_id=7, extra_pools=transfers)

        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(controller.write_queue, [])

    def test_aux_free_error_still_releases_primary_and_other_aux(self):
        controller, pools, _ = _make_controller(raise_free_host_pool=PoolName.SWA)
        transfers = _write_transfers()
        reservation = controller.reserve_write(
            _indices(0, 4), node_id=7, extra_pools=transfers
        )

        with self.assertRaisesRegex(RuntimeError, "injected free failure"):
            controller.abort_write(reservation)

        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(controller.write_queue, [])

    def test_unregistered_explicit_or_derived_pool_aborts_whole_reservation(self):
        missing_transfers = (
            PoolTransfer(name=PoolName.INDEXER, device_indices=_indices(20, 4)),
            PoolTransfer(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
            ),
        )
        for missing in missing_transfers:
            with self.subTest(derived=missing.indices_from_pool is not None):
                controller, pools, _ = _make_controller()
                transfers = [
                    PoolTransfer(name=PoolName.SWA, device_indices=_indices(10, 4)),
                    missing,
                ]

                reservation = controller.reserve_write(
                    _indices(0, 4), node_id=7, extra_pools=transfers
                )

                self.assertIsNone(reservation)
                _assert_one_free(self, pools.primary_host, _indices(100, 4))
                _assert_one_free(self, pools.swa_host, _indices(200, 4))
                self.assertIsNone(transfers[0].host_indices)
                self.assertEqual(controller.write_queue, [])


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

    def test_raising_aux_allocator_rolls_back_primary_and_prior_aux(self):
        controller, pools, _ = _make_controller(raise_device_pool=PoolName.MAMBA)
        transfers = _load_transfers()

        with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
            controller.reserve_load(_indices(0, 4), node_id=9, extra_pools=transfers)

        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertIsNone(transfers[0].device_indices)
        self.assertIsNone(transfers[1].device_indices)
        self.assertEqual(controller.load_queue, [])


class TestUnifiedWriteConsensus(unittest.TestCase):
    def _setup(self, *, peer_succeeded: bool, raise_host_pool=None):
        controller, pools, entries = _make_controller(raise_host_pool=raise_host_pool)
        swa_transfer, mamba_transfer = _write_transfers()[:2]
        base = _component(ComponentType.FULL)
        swa = _component(ComponentType.SWA, [swa_transfer])
        mamba = _component(ComponentType.MAMBA, [mamba_transfer])
        cache = _make_unified_cache(controller, [base, swa, mamba])
        cache.evict_host = mock.Mock()
        cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(delta=0))
        cache._track_write_through_node = mock.Mock()
        node = _make_node(cache)
        node.component_data[BASE_COMPONENT_TYPE].value = _indices(0, 4)

        def before_reduce():
            self.assertEqual(controller.write_queue, [])
            controller.start_writing.assert_not_called()
            cache.evict_host.assert_not_called()
            cache.inc_lock_ref.assert_not_called()
            cache._track_write_through_node.assert_not_called()
            base.commit_hicache_transfer.assert_not_called()
            swa.commit_hicache_transfer.assert_not_called()
            mamba.commit_hicache_transfer.assert_not_called()
            self.assertEqual(pools.primary_host.alloc_calls, [4])
            self.assertEqual(pools.swa_host.alloc_calls, [4])
            self.assertEqual(pools.mamba_host.alloc_calls, [1])

        _install_consensus(
            cache,
            peer_succeeded=peer_succeeded,
            before_reduce=before_reduce,
        )
        return (
            cache,
            node,
            controller,
            pools,
            entries,
            base,
            swa,
            mamba,
        )

    def test_peer_failure_aborts_every_allocation_without_queue_or_mutation(self):
        (
            cache,
            node,
            controller,
            pools,
            entries,
            base,
            swa,
            mamba,
        ) = self._setup(peer_succeeded=False)

        written = cache.write_backup(node)

        self.assertEqual(written, 0)
        _assert_min_reduce(self, cache)
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        for entry in entries.values():
            entry.host_evict_fn.assert_not_called()
        cache.evict_host.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        cache._track_write_through_node.assert_not_called()
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()

    def test_success_queues_then_commits_tree_state(self):
        cache, node, controller, pools, _, base, swa, mamba = self._setup(
            peer_succeeded=True
        )

        written = cache.write_backup(node)

        self.assertEqual(written, 4)
        _assert_min_reduce(self, cache)
        self.assertEqual(len(controller.write_queue), 1)
        operation = controller.write_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [node.id])
        controller.start_writing.assert_called_once()
        self.assertEqual(pools.primary_host.freed, [])
        self.assertEqual(pools.swa_host.freed, [])
        self.assertEqual(pools.mamba_host.freed, [])
        base.commit_hicache_transfer.assert_called_once()
        swa.commit_hicache_transfer.assert_called_once()
        mamba.commit_hicache_transfer.assert_called_once()
        cache.inc_lock_ref.assert_called_once_with(node)
        cache._track_write_through_node.assert_called_once()

    def test_local_allocator_exception_is_cleaned_then_reduced_as_failure(self):
        cache, node, controller, pools, _, base, swa, mamba = self._setup(
            peer_succeeded=True,
            raise_host_pool=PoolName.MAMBA,
        )

        written = cache.write_backup(node)

        self.assertEqual(written, 0)
        _assert_min_reduce(self, cache)
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        self.assertEqual(pools.mamba_host.freed, [])
        self.assertEqual(controller.write_queue, [])
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()

    def test_consensus_exception_aborts_every_local_reservation(self):
        cache, node, controller, pools, _, base, swa, mamba = self._setup(
            peer_succeeded=True
        )
        cache._all_reduce_attn_groups.side_effect = RuntimeError(
            "injected reduce failure"
        )

        with self.assertRaisesRegex(RuntimeError, "injected reduce failure"):
            cache.write_backup(node)

        _assert_min_reduce(self, cache)
        _assert_one_free(self, pools.primary_host, _indices(100, 4))
        _assert_one_free(self, pools.swa_host, _indices(200, 4))
        _assert_one_free(self, pools.mamba_host, _indices(300, 1))
        self.assertEqual(controller.write_queue, [])
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()

    def test_multi_rank_write_back_is_rejected_without_side_effects(self):
        cache, node, controller, _, _, base, swa, mamba = self._setup(
            peer_succeeded=False
        )

        with self.assertRaisesRegex(RuntimeError, "cannot run independently"):
            cache.write_backup(node, write_back=True)

        cache._all_reduce_attn_groups.assert_not_called()
        self.assertEqual(controller.write_queue, [])
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        cache._track_write_through_node.assert_not_called()

    def test_multi_rank_write_back_config_is_rejected_at_init(self):
        controller, _, _ = _make_controller()
        cache = _make_unified_cache(
            controller,
            [_component(ComponentType.FULL)],
        )

        with self.assertRaisesRegex(ValueError, "not TP/PP safe"):
            cache.init_hicache(
                SimpleNamespace(hicache_write_policy="write_back"),
                params=None,
            )


class TestUnifiedLoadConsensus(unittest.TestCase):
    def _setup(
        self,
        *,
        peer_succeeded: bool,
        fail_request_mamba: bool = False,
        raise_device_pool=None,
    ):
        controller, pools, entries = _make_controller(
            raise_device_pool=raise_device_pool
        )
        request_mamba = _TrackingPool(700, fail=fail_request_mamba)
        base = _component(ComponentType.FULL)
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(20, 4),
        )
        swa = _component(ComponentType.SWA, [swa_transfer])
        cache = _make_unified_cache(controller, [base, swa])
        node = _make_node(cache)
        node.component_data[ComponentType.FULL].host_value = _indices(10, 4)
        node.component_data[ComponentType.SWA].host_value = swa_transfer.host_indices
        node.component_data[ComponentType.MAMBA].host_value = _indices(30, 1)
        req = SimpleNamespace(mamba_pool_idx=None)
        mamba, mamba_transfers = _make_mamba_component(cache, node, req, request_mamba)
        cache._components_tuple = (base, swa, mamba)
        cache.components[ComponentType.MAMBA] = mamba
        base.build_hicache_transfers.return_value = [
            PoolTransfer(
                name=PoolName.KV,
                host_indices=node.component_data[ComponentType.FULL].host_value,
                nodes_to_load=[node],
            )
        ]
        cache.load_back_threshold = 1
        cache.evict = mock.Mock()
        cache.inc_host_lock_ref = mock.Mock(return_value=IncLockRefResult(delta=0))
        cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(delta=0))
        cache._record_store_event = mock.Mock()
        cache._update_evictable_leaf_sets = mock.Mock()
        cache.ongoing_load_back = {}

        def before_reduce():
            self.assertEqual(controller.load_queue, [])
            cache.evict.assert_not_called()
            mamba.cache.evict.assert_not_called()
            cache.inc_host_lock_ref.assert_not_called()
            cache.inc_lock_ref.assert_not_called()
            cache._record_store_event.assert_not_called()
            cache._update_evictable_leaf_sets.assert_not_called()
            base.commit_hicache_transfer.assert_not_called()
            swa.commit_hicache_transfer.assert_not_called()
            mamba.commit_hicache_transfer.assert_not_called()
            self.assertEqual(cache.ongoing_load_back, {})
            if not fail_request_mamba:
                self.assertEqual(pools.primary_device.alloc_calls, [4])
                self.assertEqual(pools.swa_device.alloc_calls, [4])
                self.assertEqual(pools.mamba_device.alloc_calls, [1])

        _install_consensus(
            cache,
            peer_succeeded=peer_succeeded,
            before_reduce=before_reduce,
        )
        return (
            cache,
            node,
            req,
            controller,
            pools,
            entries,
            request_mamba,
            base,
            swa,
            mamba,
            swa_transfer,
            mamba_transfers,
        )

    def test_peer_failure_aborts_primary_swa_tree_and_request_mamba(self):
        (
            cache,
            node,
            req,
            controller,
            pools,
            entries,
            request_mamba,
            base,
            swa,
            mamba,
            swa_transfer,
            mamba_transfers,
        ) = self._setup(peer_succeeded=False)

        loaded = cache.load_back(node, req=req)

        self.assertFalse(loaded)
        _assert_min_reduce(self, cache)
        self.assertEqual(controller.load_queue, [])
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        _assert_one_free(self, pools.mamba_device, _indices(600, 1))
        _assert_one_free(self, request_mamba, _indices(700, 1))
        self.assertIsNone(swa_transfer.device_indices)
        self.assertIsNone(mamba_transfers[0].device_indices)
        torch.testing.assert_close(mamba_transfers[1].device_indices, _indices(700, 1))
        self.assertIsNone(req.mamba_pool_idx)
        for entry in entries.values():
            entry.device_evict_fn.assert_not_called()
        cache.evict.assert_not_called()
        mamba.cache.evict.assert_not_called()
        cache.inc_host_lock_ref.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        cache._record_store_event.assert_not_called()
        cache._update_evictable_leaf_sets.assert_not_called()
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()
        self.assertEqual(cache.ongoing_load_back, {})

    def test_local_mamba_failure_still_participates_without_evicting(self):
        (
            cache,
            node,
            req,
            controller,
            pools,
            entries,
            request_mamba,
            base,
            swa,
            mamba,
            _,
            _,
        ) = self._setup(peer_succeeded=True, fail_request_mamba=True)

        loaded = cache.load_back(node, req=req)

        self.assertFalse(loaded)
        _assert_min_reduce(self, cache)
        self.assertEqual(request_mamba.alloc_calls, [1])
        self.assertEqual(request_mamba.freed, [])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(pools.primary_device.alloc_calls, [])
        self.assertEqual(pools.swa_device.alloc_calls, [])
        self.assertEqual(pools.mamba_device.alloc_calls, [])
        self.assertEqual(controller.load_queue, [])
        for entry in entries.values():
            entry.device_evict_fn.assert_not_called()
        cache.evict.assert_not_called()
        mamba.cache.evict.assert_not_called()
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()

    def test_local_controller_exception_cleans_all_then_reduces_as_failure(self):
        (
            cache,
            node,
            req,
            controller,
            pools,
            _,
            request_mamba,
            base,
            swa,
            mamba,
            _,
            _,
        ) = self._setup(
            peer_succeeded=True,
            raise_device_pool=PoolName.MAMBA,
        )

        loaded = cache.load_back(node, req=req)

        self.assertFalse(loaded)
        _assert_min_reduce(self, cache)
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        _assert_one_free(self, request_mamba, _indices(700, 1))
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(controller.load_queue, [])
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()

    def test_consensus_exception_aborts_controller_and_request_mamba(self):
        (
            cache,
            node,
            req,
            controller,
            pools,
            _,
            request_mamba,
            base,
            swa,
            mamba,
            _,
            _,
        ) = self._setup(peer_succeeded=True)
        cache._all_reduce_attn_groups.side_effect = RuntimeError(
            "injected reduce failure"
        )

        with self.assertRaisesRegex(RuntimeError, "injected reduce failure"):
            cache.load_back(node, req=req)

        _assert_min_reduce(self, cache)
        _assert_one_free(self, pools.primary_device, _indices(400, 4))
        _assert_one_free(self, pools.swa_device, _indices(500, 4))
        _assert_one_free(self, pools.mamba_device, _indices(600, 1))
        _assert_one_free(self, request_mamba, _indices(700, 1))
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(controller.load_queue, [])
        base.commit_hicache_transfer.assert_not_called()
        swa.commit_hicache_transfer.assert_not_called()
        mamba.commit_hicache_transfer.assert_not_called()

    def test_success_queues_then_commits_tree_and_request_state(self):
        (
            cache,
            node,
            req,
            controller,
            pools,
            _,
            request_mamba,
            base,
            swa,
            mamba,
            _,
            mamba_transfers,
        ) = self._setup(peer_succeeded=True)

        loaded = cache.load_back(node, req=req)

        self.assertTrue(loaded)
        _assert_min_reduce(self, cache)
        self.assertEqual(len(controller.load_queue), 1)
        operation = controller.load_queue[0]
        self.assertIsInstance(operation, CacheOperation)
        self.assertEqual(operation.node_ids, [node.id])
        self.assertEqual(pools.primary_device.freed, [])
        self.assertEqual(pools.swa_device.freed, [])
        self.assertEqual(pools.mamba_device.freed, [])
        self.assertEqual(request_mamba.freed, [])
        torch.testing.assert_close(mamba_transfers[0].device_indices, _indices(600, 1))
        torch.testing.assert_close(mamba_transfers[1].device_indices, _indices(700, 1))
        torch.testing.assert_close(req.mamba_pool_idx, torch.tensor(700))
        base.commit_hicache_transfer.assert_called_once()
        swa.commit_hicache_transfer.assert_called_once()
        mamba.commit_hicache_transfer.assert_called_once()
        cache.inc_host_lock_ref.assert_called_once_with(node)
        cache.inc_lock_ref.assert_called_once_with(node)
        cache._record_store_event.assert_called_once()
        cache._update_evictable_leaf_sets.assert_called_once_with(node)
        self.assertIn(node.id, cache.ongoing_load_back)

    def test_single_rank_keeps_path_locked_while_local_eviction_makes_room(self):
        (
            cache,
            node,
            req,
            controller,
            _,
            _,
            _,
            base,
            swa,
            mamba,
            _,
            _,
        ) = self._setup(peer_succeeded=True)
        cache.tp_world_size = 1
        cache.supports_swa = mock.Mock(return_value=False)
        cache.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=mock.Mock(return_value=0)
        )
        cache.dec_lock_ref = mock.Mock()
        cache.dec_host_lock_ref = mock.Mock()

        def evict(_params):
            cache.inc_host_lock_ref.assert_called_once_with(node)
            cache.inc_lock_ref.assert_called_once_with(node)
            self.assertEqual(controller.load_queue, [])
            base.commit_hicache_transfer.assert_not_called()
            swa.commit_hicache_transfer.assert_not_called()
            mamba.commit_hicache_transfer.assert_not_called()
            return EvictResult(num_tokens_evicted=4)

        cache.evict = mock.Mock(side_effect=evict)

        loaded = cache.load_back(node, req=req)

        self.assertTrue(loaded)
        cache._all_reduce_attn_groups.assert_not_called()
        cache.evict.assert_called_once()
        self.assertEqual(cache.inc_lock_ref.call_count, 2)
        cache.dec_lock_ref.assert_called_once()
        cache.dec_host_lock_ref.assert_not_called()
        self.assertIn(node.id, cache.ongoing_load_back)


class TestMambaLoadCommit(unittest.TestCase):
    def test_resets_replayssm_cursors_for_tree_and_request_slots(self):
        mamba_pool = SimpleNamespace(
            replayssm_write_pos=torch.full((10,), 11, dtype=torch.int32),
            replayssm_cache_base=torch.full((10,), 12, dtype=torch.int32),
            replayssm_is_flush=torch.ones((10,), dtype=torch.int8),
        )
        mamba_pool.reset_replayssm_cursors = MethodType(
            MambaPool.reset_replayssm_cursors, mamba_pool
        )
        host_lru = SimpleNamespace(
            in_list=mock.Mock(return_value=True),
            remove_node=mock.Mock(),
        )
        device_lru = SimpleNamespace(insert_mru=mock.Mock())
        cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_pool=mamba_pool),
            host_lru_lists={ComponentType.MAMBA: host_lru},
            lru_lists={ComponentType.MAMBA: device_lru},
            component_evictable_size_={ComponentType.MAMBA: 0},
        )
        component = SimpleNamespace(
            component_type=ComponentType.MAMBA,
            cache=cache,
        )
        component_data = SimpleNamespace(value=None)
        node = SimpleNamespace(
            component_data={ComponentType.MAMBA: component_data},
        )
        tree_indices = _indices(3, 1)
        request_indices = _indices(7, 1)

        MambaComponent.commit_hicache_transfer(
            component,
            node,
            CacheTransferPhase.LOAD_BACK,
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=tree_indices,
                ),
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=request_indices,
                ),
            ],
        )

        for cursor in (
            mamba_pool.replayssm_write_pos,
            mamba_pool.replayssm_cache_base,
            mamba_pool.replayssm_is_flush,
        ):
            torch.testing.assert_close(
                cursor[tree_indices], torch.zeros_like(cursor[tree_indices])
            )
            torch.testing.assert_close(
                cursor[request_indices], torch.zeros_like(cursor[request_indices])
            )
            self.assertNotEqual(cursor[0].item(), 0)
        torch.testing.assert_close(component_data.value, tree_indices)
        host_lru.remove_node.assert_called_once_with(node)
        device_lru.insert_mru.assert_called_once_with(node)
        self.assertEqual(cache.component_evictable_size_[ComponentType.MAMBA], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
