"""Focused CPU tests for HiMamba HiCache TP transactions."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import cache_controller as manager_cache_controller
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation as HybridCacheOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _indices(n: int, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.int64)


def _write_node(fake) -> SimpleNamespace:
    return SimpleNamespace(
        parent=fake.root_node,
        backuped=True,
        id=17,
        value=_indices(4),
        host_value=None,
        mamba_value=_indices(1, 20),
        mamba_host_value=None,
    )


class TestHiMambaTPTransactions(unittest.TestCase):
    def test_fingerprint_rejects_opcode_mismatch_and_reports_peer_error(self):
        fake = SimpleNamespace(tp_world_size=2, tp_group=object())

        def gather_opcode_mismatch(outputs, local, group):
            outputs[0].copy_(local)
            peer = local.clone()
            peer[2] = 2
            outputs[1].copy_(peer)

        with mock.patch.object(
            torch.distributed, "all_gather", side_effect=gather_opcode_mismatch
        ):
            group_ok, peer_error = HiMambaRadixCache._tp_transaction_consensus(
                fake,
                local_ok=True,
                local_error=False,
                opcode=1,
                node_id=17,
                target_tokens=4,
                mamba_tree_rows=1,
                request_mamba_rows=0,
            )
        self.assertFalse(group_ok)
        self.assertFalse(peer_error)

        def gather_peer_error(outputs, local, group):
            outputs[0].copy_(local)
            peer = local.clone()
            peer[0] = 0
            peer[1] = 1
            outputs[1].copy_(peer)

        with mock.patch.object(
            torch.distributed, "all_gather", side_effect=gather_peer_error
        ):
            group_ok, peer_error = HiMambaRadixCache._tp_transaction_consensus(
                fake,
                local_ok=True,
                local_error=False,
                opcode=1,
                node_id=17,
                target_tokens=4,
                mamba_tree_rows=1,
                request_mamba_rows=0,
            )
        self.assertFalse(group_ok)
        self.assertTrue(peer_error)

    def test_peer_write_rejection_aborts_without_tree_or_queue_mutation(self):
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=_indices(1),
            host_indices=_indices(1, 10),
        )
        reservation = SimpleNamespace(
            host_indices=_indices(4, 30),
            pool_reservation=SimpleNamespace(transfers=[transfer]),
        )
        controller = SimpleNamespace(
            reserve_write=mock.Mock(return_value=reservation),
            abort_write=mock.Mock(),
            commit_write=mock.Mock(),
        )
        fake = SimpleNamespace(
            root_node=object(),
            tp_world_size=2,
            cache_controller=controller,
            mamba_backup_transfers=mock.Mock(return_value=[transfer]),
            _tp_transaction_consensus=mock.Mock(return_value=(False, False)),
            mamba_backup_commit=mock.Mock(),
            mamba_host_lru_list=SimpleNamespace(
                in_list=mock.Mock(return_value=False),
                reset_node_mru=mock.Mock(),
            ),
            ongoing_write_through={},
            inc_lock_ref=mock.Mock(),
            evict_host=mock.Mock(),
            _watermark_evict_host_pools=mock.Mock(),
        )
        node = _write_node(fake)

        result = HiMambaRadixCache.write_backup(fake, node)

        self.assertEqual(result, 0)
        controller.abort_write.assert_called_once_with(reservation)
        controller.commit_write.assert_not_called()
        self.assertIsNone(node.host_value)
        self.assertEqual(fake.ongoing_write_through, {})
        fake.inc_lock_ref.assert_not_called()
        fake.mamba_backup_commit.assert_not_called()

    def _load_fixture(self, *, consensus=(False, False)):
        ancestor = SimpleNamespace(evicted=False)
        node = SimpleNamespace(
            evicted=True,
            backuped=True,
            parent=ancestor,
            id=23,
            host_value=_indices(4, 40),
            value=None,
            mamba_backuped=True,
            mamba_evicted=True,
            mamba_host_value=_indices(1, 50),
            mamba_value=None,
        )
        req = SimpleNamespace(mamba_pool_idx=None)
        pending = _indices(1, 60)
        mamba_allocator = SimpleNamespace(
            alloc=mock.Mock(return_value=pending),
            free=mock.Mock(),
        )
        reservation = SimpleNamespace(
            host_indices=node.host_value,
            device_indices=_indices(4, 70),
            pool_reservation=SimpleNamespace(transfers=None),
        )

        def reserve_load(*, host_indices, node_id, extra_pools, allow_evict):
            extra_pools[0].device_indices = _indices(1, 80)
            reservation.pool_reservation.transfers = extra_pools
            return reservation

        controller = SimpleNamespace(
            reserve_load=mock.Mock(side_effect=reserve_load),
            abort_load=mock.Mock(),
            commit_load=mock.Mock(return_value=reservation.device_indices),
        )
        fake = SimpleNamespace(
            tp_world_size=2,
            load_back_threshold=1,
            cache_controller=controller,
            req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_allocator),
            inc_lock_ref=mock.Mock(return_value=SimpleNamespace(delta=0)),
            dec_lock_ref=mock.Mock(),
            _tp_transaction_consensus=mock.Mock(return_value=consensus),
            mamba_restore_transfers=lambda last, nodes, request, pending_request_indices=None: (
                [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=last.mamba_host_value,
                    ),
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=last.mamba_host_value,
                        device_indices=pending_request_indices,
                    ),
                ]
            ),
            evict=mock.Mock(),
            mamba_restore_commit=lambda restored, transfers, req=None: (
                HiMambaRadixCache.mamba_restore_commit(
                    fake, restored, transfers, req=req
                )
            ),
            _record_store_event=mock.Mock(),
            full_lru_list=SimpleNamespace(insert_mru=mock.Mock()),
            full_evictable_size_=0,
            _update_leaf_status=mock.Mock(),
            mamba_lru_list=SimpleNamespace(
                in_list=mock.Mock(return_value=False),
                reset_node_mru=mock.Mock(),
                insert_mru=mock.Mock(),
            ),
            mamba_evictable_size_=0,
            ongoing_load_back={},
        )
        return fake, node, req, pending, reservation

    def test_peer_load_rejection_rolls_back_request_and_controller_slots(self):
        fake, node, req, pending, reservation = self._load_fixture()

        result = HiMambaRadixCache.load_back(fake, node, req=req)

        self.assertIsNone(result)
        fake.cache_controller.abort_load.assert_called_once_with(reservation)
        fake.req_to_token_pool.mamba_allocator.free.assert_called_once_with(pending)
        fake.cache_controller.commit_load.assert_not_called()
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(node.value)
        self.assertIsNone(node.mamba_value)
        fake.dec_lock_ref.assert_called_once()

    def test_success_assigns_request_slot_only_after_commit(self):
        fake, node, req, pending, _ = self._load_fixture(consensus=(True, False))

        def commit_load(reservation):
            self.assertIsNone(req.mamba_pool_idx)
            return reservation.device_indices

        fake.cache_controller.commit_load.side_effect = commit_load

        result = HiMambaRadixCache.load_back(fake, node, req=req)

        self.assertEqual(result.tolist(), [70, 71, 72, 73])
        self.assertEqual(int(req.mamba_pool_idx), 60)
        self.assertEqual(node.mamba_value.tolist(), [80])
        fake.req_to_token_pool.mamba_allocator.free.assert_not_called()
        fake.cache_controller.abort_load.assert_not_called()
        fake.cache_controller.commit_load.assert_called_once()


# NOTE: no **kwargs on purpose — the cached _timing_events_supported() probe
# must fail on Event(enable_timing=True), matching every other fake-event
# harness in this suite (a kwargs-tolerant fake would cache timing support as
# True and break those harnesses later in the same process).
class _FakeEvent:
    def record(self):
        pass

    def wait(self, stream):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class TestMambaOnlyLoadOp(unittest.TestCase):
    def test_empty_kv_load_with_mamba_transfer_completes_and_acks(self):
        """A mamba-only load op (zero KV pages) must run to completion.

        The dedup source rank must still execute the mamba pool transfer on
        every layer, complete every layer event, skip the draft pool, and
        append a well-formed ack.
        """
        operations = []
        broadcasts = []
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(1, 50),
            device_indices=_indices(1, 60),
        )
        op = HybridCacheOperation(
            host_indices=torch.empty((0,), dtype=torch.int64),
            device_indices=torch.empty((0,), dtype=torch.int64),
            node_id=23,
            pool_transfers=[mamba_transfer],
        )

        class FakeHostGroup:
            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                pool_transfers=None,
            ):
                operations.append((layer_id, host_indices.numel(), pool_transfers))

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def __init__(self):
                self.completed = []

            def complete(self, layer_index):
                self.completed.append(layer_index)

        producer_event = FakeProducerEvent()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.device = torch.device("cpu")
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = object()
        controller.has_draft = True
        controller.mem_pool_host_draft = SimpleNamespace(
            layer_num=2,
            load_to_device_per_layer=mock.Mock(
                side_effect=AssertionError("draft must not load on a mamba-only op")
            ),
        )
        controller.mem_pool_device_draft = object()
        controller.layer_num = 2
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[producer_event]
        )
        controller.mla_broadcaster = SimpleNamespace(
            is_src=True,
            prepare_broadcast=lambda device_indices, stream: (device_indices, None),
            broadcast_loaded_layer=lambda layer_id, prepared: broadcasts.append(
                layer_id
            ),
        )
        controller.load_stream = object()
        controller.ack_load_queue = []

        with (
            mock.patch.object(
                hybrid_cache_controller, "device_module", _FakeDeviceModule
            ),
            mock.patch.object(
                manager_cache_controller, "device_module", _FakeDeviceModule
            ),
        ):
            producer_id = controller.start_loading()

        self.assertEqual(producer_id, 0)
        self.assertEqual(producer_event.completed, [0, 1])
        self.assertEqual(broadcasts, [0, 1])
        self.assertEqual([layer_id for layer_id, _, _ in operations], [0, 1])
        for _, kv_numel, transfers in operations:
            self.assertEqual(kv_numel, 0)
            self.assertEqual(len(transfers), 1)
            self.assertIs(transfers[0].name, PoolName.MAMBA)
            self.assertEqual(
                transfers[0].host_indices.tolist(),
                mamba_transfer.host_indices.tolist(),
            )
            self.assertEqual(
                transfers[0].device_indices.tolist(),
                mamba_transfer.device_indices.tolist(),
            )
        controller.mem_pool_host_draft.load_to_device_per_layer.assert_not_called()
        self.assertEqual(len(controller.ack_load_queue), 1)
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.node_ids, [23])
        self.assertEqual(ack.num_tokens, 0)


if __name__ == "__main__":
    unittest.main()
