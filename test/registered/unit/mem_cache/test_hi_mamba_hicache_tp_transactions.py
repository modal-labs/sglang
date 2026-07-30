"""Focused CPU tests for HiMamba HiCache TP transactions."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
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
            group_ok, peer_error, fingerprint_match = (
                HiMambaRadixCache._tp_transaction_consensus(
                    fake,
                    local_ok=True,
                    local_error=False,
                    opcode=1,
                    node_id=17,
                    target_tokens=4,
                    mamba_tree_rows=1,
                    request_mamba_rows=0,
                )
            )
        self.assertFalse(group_ok)
        self.assertFalse(peer_error)
        self.assertFalse(fingerprint_match)

        def gather_peer_error(outputs, local, group):
            outputs[0].copy_(local)
            peer = local.clone()
            peer[0] = 0
            peer[1] = 1
            outputs[1].copy_(peer)

        with mock.patch.object(
            torch.distributed, "all_gather", side_effect=gather_peer_error
        ):
            group_ok, peer_error, fingerprint_match = (
                HiMambaRadixCache._tp_transaction_consensus(
                    fake,
                    local_ok=True,
                    local_error=False,
                    opcode=1,
                    node_id=17,
                    target_tokens=4,
                    mamba_tree_rows=1,
                    request_mamba_rows=0,
                )
            )
        self.assertFalse(group_ok)
        self.assertTrue(peer_error)
        self.assertTrue(fingerprint_match)

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
            _tp_transaction_consensus=mock.Mock(
                return_value=(False, False, False)
            ),
            mamba_backup_commit=mock.Mock(),
            mamba_host_lru_list=SimpleNamespace(
                in_list=mock.Mock(return_value=False),
                reset_node_mru=mock.Mock(),
            ),
            ongoing_write_through={},
            inc_lock_ref=mock.Mock(),
            evict_host=mock.Mock(),
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

    def test_peer_host_full_retries_after_synchronized_eviction(self):
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=_indices(1),
            host_indices=None,
        )
        first = SimpleNamespace(
            host_indices=_indices(4, 30),
            pool_reservation=SimpleNamespace(transfers=[transfer]),
        )
        second = SimpleNamespace(
            host_indices=_indices(4, 40),
            pool_reservation=SimpleNamespace(transfers=[transfer]),
        )
        controller = SimpleNamespace(
            reserve_write=mock.Mock(side_effect=[first, second]),
            abort_write=mock.Mock(),
            commit_write=mock.Mock(return_value=second.host_indices),
        )
        fake = SimpleNamespace(
            root_node=object(),
            tp_world_size=2,
            cache_controller=controller,
            mamba_backup_transfers=mock.Mock(return_value=[transfer]),
            _tp_transaction_consensus=mock.Mock(
                side_effect=[
                    (False, False, True),
                    (True, False, True),
                ]
            ),
            mamba_backup_commit=mock.Mock(),
            mamba_host_lru_list=SimpleNamespace(
                in_list=mock.Mock(return_value=False),
                reset_node_mru=mock.Mock(),
            ),
            ongoing_write_through={},
            inc_lock_ref=mock.Mock(),
            evict_host=mock.Mock(),
        )
        node = _write_node(fake)

        result = HiMambaRadixCache.write_backup(fake, node)

        self.assertEqual(result, 4)
        controller.abort_write.assert_called_once_with(first)
        fake.evict_host.assert_called_once_with(4)
        self.assertEqual(controller.reserve_write.call_count, 2)
        controller.commit_write.assert_called_once_with(second)
        self.assertEqual(node.host_value.tolist(), [40, 41, 42, 43])
        self.assertIs(fake.ongoing_write_through[node.id], node)

    def _load_fixture(self, *, consensus=(False, False, False)):
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
        fake, node, req, pending, _ = self._load_fixture(
            consensus=(True, False, True)
        )

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


if __name__ == "__main__":
    unittest.main()
