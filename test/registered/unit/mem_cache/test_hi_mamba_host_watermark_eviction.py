"""CPU-only tests for symmetric host-pool watermark eviction (TP>1, L2-only).

At TP>1 every HiMamba HiCache host reservation runs with allow_evict=False,
so the lockstep watermark preflight in write_backup is the only code path
that ever reclaims host KV / host Mamba capacity. These tests drive the real
write_backup / evict_host / evict_mamba_host code over fake host pools and a
fake hybrid controller that mirrors reserve/commit/abort semantics, with the
TP consensus replaced by a local-state evaluation (rank-symmetric by
construction, matching the real fingerprint gather).
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache, HostLRUList
from sglang.srt.mem_cache.mamba_radix_cache import LRUList, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.extend(torch.as_tensor(value).tolist())


class _FakeHostPool:
    """Slot pool with the HostKVCache alloc/free/available_size surface."""

    def __init__(self, size: int):
        self.size = size
        self.page_size = 1
        self.free_slots = list(range(size))

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None
        out = torch.tensor(self.free_slots[:need_size], dtype=torch.int64)
        del self.free_slots[:need_size]
        return out

    def free(self, indices) -> int:
        values = torch.as_tensor(indices).tolist()
        self.free_slots.extend(int(v) for v in values)
        return len(values)


class _FakeHybridController:
    """Mirrors HybridCacheController reserve/commit/abort over fake pools.

    reserve_write allocates host KV for the device indices and one host
    mamba row per extra-pool transfer that arrives without host_indices,
    exactly like _reserve_pool_transfers with allow_evict=False.
    """

    write_policy = "write_back"  # keeps _inc_hit_count inert during inserts

    def __init__(self, kv_host_pool: _FakeHostPool, mamba_host_pool: _FakeHostPool):
        self.kv_host_pool = kv_host_pool
        self.mamba_host_pool = mamba_host_pool

    def reserve_write(
        self, device_indices, node_id=-1, extra_pools=None, *, allow_evict=False
    ):
        host_indices = self.kv_host_pool.alloc(len(device_indices))
        if host_indices is None:
            return None
        reserved = []
        for transfer in extra_pools or []:
            if transfer.host_indices is not None or transfer.device_indices is None:
                continue
            rows = self.mamba_host_pool.alloc(len(transfer.device_indices))
            if rows is None:
                for prev_transfer, previous in reserved:
                    self.mamba_host_pool.free(prev_transfer.host_indices)
                    prev_transfer.host_indices = previous
                self.kv_host_pool.free(host_indices)
                return None
            reserved.append((transfer, transfer.host_indices))
            transfer.host_indices = rows
        return SimpleNamespace(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
            pool_reservation=SimpleNamespace(transfers=extra_pools),
            _reserved=reserved,
        )

    def commit_write(self, reservation):
        return reservation.host_indices

    def abort_write(self, reservation):
        for transfer, previous in reservation._reserved:
            self.mamba_host_pool.free(transfer.host_indices)
            transfer.host_indices = previous
        self.kv_host_pool.free(reservation.host_indices)

    def evict_device(self, device_indices) -> int:
        return len(device_indices)

    def evict_host(self, host_indices, backup_only: bool = True) -> int:
        self.kv_host_pool.free(host_indices)
        return len(host_indices)


def _local_consensus(**kwargs):
    """Stand-in for the TP fingerprint gather: rank-symmetric local decision."""
    return kwargs["local_ok"] and not kwargs["local_error"], kwargs["local_error"]


def _build_cache(
    *,
    mamba_host_slots: int,
    kv_host_tokens: int,
    trigger_ratio: float = 0.0,
    batch_ratio: float = 0.0,
) -> HiMambaRadixCache:
    TreeNode.counter = 0
    cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
    cache.disable = False
    cache.page_size = 1
    cache.device = torch.device("cpu")
    cache.enable_storage = False
    cache.enable_kv_cache_events = False
    cache.mamba_max_states_per_path = -1
    cache.token_to_kv_pool_allocator = _RecordingAllocator()
    cache.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingAllocator())
    cache.full_kv_pool_host = _FakeHostPool(kv_host_tokens)
    cache.mamba_pool_host = _FakeHostPool(mamba_host_slots)
    cache.cache_controller = _FakeHybridController(
        cache.full_kv_pool_host, cache.mamba_pool_host
    )
    cache.tp_world_size = 2
    cache._tp_transaction_consensus = _local_consensus
    cache.host_evict_trigger_ratio = trigger_ratio
    cache.host_evict_batch_ratio = batch_ratio
    cache.ongoing_write_through = {}
    cache.ongoing_load_back = {}
    cache.evictable_full_device_leaves = set()
    cache.evictable_full_host_leaves = set()
    cache.full_lru_list = LRUList(mamba=False)
    cache.mamba_lru_list = LRUList(mamba=True)
    cache.mamba_host_lru_list = HostLRUList()
    cache.full_evictable_size_ = 0
    cache.mamba_evictable_size_ = 0
    cache.full_protected_size_ = 0
    cache.mamba_protected_size_ = 0

    root = TreeNode()
    root.key = RadixKey(array("q"), None)
    root.value = []
    root.hash_value = []
    root.full_lock_ref = 1
    root.mamba_lock_ref = 1
    cache.root_node = root
    return cache


def _insert(cache: HiMambaRadixCache, tokens: list, mamba_slot: int):
    return cache.insert(
        InsertParams(
            key=RadixKey(list(tokens), None),
            value=torch.arange(1000, 1000 + len(tokens), dtype=torch.int64),
            mamba_value=torch.tensor([mamba_slot], dtype=torch.int64),
        )
    )


def _drain_writes(cache: HiMambaRadixCache, only=None):
    """Complete pending write-through DMAs the way writing_check would."""
    for node_id, node in list(cache.ongoing_write_through.items()):
        if only is not None and node is not only:
            continue
        cache.dec_lock_ref(node)
        del cache.ongoing_write_through[node_id]


def _match(cache: HiMambaRadixCache, tokens: list):
    with mock.patch(
        "sglang.srt.runtime_context.get_server_args",
        return_value=SimpleNamespace(mamba_cache_chunk_size=64),
    ):
        return cache.match_prefix(MatchPrefixParams(key=RadixKey(list(tokens), None)))


class TestHostMambaWatermark(unittest.TestCase):
    def test_mamba_host_pool_capacity_recovers(self):
        """>pool-size distinct finished requests keep succeeding via LRU reclaim.

        Pre-fix this pins at 0 free slots with every write_backup returning 0
        (consensus reject on all ranks) from the fifth write onward.
        """
        cache = _build_cache(mamba_host_slots=4, kv_host_tokens=1024)
        leaves = []
        for i in range(10):
            token = i + 1
            _insert(cache, [token], 100 + i)
            node = cache.root_node.children[token]
            leaves.append(node)
            written = cache.write_backup(node)
            self.assertEqual(written, 1, f"write {i} rejected")
            self.assertIsNotNone(node.mamba_host_value)
            _drain_writes(cache)

        # Six oldest rows were reclaimed in LRU order; their host KV backup is
        # untouched (internal-branch semantics for device-resident nodes).
        for node in leaves[:6]:
            self.assertIsNone(node.mamba_host_value)
            self.assertFalse(node.mamba_backuped)
            self.assertFalse(cache.mamba_host_lru_list.in_list(node))
            self.assertIsNotNone(node.host_value)
        for node in leaves[6:]:
            self.assertIsNotNone(node.mamba_host_value)
            self.assertTrue(cache.mamba_host_lru_list.in_list(node))
        self.assertEqual(cache.mamba_pool_host.available_size(), 0)

    def test_watermark_trigger_and_batch_ratios(self):
        """Trigger fires at free < needed + 2% and evicts a 5% batch."""
        cache = _build_cache(
            mamba_host_slots=100,
            kv_host_tokens=4096,
            trigger_ratio=0.02,
            batch_ratio=0.05,
        )
        leaves = []
        for i in range(98):
            token = i + 1
            _insert(cache, [token], 200 + i)
            node = cache.root_node.children[token]
            leaves.append(node)
            self.assertEqual(cache.write_backup(node), 1)
            _drain_writes(cache)
        # trigger_free = 1 + int(100 * 0.02) = 3; free is still 2 >= ... no:
        # after 98 writes free == 2 < 3, so the 99th write must evict a batch
        # of max(3 - 2, int(100 * 0.05), 1) == 5 rows and log one INFO line.
        _insert(cache, [99], 298)
        node99 = cache.root_node.children[99]
        with self.assertLogs(
            "sglang.srt.mem_cache.hi_mamba_radix_cache", level="INFO"
        ) as logs:
            self.assertEqual(cache.write_backup(node99), 1)
        _drain_writes(cache)
        self.assertTrue(
            any("host watermark eviction" in line for line in logs.output)
        )
        self.assertEqual(cache.mamba_pool_host.available_size(), 6)
        for node in leaves[:5]:
            self.assertIsNone(node.mamba_host_value)
        self.assertIsNotNone(leaves[5].mamba_host_value)

        # 100th write sits above the watermark again: no eviction.
        _insert(cache, [100], 299)
        node100 = cache.root_node.children[100]
        self.assertEqual(cache.write_backup(node100), 1)
        _drain_writes(cache)
        self.assertEqual(cache.mamba_pool_host.available_size(), 5)
        self.assertIsNotNone(leaves[5].mamba_host_value)


class TestHostKVWatermark(unittest.TestCase):
    def test_kv_host_pool_watermark_recovers(self):
        cache = _build_cache(mamba_host_slots=64, kv_host_tokens=8)
        keys = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        written_nodes = []
        for i, key in enumerate(keys[:2]):
            _insert(cache, key, 100 + i)
            node = cache.root_node.children[key[0]]
            self.assertEqual(cache.write_backup(node), 4)
            _drain_writes(cache)
            written_nodes.append(node)
        self.assertEqual(cache.full_kv_pool_host.available_size(), 0)

        # Demote both to host-only so they become evictable host leaves.
        cache.evict(EvictParams(num_tokens=8))
        for node in written_nodes:
            self.assertTrue(node.evicted)
            self.assertIn(node, cache.evictable_full_host_leaves)

        # Third write reclaims host KV through the watermark preflight.
        _insert(cache, keys[2], 102)
        node3 = cache.root_node.children[keys[2][0]]
        self.assertEqual(cache.write_backup(node3), 4)
        _drain_writes(cache)

        victim, survivor = written_nodes
        self.assertIsNone(victim.host_value)
        self.assertIsNone(victim.mamba_host_value)
        self.assertFalse(victim.mamba_backuped)
        self.assertNotIn(keys[0][0], cache.root_node.children)
        self.assertIsNotNone(survivor.host_value)
        self.assertEqual(cache.full_kv_pool_host.available_size(), 0)

        # match_prefix clamps cleanly on the evicted key.
        result = _match(cache, keys[0])
        self.assertEqual(len(result.device_indices), 0)
        self.assertIs(result.last_host_node, cache.root_node)


class TestEvictionPreservation(unittest.TestCase):
    def test_inflight_and_locked_rows_never_evicted(self):
        cache = _build_cache(mamba_host_slots=2, kv_host_tokens=1024)

        # A: written, DMA still in flight (locked + in ongoing_write_through).
        _insert(cache, [1], 100)
        node_a = cache.root_node.children[1]
        self.assertEqual(cache.write_backup(node_a), 1)

        # B: written and completed.
        _insert(cache, [2], 101)
        node_b = cache.root_node.children[2]
        self.assertEqual(cache.write_backup(node_b), 1)
        _drain_writes(cache, only=node_b)

        # C: pool full; the preflight must skip locked in-flight A and evict B
        # even though A is the LRU entry.
        _insert(cache, [3], 102)
        node_c = cache.root_node.children[3]
        self.assertEqual(cache.write_backup(node_c), 1)
        _drain_writes(cache, only=node_c)
        self.assertIsNotNone(node_a.mamba_host_value)
        self.assertIsNone(node_b.mamba_host_value)

        # Release A's locks but keep it in ongoing_write_through: the map
        # guard alone must protect the in-flight DMA target row.
        cache.dec_lock_ref(node_a)
        _insert(cache, [4], 103)
        node_d = cache.root_node.children[4]
        self.assertEqual(cache.write_backup(node_d), 1)
        _drain_writes(cache, only=node_d)
        self.assertIsNotNone(node_a.mamba_host_value)
        self.assertIsNone(node_c.mamba_host_value)

        # Once A's write-through fully completes it becomes evictable again.
        del cache.ongoing_write_through[node_a.id]
        _insert(cache, [5], 104)
        node_e = cache.root_node.children[5]
        self.assertEqual(cache.write_backup(node_e), 1)
        _drain_writes(cache)
        self.assertIsNone(node_a.mamba_host_value)
        # Root is never touched.
        self.assertIsNone(cache.root_node.mamba_host_value)

    def test_tombstoned_with_host_copy_chain_fully_collapses(self):
        """Audit dead-end check: _delete_tombstone_leaf's assert and the
        cascade break must not strand tombstoned-with-host-copy nodes."""
        cache = _build_cache(mamba_host_slots=64, kv_host_tokens=1024)
        _insert(cache, [1], 100)
        parent = cache.root_node.children[1]
        self.assertEqual(cache.write_backup(parent), 1)
        _drain_writes(cache)
        _insert(cache, [1, 2], 101)
        child = parent.children[2]
        self.assertEqual(cache.write_backup(child), 1)
        _drain_writes(cache)

        # Demote both to host-only (parent becomes an interior host node with
        # a host mamba copy; its device mamba is freed => tombstone).
        cache.evict(EvictParams(num_tokens=2))
        self.assertTrue(parent.evicted and child.evicted)
        self.assertIsNone(parent.mamba_value)
        self.assertIsNotNone(parent.mamba_host_value)

        # Internal branch frees the parent's host row, leaf branch deletes the
        # child, and the cascade must then delete the parent tombstone too.
        evicted = cache.evict_mamba_host(2)
        self.assertEqual(evicted, 2)
        self.assertEqual(len(cache.root_node.children), 0)
        self.assertIsNone(parent.host_value)
        self.assertIsNone(parent.mamba_host_value)
        self.assertEqual(
            cache.full_kv_pool_host.available_size(), cache.full_kv_pool_host.size
        )
        self.assertEqual(
            cache.mamba_pool_host.available_size(), cache.mamba_pool_host.size
        )


class TestVictimDeterminism(unittest.TestCase):
    @staticmethod
    def _scripted_run():
        cache = _build_cache(mamba_host_slots=3, kv_host_tokens=64)
        for i in range(9):
            token = i + 1
            _insert(cache, [token], 100 + i)
            node = cache.root_node.children[token]
            cache.write_backup(node)
            _drain_writes(cache)
            if i in (4, 7):
                # Touch an older key: refreshes device LRUs and access times,
                # part of the shared rank-symmetric history.
                _match(cache, [i])

        signature = []
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            if node is not cache.root_node:
                signature.append(
                    (
                        tuple(node.key.token_ids),
                        node.mamba_host_value is not None,
                        node.host_value is not None,
                        node.evicted,
                    )
                )
            stack.extend(node.children.values())
        lru_order = [
            tuple(node.key.token_ids)
            for node in cache.mamba_host_lru_list.cache.values()
        ]
        return sorted(signature), lru_order

    def test_identical_history_yields_identical_victims(self):
        """Two independently-constructed caches with the same tree + LRU
        history must select identical victim sets (TP-symmetry proxy)."""
        first = self._scripted_run()
        second = self._scripted_run()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
