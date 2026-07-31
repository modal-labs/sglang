"""CPU-only unit tests for mamba_max_states_per_path in HiMambaRadixCache."""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache, HostLRUList
from sglang.srt.mem_cache.mamba_radix_cache import LRUList, MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.extend(torch.as_tensor(value).tolist())


def _build_cache(cap: int) -> HiMambaRadixCache:
    """Minimal HiMambaRadixCache wired for the CPU insert/match/prune paths."""
    cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
    cache.disable = False
    cache.page_size = 1
    cache.device = torch.device("cpu")
    cache.enable_storage = False
    cache.enable_kv_cache_events = False
    cache.mamba_max_states_per_path = cap
    cache.token_to_kv_pool_allocator = _RecordingAllocator()
    cache.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingAllocator())
    # write_back short-circuits _inc_hit_count, keeping inserts controller-free
    cache.cache_controller = SimpleNamespace(write_policy="write_back")
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
    """Insert a root-anchored key; each call adds one deeper mamba state."""
    return cache.insert(
        InsertParams(
            key=RadixKey(list(tokens), None),
            value=torch.arange(1000, 1000 + len(tokens), dtype=torch.int64),
            mamba_value=torch.tensor([mamba_slot], dtype=torch.int64),
        )
    )


def _build_chain(cache: HiMambaRadixCache, length: int) -> list:
    """Insert [1], [1,2], ... producing one mamba-holding node per level."""
    nodes = []
    parent = cache.root_node
    for depth in range(1, length + 1):
        _insert(cache, list(range(1, depth + 1)), 10 + depth - 1)
        parent = parent.children[depth]
        nodes.append(parent)
    return nodes


def _match(cache: HiMambaRadixCache, tokens: list):
    with mock.patch(
        "sglang.srt.runtime_context.get_server_args",
        return_value=SimpleNamespace(mamba_cache_chunk_size=64),
    ):
        return cache.match_prefix(MatchPrefixParams(key=RadixKey(list(tokens), None)))


class TestHiMambaPathStateCap(unittest.TestCase):
    def test_cap_disabled_keeps_all_states(self):
        cache = _build_cache(cap=-1)
        nodes = _build_chain(cache, 3)

        self.assertTrue(all(node.mamba_value is not None for node in nodes))
        self.assertEqual(cache.mamba_evictable_size_, 3)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [])
        self.assertTrue(all(cache.mamba_lru_list.in_list(node) for node in nodes))

    def test_insert_prunes_shallowest_interior_state(self):
        cache = _build_cache(cap=2)
        nodes = _build_chain(cache, 3)

        self.assertIsNone(nodes[0].mamba_value)
        self.assertIsNotNone(nodes[1].mamba_value)
        self.assertIsNotNone(nodes[2].mamba_value)
        # only the shallowest device slot was returned to the allocator
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [10])
        self.assertFalse(cache.mamba_lru_list.in_list(nodes[0]))
        self.assertEqual(cache.mamba_evictable_size_, 2)
        # full KV is never touched by the cap
        self.assertTrue(all(node.value is not None for node in nodes))
        self.assertEqual(cache.full_evictable_size_, 3)

    def test_match_prefix_clamps_after_prune(self):
        cache = _build_cache(cap=2)
        _build_chain(cache, 3)

        # deepest state survives: the full path is still matchable
        full = _match(cache, [1, 2, 3])
        self.assertEqual(len(full.device_indices), 3)
        # the pruned (non-backed-up) shallow node is no longer a mamba
        # boundary, so a 1-token match clamps to zero
        clamped = _match(cache, [1])
        self.assertEqual(len(clamped.device_indices), 0)

    def test_pruned_node_keeps_host_backup_and_match_boundary(self):
        cache = _build_cache(cap=2)
        nodes = _build_chain(cache, 2)
        nodes[0].mamba_host_value = torch.tensor([99], dtype=torch.int64)

        _insert(cache, [1, 2, 3], 12)

        # device state pruned, host backup untouched -> restorable via the
        # mamba-only load path
        self.assertIsNone(nodes[0].mamba_value)
        self.assertIsNotNone(nodes[0].mamba_host_value)
        self.assertEqual(nodes[0].mamba_host_value.tolist(), [99])
        self.assertTrue(nodes[0].mamba_backuped)
        # a host-backed node remains a valid mamba match boundary
        result = _match(cache, [1])
        self.assertEqual(len(result.device_indices), 1)

    def test_tail_fork_and_locked_nodes_preserved(self):
        cache = _build_cache(cap=-1)
        nodes = _build_chain(cache, 3)
        fork_child = TreeNode()
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child
        nodes[1].mamba_lock_ref = 1

        cache.mamba_max_states_per_path = 1
        _insert(cache, [1, 2, 3, 4], 13)
        tail = nodes[2].children[4]

        self.assertIsNotNone(nodes[0].mamba_value)  # fork
        self.assertIsNotNone(nodes[1].mamba_value)  # locked
        self.assertIsNone(nodes[2].mamba_value)  # only eligible interior
        self.assertIsNotNone(tail.mamba_value)  # tail
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [12])

    def test_full_locked_node_preserved(self):
        cache = _build_cache(cap=-1)
        nodes = _build_chain(cache, 2)
        nodes[0].full_lock_ref = 1

        cache.mamba_max_states_per_path = 2
        _insert(cache, [1, 2, 3], 12)

        self.assertIsNotNone(nodes[0].mamba_value)
        self.assertIsNone(nodes[1].mamba_value)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [11])

    def test_inflight_backup_node_preserved(self):
        cache = _build_cache(cap=-1)
        nodes = _build_chain(cache, 2)
        cache.ongoing_write_through[nodes[0].id] = nodes[0]

        cache.mamba_max_states_per_path = 2
        _insert(cache, [1, 2, 3], 12)

        self.assertIsNotNone(nodes[0].mamba_value)  # mid-backup DMA source
        self.assertIsNone(nodes[1].mamba_value)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [11])

    def test_inflight_load_back_node_preserved(self):
        cache = _build_cache(cap=-1)
        nodes = _build_chain(cache, 2)
        cache.ongoing_load_back[nodes[0].id] = nodes[0]

        cache.mamba_max_states_per_path = 2
        _insert(cache, [1, 2, 3], 12)

        self.assertIsNotNone(nodes[0].mamba_value)  # mid-load-back DMA target
        self.assertIsNone(nodes[1].mamba_value)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [11])

    def test_enforcement_is_shared_with_base_class(self):
        # The pruning mechanism lives in MambaRadixCache; HiMamba only
        # customizes the in-flight-transfer skip rules via the hook.
        self.assertIs(
            HiMambaRadixCache._enforce_mamba_path_state_cap,
            MambaRadixCache._enforce_mamba_path_state_cap,
        )
        self.assertIsNot(
            HiMambaRadixCache._mamba_cap_extra_skip,
            MambaRadixCache._mamba_cap_extra_skip,
        )


if __name__ == "__main__":
    unittest.main()
