"""CPU-only unit tests for the per-path Mamba checkpoint cap."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import argparse
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.mamba_radix_cache import LRUList, MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedLRUList,
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.server_args import ServerArgs


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.extend(value.tolist())


class _FakeUnifiedCache:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingAllocator())
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }
        self.host_lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }
        self.evicted = []
        self.cascaded = []

    def _evict_component_and_detach_lru(self, node, component, **kwargs):
        self.evicted.append(node)
        return UnifiedRadixCache._evict_component_and_detach_lru(
            self, node, component, **kwargs
        )

    def _cascade_evict(self, node, component, tracker):
        self.cascaded.append(node)


def _build_unified_chain(cap, length=3):
    cache = _FakeUnifiedCache()
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.mamba_max_states_per_path = cap

    nodes = []
    parent = cache.root_node
    for index in range(length):
        node = UnifiedTreeNode(cache.tree_components)
        node.parent = parent
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.MAMBA].value = torch.tensor([index])
        parent.children[index] = node
        cache.component_evictable_size_[ComponentType.MAMBA] += 1
        cache.lru_lists[ComponentType.MAMBA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, cache


class TestMambaPathStateCap(unittest.TestCase):
    def test_server_arg_defaults_to_unlimited(self):
        self.assertEqual(
            ServerArgs(model_path="dummy").mamba_max_states_per_path,
            -1,
        )

    def test_server_arg_cli(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            ["--model-path", "dummy", "--mamba-max-states-per-path", "3"]
        )

        self.assertEqual(args.mamba_max_states_per_path, 3)

    def test_server_arg_rejects_zero_and_values_below_negative_one(self):
        for value in (0, -2):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "must be -1 \\(unlimited\\) or a positive integer",
            ):
                ServerArgs(
                    model_path="dummy",
                    mamba_max_states_per_path=value,
                )

    def test_unified_cache_removes_only_shallow_mamba_state(self):
        component, nodes, cache = _build_unified_chain(cap=2)

        component._evict_excess_path_states(nodes[-1])

        self.assertEqual(cache.evicted, [nodes[0]])
        self.assertEqual(cache.cascaded, [nodes[0]])
        self.assertIsNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[-1].component_data[ComponentType.MAMBA].value)
        self.assertEqual(
            cache.req_to_token_pool.mamba_allocator.freed,
            [0],
        )
        self.assertEqual(cache.component_evictable_size_[ComponentType.MAMBA], 2)
        self.assertFalse(cache.lru_lists[ComponentType.MAMBA].in_list(nodes[0]))
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )

    def test_unified_cache_cap_is_soft_for_fork_and_locked_nodes(self):
        component, nodes, cache = _build_unified_chain(cap=1, length=4)
        fork_child = UnifiedTreeNode(cache.tree_components)
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child
        nodes[1].component_data[ComponentType.MAMBA].lock_ref = 1

        component._evict_excess_path_states(nodes[-1])

        self.assertEqual(cache.evicted, [nodes[2]])
        self.assertIsNotNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[1].component_data[ComponentType.MAMBA].value)
        self.assertIsNone(nodes[2].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[3].component_data[ComponentType.MAMBA].value)

    def test_unified_cache_preserves_existing_host_backup(self):
        component, nodes, cache = _build_unified_chain(cap=2)
        mamba_data = nodes[0].component_data[ComponentType.MAMBA]
        mamba_data.host_value = torch.tensor([10])

        component._evict_excess_path_states(nodes[-1])

        self.assertIsNone(mamba_data.value)
        self.assertIsNotNone(mamba_data.host_value)
        self.assertTrue(cache.host_lru_lists[ComponentType.MAMBA].in_list(nodes[0]))

    def test_unified_cache_negative_one_disables_cap(self):
        component, nodes, cache = _build_unified_chain(cap=-1)

        component._evict_excess_path_states(nodes[-1])

        self.assertEqual(cache.evicted, [])
        self.assertTrue(
            all(
                node.component_data[ComponentType.MAMBA].value is not None
                for node in nodes
            )
        )


def _build_base_cache(cap: int) -> MambaRadixCache:
    """Minimal base MambaRadixCache (no hierarchical cache controller)."""
    cache = MambaRadixCache.__new__(MambaRadixCache)
    cache.disable = False
    cache.page_size = 1
    cache.device = torch.device("cpu")
    cache.enable_kv_cache_events = False
    cache.kv_event_queue = []
    cache.mamba_cache_chunk_size = 64
    cache.mamba_max_states_per_path = cap
    cache.token_to_kv_pool_allocator = _RecordingAllocator()
    cache.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingAllocator())
    cache.full_lru_list = LRUList(mamba=False)
    cache.mamba_lru_list = LRUList(mamba=True)
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


def _base_insert(cache: MambaRadixCache, tokens: list, mamba_slot: int):
    """Insert a root-anchored key; each call adds one deeper mamba state."""
    return cache.insert(
        InsertParams(
            key=RadixKey(list(tokens), None),
            value=torch.arange(1000, 1000 + len(tokens), dtype=torch.int64),
            mamba_value=torch.tensor([mamba_slot], dtype=torch.int64),
        )
    )


def _build_base_chain(cache: MambaRadixCache, length: int) -> list:
    """Insert [1], [1,2], ... producing one mamba-holding node per level."""
    nodes = []
    parent = cache.root_node
    for depth in range(1, length + 1):
        _base_insert(cache, list(range(1, depth + 1)), 10 + depth - 1)
        parent = parent.children[depth]
        nodes.append(parent)
    return nodes


class TestBaseMambaRadixCachePathStateCap(unittest.TestCase):
    """The cap must hold without --enable-hierarchical-cache (base class)."""

    def test_cap_disabled_keeps_all_states(self):
        cache = _build_base_cache(cap=-1)
        nodes = _build_base_chain(cache, 3)

        self.assertTrue(all(node.mamba_value is not None for node in nodes))
        self.assertEqual(cache.mamba_evictable_size_, 3)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [])
        self.assertTrue(all(cache.mamba_lru_list.in_list(node) for node in nodes))

    def test_insert_prunes_shallowest_interior_state(self):
        cache = _build_base_cache(cap=2)
        nodes = _build_base_chain(cache, 3)

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

    def test_tombstone_revive_branch_enforces_cap(self):
        # Hitting the elif (revived tombstone) branch of _insert_helper: a
        # split node with no mamba state receives one and must trigger the
        # same cap enforcement as a new leaf.
        cache = _build_base_cache(cap=-1)
        _base_insert(cache, [1], 10)
        _base_insert(cache, [1, 2, 3], 11)
        node_a = cache.root_node.children[1]

        cache.mamba_max_states_per_path = 1
        _base_insert(cache, [1, 2], 12)  # splits [2,3]; tail is the split node

        split_node = node_a.children[2]
        self.assertIsNotNone(split_node.mamba_value)  # tail preserved
        self.assertIsNone(node_a.mamba_value)  # shallowest interior pruned
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [10])

    def test_match_prefix_clamps_after_prune(self):
        cache = _build_base_cache(cap=2)
        _build_base_chain(cache, 3)

        # deepest state survives: the full path is still matchable
        full = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3], None)))
        self.assertEqual(len(full.device_indices), 3)
        # the pruned shallow node is no longer a mamba boundary, so a
        # 1-token match clamps to zero
        clamped = cache.match_prefix(MatchPrefixParams(key=RadixKey([1], None)))
        self.assertEqual(len(clamped.device_indices), 0)

    def test_tail_fork_and_locked_nodes_preserved(self):
        cache = _build_base_cache(cap=-1)
        nodes = _build_base_chain(cache, 3)
        fork_child = TreeNode()
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child
        nodes[1].mamba_lock_ref = 1

        cache.mamba_max_states_per_path = 1
        _base_insert(cache, [1, 2, 3, 4], 13)
        tail = nodes[2].children[4]

        self.assertIsNotNone(nodes[0].mamba_value)  # fork
        self.assertIsNotNone(nodes[1].mamba_value)  # mamba-locked
        self.assertIsNone(nodes[2].mamba_value)  # only eligible interior
        self.assertIsNotNone(tail.mamba_value)  # tail
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [12])

    def test_full_locked_node_preserved(self):
        cache = _build_base_cache(cap=-1)
        nodes = _build_base_chain(cache, 2)
        nodes[0].full_lock_ref = 1

        cache.mamba_max_states_per_path = 2
        _base_insert(cache, [1, 2, 3], 12)

        self.assertIsNotNone(nodes[0].mamba_value)
        self.assertIsNone(nodes[1].mamba_value)
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [11])

    def test_base_extra_skip_hook_defaults_to_never_skipping(self):
        cache = _build_base_cache(cap=-1)
        node = TreeNode()
        self.assertFalse(cache._mamba_cap_extra_skip(node))


if __name__ == "__main__":
    unittest.main()
