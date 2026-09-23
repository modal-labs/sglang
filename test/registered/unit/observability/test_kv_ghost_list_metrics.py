"""Pure-CPU unit tests for the KV ghost list (capacity-miss metrics).

``sglang.srt.mem_cache.kv_ghost_list`` remembers the per-page digests of radix
nodes whose last copy was destroyed and recognises them when an insert brings
the same prefix back. These tests cover the digest chain (prefix identity,
split stability, namespaces), the bounded list (hit grouping, TTL, capacity),
the collector contract, and an end-to-end insert / evict / re-insert sequence
on both RadixCache and the default UnifiedRadixCache.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import os
import time
import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_ghost_list import (
    GHOST_LIST_MAX_AUTO_PAGES,
    GhostHit,
    KVGhostList,
    KVGhostTracker,
    build_kv_ghost_tracker,
    chain_page_digests,
    ensure_ghost_digests,
    ghost_namespace_seed,
    resolve_kv_ghost_list_pages,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    KV_AGE_BUCKETS,
    KV_GHOST_DISCARD_REASONS,
    RadixCacheMetricsCollector,
    kv_age_bucket,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def inc(self, value=1):
        self.metric.increments.append((self.labels, value))

    def observe(self, value):
        self.metric.observations.append((self.labels, value))


class _RecordingMetric:
    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.increments = []
        self.observations = []

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


class _RecordingRadixCacheMetricsCollector(RadixCacheMetricsCollector):
    _counter_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


def _key(tokens, extra_key=None, cache_salt=None):
    return RadixKey(
        token_ids=array("q", tokens), extra_key=extra_key, cache_salt=cache_salt
    )


def _node(parent, tokens, **kw):
    node = TreeNode()
    node.parent = parent
    node.key = _key(tokens, **kw)
    return node


def _root():
    root = TreeNode()
    root.key = _key([])
    root.ghost_hash = []
    return root


class TestDigestChain(unittest.TestCase):
    def test_same_prefix_same_digests_regardless_of_node_boundaries(self):
        # [1..8] as one node, or as [1..4] -> [5..8]: identical page digests.
        root_a = _root()
        whole = _node(root_a, range(1, 9))
        root_b = _root()
        head = _node(root_b, range(1, 5))
        tail = _node(head, range(5, 9))
        self.assertEqual(
            ensure_ghost_digests(whole, 1),
            ensure_ghost_digests(head, 1) + ensure_ghost_digests(tail, 1),
        )

    def test_same_tokens_different_prefix_differ(self):
        root = _root()
        a = _node(_node(root, [1, 2]), [7, 8])
        b = _node(_node(root, [3, 4]), [7, 8])
        self.assertNotEqual(ensure_ghost_digests(a, 1), ensure_ghost_digests(b, 1))

    def test_page_size_groups_tokens_and_drops_the_partial_tail(self):
        root = _root()
        node = _node(root, range(10))
        digests = ensure_ghost_digests(node, 4)
        self.assertEqual(len(digests), 2)
        self.assertEqual(digests, chain_page_digests(node.key, bytes(8), 4))

    def test_namespace_changes_the_chain(self):
        root = _root()
        plain = _node(root, [1, 2, 3])
        lora = _node(root, [1, 2, 3], extra_key="lora-a")
        salted = _node(root, [1, 2, 3], cache_salt="s")
        d = {
            "plain": ensure_ghost_digests(plain, 1),
            "lora": ensure_ghost_digests(lora, 1),
            "salt": ensure_ghost_digests(salted, 1),
        }
        self.assertEqual(len({tuple(v) for v in d.values()}), 3)
        self.assertNotEqual(
            ghost_namespace_seed(None, None), ghost_namespace_seed("", None)
        )
        self.assertNotEqual(
            ghost_namespace_seed("a", None), ghost_namespace_seed(None, "a")
        )

    def test_lazy_fill_walks_missing_ancestors(self):
        root = _root()
        a = _node(root, [1, 2])
        b = _node(a, [3, 4])
        c = _node(b, [5, 6])
        # Only the leaf is asked for; the ancestors get filled on the way.
        leaf = ensure_ghost_digests(c, 1)
        self.assertIsNotNone(a.ghost_hash)
        self.assertIsNotNone(b.ghost_hash)
        fresh = _node(root, [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            a.ghost_hash + b.ghost_hash + leaf, ensure_ghost_digests(fresh, 1)
        )

    def test_radix_cache_split_partitions_digests(self):
        cache = RadixCache.create_simulated(page_size=1)
        cache.insert(InsertParams(key=_key([1, 2, 3, 4])))
        node = cache.root_node.children[
            cache.root_node.children.keys().__iter__().__next__()
        ]
        before = ensure_ghost_digests(node, 1)
        new_node = cache._split_node(node.key, node, 2)
        self.assertEqual(new_node.ghost_hash, before[:2])
        self.assertEqual(node.ghost_hash, before[2:])


class TestKVGhostList(unittest.TestCase):
    def test_hit_pops_and_reports_delay(self):
        ghosts = KVGhostList(capacity=10, ttl_seconds=100.0)
        ghosts.record([1, 2, 3], now=10.0)
        self.assertEqual(len(ghosts), 3)
        self.assertEqual(ghosts.match([1, 2, 3], now=15.0), [GhostHit(5.0, 3, 5.0)])
        self.assertEqual(len(ghosts), 0)
        # Consumed: the same prefix coming back again is not a second hit.
        self.assertEqual(ghosts.match([1, 2, 3], now=16.0), [])

    def test_partial_and_grouped_hits(self):
        ghosts = KVGhostList(capacity=10, ttl_seconds=100.0)
        ghosts.record([1, 2], now=10.0)
        ghosts.record([3], now=20.0)
        # Pages evicted together form one hit; a miss in the middle splits runs.
        self.assertEqual(
            ghosts.match([1, 2, 3, 99, 5], now=30.0),
            [GhostHit(20.0, 2, 20.0), GhostHit(10.0, 1, 10.0)],
        )

    def test_ttl_expires_on_record_and_on_match(self):
        ghosts = KVGhostList(capacity=10, ttl_seconds=10.0)
        ghosts.record([1], now=0.0)
        # Stale but not yet purged: a match does not count it.
        self.assertEqual(ghosts.match([1], now=11.0), [])
        ghosts.record([2], now=0.0)
        ghosts.record([3], now=5.0)
        # At 16.0 both the 0.0 and the 5.0 entries are past the 10 s TTL.
        self.assertEqual(ghosts.record([4], now=16.0), (2, 0))
        self.assertEqual(len(ghosts), 1)
        self.assertEqual(ghosts.match([4], now=17.0), [GhostHit(1.0, 1, 1.0)])

    def test_capacity_drops_oldest_first(self):
        ghosts = KVGhostList(capacity=2, ttl_seconds=100.0)
        ghosts.record([1], now=1.0)
        ghosts.record([2], now=2.0)
        self.assertEqual(ghosts.record([3], now=3.0), (0, 1))
        self.assertEqual(ghosts.match([1], now=4.0), [])
        self.assertEqual(
            ghosts.match([2, 3], now=4.0),
            [GhostHit(2.0, 1, 2.0), GhostHit(1.0, 1, 1.0)],
        )

    def test_re_record_refreshes_time_and_position(self):
        ghosts = KVGhostList(capacity=2, ttl_seconds=100.0)
        ghosts.record([1], now=1.0)
        ghosts.record([2], now=2.0)
        ghosts.record([1], now=3.0)
        # 2 is now the oldest and goes first under capacity pressure.
        ghosts.record([3], now=4.0)
        self.assertEqual(ghosts.match([2], now=5.0), [])
        self.assertEqual(ghosts.match([1], now=5.0), [GhostHit(2.0, 1, 2.0)])

    def test_idle_runs_from_last_access(self):
        ghosts = KVGhostList(capacity=10, ttl_seconds=100.0)
        ghosts.record([1, 2], now=10.0, last_access=4.0)
        self.assertEqual(ghosts.match([1, 2], now=15.0), [GhostHit(5.0, 2, 11.0)])
        # A last access after the eviction time (clock skew) clamps to it.
        ghosts.record([3], now=10.0, last_access=12.0)
        self.assertEqual(ghosts.match([3], now=15.0), [GhostHit(5.0, 1, 5.0)])

    def test_same_eviction_time_different_last_access_splits_runs(self):
        # Two nodes dropped in one evict() pass share ``now`` but not their
        # last access; they are two fragments, each with its own idle time.
        ghosts = KVGhostList(capacity=10, ttl_seconds=100.0)
        ghosts.record([1], now=10.0, last_access=2.0)
        ghosts.record([2], now=10.0, last_access=8.0)
        self.assertEqual(
            ghosts.match([1, 2], now=20.0),
            [GhostHit(10.0, 1, 18.0), GhostHit(10.0, 1, 12.0)],
        )

    def test_trigger_travels_with_the_entry_and_splits_runs(self):
        ghosts = KVGhostList(capacity=10, ttl_seconds=100.0)
        # Same eviction pass and last access, different walks: two fragments.
        ghosts.record([1], now=10.0, last_access=5.0, trigger="mamba")
        ghosts.record([2], now=10.0, last_access=5.0, trigger="full")
        self.assertEqual(
            ghosts.match([1, 2], now=20.0),
            [GhostHit(10.0, 1, 15.0, "mamba"), GhostHit(10.0, 1, 15.0, "full")],
        )

    def test_ttl_runs_from_eviction_not_last_access(self):
        ghosts = KVGhostList(capacity=10, ttl_seconds=10.0)
        ghosts.record([1], now=100.0, last_access=0.0)
        self.assertEqual(ghosts.match([1], now=105.0), [GhostHit(5.0, 1, 105.0)])


class TestSizing(unittest.TestCase):
    class _Alloc:
        def __init__(self, size):
            self.size = size

    def test_auto_size_is_pool_pages_capped(self):
        self.assertEqual(resolve_kv_ghost_list_pages(self._Alloc(6400), 64, -1), 100)
        self.assertEqual(
            resolve_kv_ghost_list_pages(self._Alloc(10**9), 1, -1),
            GHOST_LIST_MAX_AUTO_PAGES,
        )
        self.assertEqual(
            resolve_kv_ghost_list_pages(None, 64, -1), GHOST_LIST_MAX_AUTO_PAGES
        )

    def test_explicit_size_wins_and_zero_disables(self):
        self.assertEqual(resolve_kv_ghost_list_pages(self._Alloc(6400), 64, 5), 5)
        self.assertEqual(resolve_kv_ghost_list_pages(self._Alloc(6400), 64, 0), 0)
        collector = _RecordingRadixCacheMetricsCollector(labels={"cache_type": "x"})
        with mock.patch.dict(os.environ, {"SGLANG_KV_GHOST_LIST_PAGES": "0"}):
            self.assertIsNone(build_kv_ghost_tracker(self._Alloc(6400), 64, collector))
        with mock.patch.dict(os.environ, {"SGLANG_KV_GHOST_LIST_PAGES": "-1"}):
            tracker = build_kv_ghost_tracker(self._Alloc(6400), 64, collector)
        self.assertIsInstance(tracker, KVGhostTracker)
        self.assertEqual(tracker.ghost_list.capacity, 100)
        self.assertIsNone(build_kv_ghost_tracker(self._Alloc(6400), 64, None))


class TestCollectorContract(unittest.TestCase):
    def setUp(self):
        self.collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "RadixCache"}
        )

    def test_metric_names_and_labelnames(self):
        c = self.collector
        self.assertEqual(
            c.kv_reinsert_after_evict_seconds.name,
            "sglang:kv_reinsert_after_evict_seconds",
        )
        self.assertEqual(
            c.kv_recomputed_after_evict_tokens.name,
            "sglang:kv_recomputed_after_evict_tokens_total",
        )
        self.assertEqual(
            c.kv_recomputed_after_evict_tokens.labelnames, ("cache_type", "age_le")
        )
        self.assertEqual(
            c.kv_recomputed_idle_seconds.name, "sglang:kv_recomputed_idle_seconds"
        )
        self.assertEqual(
            c.kv_recomputed_idle_tokens.name,
            "sglang:kv_recomputed_idle_tokens_total",
        )
        self.assertEqual(
            c.kv_recomputed_idle_tokens.labelnames,
            ("cache_type", "age_le", "trigger"),
        )
        self.assertEqual(c.kv_inserted_tokens.name, "sglang:kv_inserted_tokens_total")
        self.assertEqual(
            c.kv_ghost_list_recorded_pages.name,
            "sglang:kv_ghost_list_recorded_pages_total",
        )
        self.assertEqual(
            c.kv_ghost_list_discarded_pages.labelnames, ("cache_type", "reason")
        )
        self.assertEqual(
            set(c._kv_ghost_discarded_children), set(KV_GHOST_DISCARD_REASONS)
        )

    def test_reinsert_observation_and_token_bucket(self):
        c = self.collector
        c.observe_kv_reinsert_after_evict(90.0, 128, 400.0)
        self.assertEqual(
            c.kv_reinsert_after_evict_seconds.observations,
            [({"cache_type": "RadixCache"}, 90.0)],
        )
        self.assertEqual(
            c.kv_recomputed_after_evict_tokens.increments,
            [({"cache_type": "RadixCache", "age_le": "120"}, 128)],
        )
        # The idle-time pair is bucketed on the gap since last access.
        self.assertEqual(
            c.kv_recomputed_idle_seconds.observations,
            [({"cache_type": "RadixCache"}, 400.0)],
        )
        self.assertEqual(
            c.kv_recomputed_idle_tokens.increments,
            [
                (
                    {
                        "cache_type": "RadixCache",
                        "age_le": kv_age_bucket(400.0),
                        "trigger": "full",
                    },
                    128,
                )
            ],
        )
        c.observe_kv_reinsert_after_evict(
            KV_AGE_BUCKETS[-1] + 1, 1, KV_AGE_BUCKETS[-1] + 1
        )
        self.assertEqual(
            c.kv_recomputed_after_evict_tokens.increments[-1][0]["age_le"], "+Inf"
        )
        self.assertEqual(
            c.kv_recomputed_idle_tokens.increments[-1][0]["age_le"], "+Inf"
        )
        c.observe_kv_reinsert_after_evict(10.0, 64, 30.0, "mamba")
        self.assertEqual(
            c.kv_recomputed_idle_tokens.increments[-1][0]["trigger"], "mamba"
        )

    def test_tracker_routes_to_collector(self):
        c = self.collector
        tracker = KVGhostTracker(KVGhostList(capacity=2, ttl_seconds=100.0), 1, c)
        root = _root()
        a = _node(root, [1, 2, 3])
        a.last_access_time = 4.0
        tracker.on_dropped(a, now=10.0)
        self.assertEqual(c.kv_ghost_list_recorded_pages.increments[-1][1], 3)
        self.assertEqual(
            c.kv_ghost_list_discarded_pages.increments,
            [({"cache_type": "RadixCache", "reason": "capacity"}, 1)],
        )
        b = _node(_root(), [1, 2, 3])
        tracker.on_inserted(b, now=25.0)
        self.assertEqual(
            c.kv_inserted_tokens.increments, [({"cache_type": "RadixCache"}, 3)]
        )
        # Page 1 fell to the capacity cap; pages 2 and 3 came back as one fragment.
        self.assertEqual(c.kv_reinsert_after_evict_seconds.observations[-1][1], 15.0)
        self.assertEqual(c.kv_recomputed_idle_seconds.observations[-1][1], 21.0)
        self.assertEqual(c.kv_recomputed_idle_tokens.increments[-1][1], 2)
        self.assertEqual(c.kv_recomputed_after_evict_tokens.increments[-1][1], 2)
        self.assertEqual(
            c.kv_ghost_list_discarded_pages.increments[-1],
            ({"cache_type": "RadixCache", "reason": "reinsert"}, 2),
        )
        tracker.on_dropped(a, now=30.0)
        tracker.reset()
        self.assertEqual(
            c.kv_ghost_list_discarded_pages.increments[-1],
            ({"cache_type": "RadixCache", "reason": "flush"}, 2),
        )
        self.assertEqual(len(tracker.ghost_list), 0)


def _pools(page_size):
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=page_size,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    return req_to_token_pool, allocator


def _install_tracker(collector, page_size=1, last_access_attr="last_access_time"):
    return KVGhostTracker(
        KVGhostList(capacity=64, ttl_seconds=3600.0),
        page_size,
        collector,
        last_access_attr,
    )


def _assert_idle_spans_eviction(test, collector, t_insert, t_reinsert):
    """The recompute's idle time covers the eviction-to-reinsert delay and is
    no longer than the whole insert-to-reinsert span."""
    ((_, delay),) = collector.kv_reinsert_after_evict_seconds.observations
    ((_, idle),) = collector.kv_recomputed_idle_seconds.observations
    test.assertGreaterEqual(idle, delay)
    test.assertLessEqual(idle, t_reinsert - t_insert)


class TestRadixCacheGhostList(unittest.TestCase):
    """insert -> evict -> re-insert on a CPU RadixCache records one reinsert
    whose token weight is the recreated node; unrelated inserts do not."""

    def _build_cache(self):
        req_to_token_pool, allocator = _pools(page_size=1)
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "RadixCache"}
        )
        cache.metrics_collector = collector
        cache.kv_ghost = _install_tracker(collector)
        return cache, allocator, collector

    def test_recompute_after_eviction_is_one_reinsert(self):
        cache, allocator, collector = self._build_cache()
        tokens = array("q", [1, 2, 3, 4])
        t_insert = time.monotonic()
        cache.insert(
            InsertParams(key=RadixKey(token_ids=tokens), value=allocator.alloc(4))
        )
        self.assertEqual(collector.kv_inserted_tokens.increments[-1][1], 4)
        self.assertEqual(collector.kv_reinsert_after_evict_seconds.observations, [])

        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(collector.kv_ghost_list_recorded_pages.increments[-1][1], 4)
        self.assertEqual(len(cache.kv_ghost.ghost_list), 4)

        # A different prompt after the eviction is a cold miss, not a reinsert.
        other = array("q", [9, 9, 9])
        cache.insert(
            InsertParams(key=RadixKey(token_ids=other), value=allocator.alloc(3))
        )
        self.assertEqual(collector.kv_reinsert_after_evict_seconds.observations, [])

        time.sleep(0.01)
        cache.insert(
            InsertParams(key=RadixKey(token_ids=tokens), value=allocator.alloc(4))
        )
        _assert_idle_spans_eviction(self, collector, t_insert, time.monotonic())
        obs = collector.kv_reinsert_after_evict_seconds.observations
        self.assertEqual(len(obs), 1)
        self.assertGreaterEqual(obs[0][1], 0.0)
        self.assertEqual(
            collector.kv_recomputed_after_evict_tokens.increments[-1][1], 4
        )
        self.assertEqual(
            collector.kv_ghost_list_discarded_pages.increments[-1],
            ({"cache_type": "RadixCache", "reason": "reinsert"}, 4),
        )
        self.assertEqual(len(cache.kv_ghost.ghost_list), 0)

    def test_extension_of_a_resident_prefix_is_not_a_reinsert(self):
        cache, allocator, collector = self._build_cache()
        base = array("q", [1, 2, 3, 4])
        cache.insert(
            InsertParams(key=RadixKey(token_ids=base), value=allocator.alloc(4))
        )
        cache.evict(EvictParams(num_tokens=4))
        # Only the suffix [3, 4] of the evicted node comes back, under a new head.
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [7, 8, 3, 4])),
                value=allocator.alloc(4),
            )
        )
        self.assertEqual(collector.kv_reinsert_after_evict_seconds.observations, [])
        # The full prefix does, and only its 4 tokens count.
        cache.insert(
            InsertParams(key=RadixKey(token_ids=base), value=allocator.alloc(4))
        )
        self.assertEqual(
            collector.kv_recomputed_after_evict_tokens.increments[-1][1], 4
        )

    def test_partial_reinsert_after_split_eviction(self):
        cache, allocator, collector = self._build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 3, 4])),
                value=allocator.alloc(4),
            )
        )
        # Splits [1,2,3,4] into [1,2] -> [3,4]; evicting 2 tokens drops the leaf.
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 5, 6])),
                value=allocator.alloc(4),
            )
        )
        cache.evict(EvictParams(num_tokens=2))
        recorded = collector.kv_ghost_list_recorded_pages.increments[-1][1]
        self.assertEqual(recorded, 2)
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 3, 4])),
                value=allocator.alloc(4),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 5, 6])),
                value=allocator.alloc(4),
            )
        )
        # Exactly one of the two leaves was dropped and exactly it came back.
        self.assertEqual(len(collector.kv_reinsert_after_evict_seconds.observations), 1)
        self.assertEqual(
            collector.kv_recomputed_after_evict_tokens.increments[-1][1], 2
        )

    def test_reset_flushes_ghosts(self):
        cache, allocator, collector = self._build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2])), value=allocator.alloc(2)
            )
        )
        cache.evict(EvictParams(num_tokens=2))
        cache.reset()
        self.assertEqual(
            collector.kv_ghost_list_discarded_pages.increments[-1],
            ({"cache_type": "RadixCache", "reason": "flush"}, 2),
        )
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2])), value=allocator.alloc(2)
            )
        )
        self.assertEqual(collector.kv_reinsert_after_evict_seconds.observations, [])


class TestUnifiedRadixCacheGhostList(unittest.TestCase):
    """Same sequence on the default UnifiedRadixCache (Python tree core)."""

    def _build_cache(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )
        req_to_token_pool, allocator = _pools(page_size=1)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            )
        )
        collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "UnifiedRadixCache"}
        )
        # What UnifiedRadixCache.__init__ does when metrics are enabled.
        cache.metrics_collector = collector
        cache.tree_core.kv_ghost = _install_tracker(
            collector, last_access_attr="last_access_wall"
        )
        return cache, allocator, collector

    def test_recompute_after_eviction_is_one_reinsert(self):
        cache, allocator, collector = self._build_cache()
        tokens = array("q", [1, 2, 3, 4])
        t_insert = time.monotonic()
        cache.insert(
            InsertParams(key=RadixKey(token_ids=tokens), value=allocator.alloc(4))
        )
        self.assertEqual(collector.kv_inserted_tokens.increments[-1][1], 4)

        time.sleep(0.01)
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(collector.kv_ghost_list_recorded_pages.increments[-1][1], 4)

        cache.insert(
            InsertParams(key=RadixKey(token_ids=tokens), value=allocator.alloc(4))
        )
        # The idle time is on the wall clock (last_access_wall), so it spans the
        # 10 ms before the eviction; the ordering counter would read ~0 or huge.
        _assert_idle_spans_eviction(self, collector, t_insert, time.monotonic())
        ((_, idle),) = collector.kv_recomputed_idle_seconds.observations
        self.assertGreaterEqual(idle, 0.01)
        obs = collector.kv_reinsert_after_evict_seconds.observations
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0][0], {"cache_type": "UnifiedRadixCache"})
        self.assertEqual(
            collector.kv_recomputed_after_evict_tokens.increments[-1][1], 4
        )
        self.assertEqual(len(cache.tree_core.kv_ghost.ghost_list), 0)

    def test_split_keeps_digests_aligned_with_a_fresh_tree(self):
        cache, allocator, collector = self._build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 3, 4])),
                value=allocator.alloc(4),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 5, 6])),
                value=allocator.alloc(4),
            )
        )
        # Evict everything, then bring one of the two prompts back whole.
        cache.evict(EvictParams(num_tokens=6))
        self.assertEqual(
            sum(v for _, v in collector.kv_ghost_list_recorded_pages.increments), 6
        )
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 5, 6])),
                value=allocator.alloc(4),
            )
        )
        # [1,2] and [5,6] were dropped as separate fragments (the shared head
        # went last), so they come back as two hits totalling 4 tokens.
        self.assertEqual(
            sum(v for _, v in collector.kv_recomputed_after_evict_tokens.increments), 4
        )
        self.assertEqual(len(cache.tree_core.kv_ghost.ghost_list), 2)


if __name__ == "__main__":
    unittest.main()
