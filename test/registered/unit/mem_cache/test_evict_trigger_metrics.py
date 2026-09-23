"""CPU tests for the eviction-trigger metrics on the unified radix cache.

A Full + Mamba UnifiedRadixCache frees KV and Mamba state through several
walks: the Full LRU, the Mamba LRU (which deletes a leaf along with its Full
KV, or drops only the state on an interior node), the per-path Mamba state cap,
and the shared-pool donor walk. These tests drive real evictions and check that
``sglang:kv_evicted_tokens_by_trigger_total``,
``sglang:mamba_states_evicted_total`` and the Mamba-gated miss cause attribute
each one to the walk that caused it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import test_unified_radix_cache_unittest as unified_suite
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    get_mamba_cache_miss_cause,
)
from sglang.srt.mem_cache.kv_ghost_list import KVGhostList, KVGhostTracker
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    KV_EVICT_TRIGGERS,
    MAMBA_CACHE_MISS_CAUSES,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
)
from sglang.srt.sampling.sampling_params import SamplingParams


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


def _trigger_samples(collector):
    """(tier, outcome, trigger, mamba_state, tokens) per eviction sample."""
    return [
        (
            labels["tier"],
            labels["outcome"],
            labels["trigger"],
            labels["mamba_state"],
            n,
        )
        for labels, n in collector.kv_evicted_tokens_by_trigger.increments
    ]


def _state_samples(collector):
    return [
        (labels["trigger"], labels["node"])
        for labels, _ in collector.mamba_states_evicted.increments
    ]


class TestEvictTriggerOnMambaCache(unittest.TestCase):
    cfg = CacheConfig(page_size=1, components=(ComponentType.FULL, ComponentType.MAMBA))

    def setUp(self):
        # The shared fixture asks for an accelerator; these pools and trees are
        # plain tensors and run on CPU.
        with mock.patch.object(unified_suite, "get_device", return_value="cpu"):
            self.cache, self.allocator, self.req_to_token_pool = build_fixture(self.cfg)
        self.collector = _RecordingRadixCacheMetricsCollector(
            labels={"cache_type": "UnifiedRadixCache"}
        )
        # What UnifiedRadixCache.__init__ installs when metrics are enabled.
        self.cache.metrics_collector = self.collector
        core = self.cache.tree_core
        core.kv_age_observer = self.cache._observe_kv_age_event
        core.mamba_evict_observer = self.collector.increment_mamba_state_evicted
        self._rid = 0

    def _insert(self, tokens):
        req = Req(
            rid=self._rid,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self._rid += 1
        self.req_to_token_pool.alloc([req])
        value = self.allocator.alloc(len(tokens))
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value[: len(tokens)],
                mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
            )
        )

    def _match(self, tokens):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )

    def test_mamba_lru_on_interior_drops_only_the_state(self):
        # root -> a[1,2,3] -> b[4,5,6]; a's state is the LRU tail.
        self._insert([1, 2, 3])
        self._insert([1, 2, 3, 4, 5, 6])
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))

        self.assertEqual(_state_samples(self.collector), [("mamba", "interior")])
        # No KV left the tree: the interior node keeps its Full KV.
        self.assertEqual(_trigger_samples(self.collector), [])
        self.assertEqual(self._match([1, 2, 3, 4, 5, 6]).full_kv_hit_length, 6)
        self.assertEqual(self.cache.tree_core.evict_trigger, "other")

    def test_mamba_lru_on_leaf_deletes_its_kv(self):
        self._insert([1, 2, 3])
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))

        self.assertEqual(_state_samples(self.collector), [("mamba", "leaf")])
        self.assertEqual(
            _trigger_samples(self.collector),
            [("device", "dropped", "mamba", "present", 3)],
        )
        self.assertEqual(self._match([1, 2, 3]).full_kv_hit_length, 0)

    def test_full_lru_reports_stateless_leaves(self):
        # Drop a's state first (interior), then evict all KV with the Full walk:
        # b goes with its state, then a goes as a leaf that no longer has one.
        self._insert([1, 2, 3])
        self._insert([1, 2, 3, 4, 5, 6])
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.cache.evict(EvictParams(num_tokens=6))

        self.assertEqual(
            _trigger_samples(self.collector),
            [
                ("device", "dropped", "full", "present", 3),
                ("device", "dropped", "full", "absent", 3),
            ],
        )
        self.assertEqual(
            _state_samples(self.collector),
            [("mamba", "interior"), ("full", "leaf")],
        )

    def test_trigger_sums_match_kv_age_tokens(self):
        self._insert([1, 2, 3])
        self._insert([1, 2, 3, 4, 5, 6])
        self._insert([7, 8])
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.cache.evict(EvictParams(num_tokens=8))

        by_trigger = sum(n for *_, n in _trigger_samples(self.collector))
        by_age = sum(
            n
            for labels, n in self.collector.kv_age_tokens.increments
            if labels["event"] == "evict"
        )
        self.assertEqual(by_trigger, by_age)
        self.assertEqual(by_trigger, 8)

    def test_path_cap_labels_its_evictions(self):
        mamba = self.cache.tree_core.components_by_type[ComponentType.MAMBA]
        mamba.mamba_max_states_per_path = 1
        self._insert([1, 2, 3])
        self._insert([1, 2, 3, 4, 5, 6])

        self.assertEqual(
            _state_samples(self.collector), [("mamba_path_cap", "interior")]
        )
        self.assertEqual(self.cache.tree_core.evict_trigger, "other")

    def test_ghost_list_attributes_recompute_to_the_mamba_walk(self):
        tracker = KVGhostTracker(
            KVGhostList(capacity=64, ttl_seconds=3600.0),
            1,
            self.collector,
            last_access_attr="last_access_wall",
        )
        self.cache.tree_core.kv_ghost = tracker
        self._insert([1, 2, 3])
        # The Mamba LRU deletes the leaf and its Full KV ...
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        # ... and the same prefix comes back and is recomputed.
        self._insert([1, 2, 3])

        self.assertEqual(
            [
                (labels["trigger"], n)
                for labels, n in self.collector.kv_recomputed_idle_tokens.increments
            ],
            [("mamba", 3)],
        )

    def test_gap_flag_distinguishes_evicted_from_never_saved(self):
        # root -> a[1,2,3] -> b[4,5,6] -> c[7,8,9]; drop a's and b's states.
        self._insert([1, 2, 3])
        self._insert([1, 2, 3, 4, 5, 6])
        self._insert([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=2))

        # Diverging after b: the Full KV reaches 6 tokens, the last reusable
        # state is the root, and a / b once had states.
        evicted = self._match([1, 2, 3, 4, 5, 6, 99])
        self.assertEqual(evicted.full_kv_hit_length, 6)
        self.assertTrue(evicted.mamba_state_evicted_in_gap)
        self.assertEqual(get_mamba_cache_miss_cause(evicted), "state_evicted")

        # Diverging inside a fresh node: the split point never had a state.
        self._insert([20, 21, 22, 23])
        never = self._match([20, 21, 99])
        self.assertEqual(never.full_kv_hit_length, 2)
        self.assertFalse(never.mamba_state_evicted_in_gap)
        self.assertEqual(get_mamba_cache_miss_cause(never), "never_saved")

    def test_a_new_state_clears_the_flag(self):
        self._insert([1, 2, 3])
        self._insert([1, 2, 3, 4, 5, 6])
        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        # A request ending at a again commits a fresh state there.
        self._insert([1, 2, 3])

        result = self._match([1, 2, 3, 99])
        self.assertFalse(result.mamba_state_evicted_in_gap)
        self.assertEqual(result.full_kv_hit_length, 3)


class TestEvictTriggerMapping(unittest.TestCase):
    def test_walk_triggers(self):
        f = UnifiedRadixCache._evict_trigger_for
        self.assertEqual(f(ComponentType.FULL, None), "full")
        self.assertEqual(
            f(ComponentType.FULL, {ComponentType.FULL: (ComponentType.FULL, 8)}),
            "full",
        )
        # evict_for_alloc's donor walk: Full KV freed to fund Mamba capacity.
        self.assertEqual(
            f(ComponentType.FULL, {ComponentType.FULL: (ComponentType.MAMBA, 2)}),
            "mamba_donor",
        )
        self.assertEqual(f(ComponentType.SWA, None), "swa")
        self.assertEqual(f(ComponentType.MAMBA, None), "mamba")
        for ct in (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA):
            self.assertIn(f(ct, None), KV_EVICT_TRIGGERS)


class TestCollectorDefaults(unittest.TestCase):
    def test_single_component_caches_default_by_tier(self):
        c = _RecordingRadixCacheMetricsCollector(labels={"cache_type": "RadixCache"})
        c.observe_kv_eviction(30.0, 40.0, 2, 64, "device", "dropped")
        c.observe_kv_eviction(900.0, 1000.0, 2, 32, "host", "dropped")
        self.assertEqual(
            [
                (l["tier"], l["trigger"], l["mamba_state"], l["age_le"], n)
                for l, n in c.kv_evicted_tokens_by_trigger.increments
            ],
            [
                ("device", "full", "none", "30", 64),
                ("host", "host", "none", "1200", 32),
            ],
        )


class TestMambaMissCause(unittest.TestCase):
    def test_prefill_stats_split_by_cause(self):
        reqs = [
            SimpleNamespace(
                mamba_cache_miss_tokens=100,
                mamba_cache_miss_cause="state_evicted",
                _mamba_cache_miss_reported=False,
            ),
            SimpleNamespace(
                mamba_cache_miss_tokens=40,
                mamba_cache_miss_cause="never_saved",
                _mamba_cache_miss_reported=False,
            ),
            SimpleNamespace(
                mamba_cache_miss_tokens=60,
                mamba_cache_miss_cause="state_evicted",
                _mamba_cache_miss_reported=False,
            ),
        ]

        class _Adder(SimpleNamespace):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return 0

        stats = PrefillStats.from_adder(
            _Adder(can_run_list=reqs, new_token_ratio=1.0), []
        )
        self.assertEqual(
            (stats.mamba_cache_miss_requests, stats.mamba_cache_miss_tokens), (3, 200)
        )
        self.assertEqual(
            stats.mamba_cache_miss_by_cause,
            {"state_evicted": (2, 160), "never_saved": (1, 40)},
        )

    def test_collector_labels_cause(self):
        c = object.__new__(SchedulerMetricsCollector)
        c.labels = {"model_name": "m"}
        c.mamba_cache_miss_requests_total = _RecordingMetric("r")
        c.mamba_cache_miss_tokens_total = _RecordingMetric("t")
        c.increment_mamba_cache_miss(
            num_requests=2, num_tokens=160, cause="state_evicted"
        )
        self.assertEqual(
            c.mamba_cache_miss_tokens_total.increments,
            [({"model_name": "m", "cause": "state_evicted"}, 160)],
        )
        self.assertEqual(set(MAMBA_CACHE_MISS_CAUSES), {"state_evicted", "never_saved"})


if __name__ == "__main__":
    unittest.main()
