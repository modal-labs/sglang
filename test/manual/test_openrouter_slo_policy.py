"""Unit tests for --schedule-policy openrouter_slo and the slo_429 admission
predicate.

The policy sorts the waiting queue by virtual arrival time:
    wait_queue_entry_time + slope * min_uncached_seen
i.e. FCFS handicapped by uncached work, matching a TTFT SLO line of
base + slope * tokens.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase

SLOPE_S = 0.001  # 1 ms per uncached token


def _make_req(rid, num_tokens, entry_time, matched=0):
    req = Req(
        rid,
        " ".join(str(i) for i in range(num_tokens)),
        array("q", range(num_tokens)),
        SamplingParams(),
    )
    req.time_stats.wait_queue_entry_time = entry_time
    req.num_matched_prefix_tokens = matched
    return req


def _sort(queue, deprioritized=frozenset()):
    SchedulePolicy._sort_by_openrouter_slo(queue, deprioritized, SLOPE_S)
    return [r.rid for r in queue]


class TestOpenrouterSloSort(CustomTestCase):

    def test_policy_resolves_cache_aware(self):
        policy = SchedulePolicy(
            policy="openrouter_slo",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAwarePolicy.OPENROUTER_SLO)

    def test_fully_cached_is_fcfs(self):
        # Fully cached requests carry no handicap: pure arrival order,
        # regardless of context length.
        a = _make_req("a", 1000, entry_time=1.0, matched=1000)
        b = _make_req("b", 10, entry_time=2.0, matched=10)
        self.assertEqual(_sort([b, a]), ["a", "b"])

    def test_cold_request_handicapped_by_uncached_tokens(self):
        # Cold 5000-token req arriving at t=0 has virtual arrival 5.0;
        # hot req arriving 3s later has virtual arrival ~3.0 and goes first;
        # hot req arriving 6s later has virtual arrival ~6.0 and does not.
        cold = _make_req("cold", 5000, entry_time=0.0, matched=0)
        hot_early = _make_req("hot_early", 5000, entry_time=3.0, matched=5000)
        hot_late = _make_req("hot_late", 5000, entry_time=6.0, matched=5000)
        self.assertEqual(
            _sort([cold, hot_late, hot_early]), ["hot_early", "cold", "hot_late"]
        )

    def test_small_cold_beats_large_cold_at_same_arrival(self):
        small = _make_req("small", 2000, entry_time=1.0, matched=0)
        large = _make_req("large", 100_000, entry_time=1.0, matched=0)
        self.assertEqual(_sort([large, small]), ["small", "large"])

    def test_tie_break_prefers_longer_cached_prefix(self):
        # Same virtual arrival: 1000 uncached each, same entry time. The one
        # with the larger cached prefix goes first.
        big_cache = _make_req("big_cache", 50_000, entry_time=1.0, matched=49_000)
        no_cache = _make_req("no_cache", 1000, entry_time=1.0, matched=0)
        self.assertEqual(_sort([no_cache, big_cache]), ["big_cache", "no_cache"])

    def test_deprioritized_sort_last(self):
        follower = _make_req("follower", 10, entry_time=0.0, matched=10)
        cold = _make_req("cold", 50_000, entry_time=0.0, matched=0)
        self.assertEqual(
            _sort([follower, cold], deprioritized={"follower"}),
            ["cold", "follower"],
        )

    def test_running_min_ignores_eviction(self):
        # A request whose cache is evicted while waiting keeps its original
        # (smaller) handicap: eviction can never demote it.
        req = _make_req("r", 10_000, entry_time=0.0, matched=9_000)
        self.assertEqual(SchedulePolicy.update_min_uncached_seen(req), 1000)
        req.num_matched_prefix_tokens = 0  # evicted
        self.assertEqual(SchedulePolicy.update_min_uncached_seen(req), 1000)
        # An improved match promotes it.
        req.num_matched_prefix_tokens = 9_500
        self.assertEqual(SchedulePolicy.update_min_uncached_seen(req), 500)

    def test_calc_priority_end_to_end(self):
        tree_cache = RadixCache.create_simulated()
        cold = _make_req("cold", 5000, entry_time=0.0)
        hot = _make_req("hot", 5000, entry_time=3.0)
        queue = [cold, hot]

        policy = SchedulePolicy(
            policy="openrouter_slo",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        fake_args = SimpleNamespace(
            slo_ttft_slope_ms_per_uncached_token=1.0,
            disaggregation_mode="null",
        )
        with mock.patch(
            "sglang.srt.managers.schedule_policy.get_server_args",
            return_value=fake_args,
        ):
            # Simulated tree: both requests match nothing, so both are fully
            # uncached with equal handicap -> arrival order.
            policy.calc_priority(queue)
        self.assertEqual([r.rid for r in queue], ["cold", "hot"])
        self.assertEqual(cold.min_uncached_seen, 5000)


class TestSloAdmissionCheck(CustomTestCase):
    """Tests Scheduler._slo_admission_check via a minimal fake scheduler."""

    def _fake_scheduler(self, waiting_queue=(), r_est=10_000.0, enforce=False):
        from sglang.srt.managers.scheduler import Scheduler

        sent = []
        fake = SimpleNamespace(
            server_args=SimpleNamespace(
                schedule_policy="openrouter_slo",
                slo_prefill_tokens_per_s=r_est,
                slo_ttft_slope_ms_per_uncached_token=1.0,
                slo_ttft_base_s=10.0,
                slo_429_margin_s=1.0,
                openrouter_slo_429=enforce,
            ),
            tree_cache=RadixCache.create_simulated(),
            chunked_req=None,
            waiting_queue=list(waiting_queue),
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda out, req: sent.append(out)
                )
            ),
        )
        return (
            lambda req: Scheduler._slo_admission_check(fake, req),
            sent,
        )

    def test_admits_when_queue_empty(self):
        check, sent = self._fake_scheduler(enforce=True)
        req = _make_req("r", 5000, entry_time=0.0)
        self.assertFalse(check(req))
        self.assertEqual(sent, [])

    @staticmethod
    def _blocker():
        # 5M queued uncached tokens at 10K tok/s = 500s predicted wait, far
        # over any limit. Entry time set so its virtual arrival
        # (entry + 1ms * 5M = entry + 5000s) is earlier than an incoming
        # request's (perf_counter-based) virtual arrival.
        import time

        blocker = _make_req(
            "blocker", 100, entry_time=time.perf_counter() - 6000.0, matched=0
        )
        blocker.min_uncached_seen = 5_000_000
        return blocker

    def test_shadow_mode_never_rejects(self):
        check, sent = self._fake_scheduler(
            waiting_queue=[self._blocker()], enforce=False
        )
        req = _make_req("r", 5000, entry_time=0.0)
        self.assertFalse(check(req))
        self.assertEqual(sent, [])

    def test_enforce_rejects_predicted_miss(self):
        check, sent = self._fake_scheduler(
            waiting_queue=[self._blocker()], enforce=True
        )
        req = _make_req("r", 5000, entry_time=0.0)
        self.assertTrue(check(req))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].finished_reason["status_code"].value, 429)

    def test_later_virtual_arrival_work_not_counted(self):
        # The blocker's virtual arrival is later than the incoming request's,
        # so its work does not delay the incoming request.
        blocker = _make_req("blocker", 100, entry_time=1e12, matched=0)
        blocker.min_uncached_seen = 5_000_000
        check, sent = self._fake_scheduler(waiting_queue=[blocker], enforce=True)
        req = _make_req("r", 5000, entry_time=0.0)
        self.assertFalse(check(req))

    def test_disabled_without_throughput_estimate(self):
        check, sent = self._fake_scheduler(r_est=0.0, enforce=True)
        req = _make_req("r", 5000, entry_time=0.0)
        self.assertFalse(check(req))

    def test_never_rejects_requests_with_output(self):
        # Retracted / requeued requests already streamed tokens; a 429 there
        # would kill a stream mid-generation.
        check, sent = self._fake_scheduler(
            waiting_queue=[self._blocker()], enforce=True
        )
        req = _make_req("r", 5000, entry_time=0.0)
        req.output_ids = [1, 2, 3]
        self.assertFalse(check(req))
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
