"""Decode-throughput timing must come from the scheduler, not from output delivery.

Drives the real ``_GenerationStreamAccumulator.accept()`` emission rules, the
real ``SchedulerReqTimeStats`` pickle round-trip (cross-process clock rebase)
and the real ``TokenizerManager.collect_metrics`` slice, with a fake clock so
that the scheduler samples one token per 10ms and the API server only sees
the batches the streamer actually emits.
"""

import asyncio
import pickle
import unittest
from array import array
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin import scheduler as dllm_sched
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.metrics_collector import (
    SupportedKwargsFilter,
    TokenizerMetricsCollector,
    filter_supported_kwargs,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FORCE_STREAM_INTERVAL = 50
TOKEN_PERIOD_S = 0.010
# A batch emitted by the scheduler reaches the tokenizer this much later.
IPC_DELAY_S = 0.002


class _FakeReq:
    def __init__(self, rid: str, stream: bool, stream_interval: Optional[int]):
        self.rid = rid
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.stream = stream
        self.sampling_params = SimpleNamespace(
            stream_interval=stream_interval,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids: List[int] = []
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = [1, 2, 3]
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        self._finished = False
        self.time_stats = rts.SchedulerReqTimeStats()
        self.time_stats.enable_metrics = True
        self.time_stats.metrics_collector = mock.Mock()

    @property
    def output_ids_through_stop(self):
        return self.output_ids

    def finished(self):
        return self._finished

    def init_incremental_detokenize(self):
        return self.output_ids, 0

    def check_match_stop_str_prefix(self):
        return False


class _RecordingCollector:
    labels = {"model_name": "m"}

    def __init__(self):
        self.finished = []

    def observe_time_to_first_token(self, *args, **kwargs):
        pass

    def observe_inter_token_latency(self, *args, **kwargs):
        pass

    def observe_one_finished_request(self, labels, *args, **kwargs):
        self.finished.append(kwargs)


class _Sim:
    """One request through scheduler stamping, streamer emission and tokenizer receive."""

    def __init__(self, test: "TestDecodeThroughputFirstToken"):
        self.test = test
        self.now = 100.0
        self.collector = _RecordingCollector()
        self.tm = object.__new__(TokenizerManager)
        self.tm.metrics_collector = self.collector
        self.tm._finished_request_kwargs_filter = SupportedKwargsFilter(
            self.collector.observe_one_finished_request
        )
        self.tm.enable_priority_scheduling = False
        self.tm.disaggregation_mode = DisaggregationMode.NULL

    def _accumulator(self):
        return _GenerationStreamAccumulator(
            return_logprob=False,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=FORCE_STREAM_INTERVAL,
            get_cached_tokens_details=lambda req: None,
        )

    def run(
        self,
        *,
        num_tokens: int,
        stream: bool,
        stream_interval: Optional[int] = None,
        tokens_per_step: int = 1,
        retract_after: Optional[int] = None,
    ):
        req = _FakeReq("r", stream, stream_interval)
        obj = SimpleNamespace(stream=stream, custom_labels=None, sampling_params={})
        state = ReqState([], False, asyncio.Event(), obj, rts.APIServerReqTimeStats())
        state.time_stats.created_time = self.now
        self.delivered_batches = 0
        self.scheduler_first_token_time = None

        with mock.patch.object(rts.time, "perf_counter", lambda: self.now):
            while len(req.output_ids) < num_tokens:
                self.now += TOKEN_PERIOD_S
                if not req.output_ids:
                    # process_batch_result_prefill: stamp, then append token one
                    req.time_stats.set_prefill_finished_time()
                    req.output_ids.append(len(req.output_ids))
                    self.scheduler_first_token_time = self.now
                else:
                    # process_batch_result_decode: extend by the accepted tokens
                    n = min(tokens_per_step, num_tokens - len(req.output_ids))
                    req.output_ids.extend(
                        range(len(req.output_ids), len(req.output_ids) + n)
                    )
                    req.time_stats.set_last_decode_finish_time()
                if len(req.output_ids) >= num_tokens:
                    req._finished = True
                    req.time_stats.set_completion_time()
                self._stream_output(req, state)
                if retract_after is not None and len(req.output_ids) == retract_after:
                    self._retract_and_readmit(req)
        return state

    def _retract_and_readmit(self, req):
        req.time_stats.set_retract_time()
        req.retraction_count += 1
        self.now += 5 * TOKEN_PERIOD_S  # sits in the waiting queue
        req.time_stats.set_wait_queue_entry_time()
        req.time_stats.set_forward_entry_time()
        self.now += TOKEN_PERIOD_S  # re-prefill (no new token is appended)
        req.time_stats.set_prefill_finished_time()

    def _stream_output(self, req, state):
        acc = self._accumulator()
        acc.accept(req=req)
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        if payload is None:
            return
        self.delivered_batches += 1
        # Cross the scheduler -> detokenizer -> tokenizer process boundaries
        # (default SGLANG_USE_PICKLE_IPC: the detokenizer unpickles and
        # re-pickles the stats), each process with its own clock offset.
        wire = pickle.dumps(unwrap_from_pickle(payload.time_stats))
        with mock.patch.object(
            rts,
            "global_diff_realtime_monotonic",
            rts.global_diff_realtime_monotonic + 7.0,
        ):
            wire = pickle.dumps(pickle.loads(wire))
        with mock.patch.object(
            rts,
            "global_diff_realtime_monotonic",
            rts.global_diff_realtime_monotonic + 11.0,
        ):
            time_stats = pickle.loads(wire)
        # The tokenizer's clock reads later than the scheduler's stamp; the
        # scheduler itself is not delayed by delivery.
        self.now += IPC_DELAY_S
        try:
            self._tokenizer_receive(req, state, time_stats, payload)
        finally:
            self.now -= IPC_DELAY_S

    def _tokenizer_receive(self, req, state, time_stats, payload):
        recv_obj = SimpleNamespace(
            completion_tokens=payload.completion_tokens,
            prompt_tokens=payload.prompt_tokens,
            cached_tokens=payload.cached_tokens,
            time_stats=time_stats,
            finished_reasons=payload.finished_reasons,
        )
        # Mirrors the tokenizer's receive loop for the metrics-relevant part.
        if state.time_stats.first_token_time == 0.0:
            state.time_stats.set_first_token_time()
        if req.finished():
            state.finished = True
            state.time_stats.set_finished_time()
            self.meta_info = state.time_stats.convert_to_output_meta_info(
                recv_obj.time_stats[0], recv_obj.completion_tokens[0]
            )
        self.tm.collect_metrics(state, recv_obj, 0)

    def observed(self) -> Optional[float]:
        (kwargs,) = self.collector.finished
        return kwargs["decode_throughput"]


class TestDecodeThroughputFirstToken(CustomTestCase):
    # tokens/s the scheduler really produced: one token every TOKEN_PERIOD_S
    TRUE_TPS = 1.0 / TOKEN_PERIOD_S

    def _assert_true_throughput(self, sim: _Sim, num_tokens: int):
        expected = (num_tokens - 1) / ((num_tokens - 1) * TOKEN_PERIOD_S)
        self.assertAlmostEqual(sim.observed(), expected, delta=1e-3)
        self.assertAlmostEqual(sim.meta_info["decode_throughput"], expected, delta=1e-3)
        self.assertAlmostEqual(expected, self.TRUE_TPS)

    def test_non_stream_below_force_interval_uses_scheduler_window(self):
        sim = _Sim(self)
        sim.run(num_tokens=10, stream=False)
        # The API server saw exactly one batch (at finish): the old API-side
        # window would be zero / undefined here.
        self.assertEqual(sim.delivered_batches, 1)
        self._assert_true_throughput(sim, 10)

    def test_non_stream_above_force_interval_uses_scheduler_window(self):
        sim = _Sim(self)
        state = sim.run(num_tokens=100, stream=False)
        self.assertEqual(sim.delivered_batches, 2)  # token 50 and finish
        # API-side window starts at token 50 -> ~2x inflation; scheduler is exact.
        api_side = state.time_stats.get_decode_throughput(100)
        self.assertGreater(api_side, 1.8 * self.TRUE_TPS)
        self._assert_true_throughput(sim, 100)

    def test_stream_interval_gt_one_uses_scheduler_window(self):
        sim = _Sim(self)
        state = sim.run(num_tokens=31, stream=True, stream_interval=3)
        # first emission at token 1 (len % 3 == 1), then 4, 7, ... and finish.
        self.assertGreater(sim.delivered_batches, 2)
        self._assert_true_throughput(sim, 31)
        # With interval > 1 the API-side first batch is token one, so the two
        # only differ by delivery jitter (IPC delay on the final batch).
        api_side = state.time_stats.get_decode_throughput(31)
        self.assertAlmostEqual(api_side, sim.observed(), delta=0.02 * self.TRUE_TPS)

    def test_stream_interval_one_matches_previous_definition(self):
        sim = _Sim(self)
        state = sim.run(num_tokens=100, stream=True, stream_interval=1)
        self.assertEqual(sim.delivered_batches, 100)
        self._assert_true_throughput(sim, 100)
        # Regression guard for the streaming series (the one the dashboard
        # reads): the scheduler-derived value equals the old API-derived
        # definition up to delivery jitter of the first/last batch.
        api_side = state.time_stats.get_decode_throughput(100)
        self.assertAlmostEqual(api_side, sim.observed(), delta=0.01 * self.TRUE_TPS)

    def test_stream_and_non_stream_agree(self):
        stream = _Sim(self)
        stream.run(num_tokens=100, stream=True, stream_interval=1)
        non_stream = _Sim(self)
        non_stream.run(num_tokens=100, stream=False)
        self.assertAlmostEqual(stream.observed(), non_stream.observed(), delta=1e-3)

    def test_single_token_completion_is_not_observed(self):
        sim = _Sim(self)
        sim.run(num_tokens=1, stream=False)
        self.assertEqual(sim.observed(), 0.0)
        self.assertNotIn("decode_throughput", sim.meta_info)

    def test_two_token_completion_is_observed(self):
        sim = _Sim(self)
        sim.run(num_tokens=2, stream=False)
        self.assertEqual(sim.delivered_batches, 1)
        self._assert_true_throughput(sim, 2)

    def test_speculative_multi_token_steps(self):
        # DFLASH/EAGLE: one decode step appends several accepted tokens; token
        # one still comes from the prefill step.
        sim = _Sim(self)
        sim.run(num_tokens=100, stream=False, tokens_per_step=4)
        steps = 1 + (99 + 3) // 4
        expected = 99 / ((steps - 1) * TOKEN_PERIOD_S)
        self.assertAlmostEqual(sim.observed(), expected, delta=1e-3)

    def test_retraction_keeps_first_token_time(self):
        sim = _Sim(self)
        sim.run(num_tokens=60, stream=False, retract_after=20)
        # 59 decode intervals plus the retraction gap (5 wait + 1 re-prefill).
        expected = 59 / (65 * TOKEN_PERIOD_S)
        self.assertAlmostEqual(sim.observed(), expected, delta=1e-3)
        self.assertLess(sim.observed(), self.TRUE_TPS)

    def test_pd_decode_node_stamps_first_token_on_prebuilt(self):
        stats = rts.SchedulerReqTimeStats()
        stats.enable_metrics = True
        stats.metrics_collector = mock.Mock()
        stats.disagg_mode = DisaggregationMode.DECODE
        with mock.patch.object(rts.time, "perf_counter", lambda: 10.0):
            stats.set_decode_prebuilt_finish_time()
        with mock.patch.object(rts.time, "perf_counter", lambda: 11.0):
            stats.set_completion_time()
        self.assertEqual(stats.first_token_time, 10.0)
        self.assertAlmostEqual(stats.get_decode_throughput(11), 10.0)

    def test_unset_stamps_are_not_shipped(self):
        stats = rts.SchedulerReqTimeStats()
        stats.enable_metrics = True
        state = stats.__getstate__()
        self.assertNotIn("first_token_time", state)
        self.assertNotIn("completion_time", state)
        with mock.patch.object(
            rts,
            "global_diff_realtime_monotonic",
            rts.global_diff_realtime_monotonic + 7.0,
        ):
            received = pickle.loads(pickle.dumps(stats))
        self.assertEqual(received.first_token_time, 0.0)
        self.assertEqual(received.completion_time, 0.0)
        self.assertEqual(received.get_decode_throughput(10), 0.0)

    def test_stamps_survive_detokenizer_relay(self):
        # With pickle IPC the detokenizer unpickles the scheduler stats and
        # pickles them again for the tokenizer; the relayed copy has no
        # collector but must still carry the stamps.
        stats = rts.SchedulerReqTimeStats()
        stats.enable_metrics = True
        stats.wait_queue_entry_time = 1001.0
        stats.forward_entry_time = 1002.0
        stats.prefill_finished_time = 1003.0
        stats.first_token_time = 1003.0
        stats.completion_time = 1004.0
        base = rts.global_diff_realtime_monotonic
        with mock.patch.object(rts, "global_diff_realtime_monotonic", base + 7.0):
            relayed = pickle.loads(pickle.dumps(stats))
            self.assertFalse(relayed.enable_metrics)
            wire = pickle.dumps(relayed)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", base + 11.0):
            received = pickle.loads(wire)
        self.assertAlmostEqual(received.get_decode_latency(), 1.0)
        self.assertAlmostEqual(received.get_decode_throughput(11), 10.0)
        self.assertAlmostEqual(
            received.forward_entry_time - received.wait_queue_entry_time, 1.0
        )
        # Metrics disabled on the scheduler: nothing is shipped at either hop.
        off = rts.SchedulerReqTimeStats()
        off.forward_entry_time = 1002.0
        self.assertEqual(off.__getstate__(), {})
        self.assertEqual(pickle.loads(pickle.dumps(off)).__getstate__(), {})

    def test_api_side_fallback_without_scheduler_stats(self):
        stats = rts.APIServerReqTimeStats()
        stats.first_token_time = 1.0
        stats.finished_time = 2.0
        self.assertAlmostEqual(stats.get_decode_throughput(11, None), 10.0)
        sched = rts.SchedulerReqTimeStats()
        sched.first_token_time = 1.0
        sched.completion_time = 1.5
        self.assertAlmostEqual(stats.get_decode_throughput(11, sched), 20.0)

    def test_incomplete_scheduler_stamps_not_observed(self):
        # Scheduler stats present but token one never stamped: the interval is
        # unknown, so do not substitute the delivery-skewed API-side stamps.
        stats = rts.APIServerReqTimeStats()
        stats.first_token_time = 1.0
        stats.finished_time = 1.000001
        sched = rts.SchedulerReqTimeStats()
        sched.set_completion_time(2.0)
        self.assertEqual(sched.first_token_time, 0.0)
        self.assertEqual(stats.get_decode_throughput(10, sched), 0.0)
        self.assertNotIn(
            "decode_throughput", stats.convert_to_output_meta_info(sched, 10)
        )

    def test_dllm_scheduler_stamps_first_token(self):
        """process_batch_result_dllm stamps token one on the first emitted block."""
        clock = [100.0]
        with mock.patch.object(
            rts.time, "perf_counter", lambda: clock[0]
        ), mock.patch.object(dllm_sched, "release_kv_cache"):
            req = _FakeReq("dllm", stream=False, stream_interval=None)
            req.full_untruncated_fill_ids = array("q", [0] * 11)
            req.extend_range = SimpleNamespace(end=7)
            req.update_finish_state = lambda new_accepted_len: None
            self_ns = SimpleNamespace(
                dllm_config=SimpleNamespace(
                    first_done_first_out_mode=False, block_size=4
                ),
                token_to_kv_pool_allocator=mock.Mock(),
                metrics_reporter=mock.Mock(num_generated_tokens=0),
                output_streamer=mock.Mock(),
                tree_cache=None,
            )
            batch = mock.Mock(reqs=[req], batch_size=lambda: 1, return_logprob=False)

            def step(ids):
                return SimpleNamespace(
                    copy_done=None,
                    next_token_ids=[mock.Mock(tolist=lambda: ids)],
                    accept_length_per_req_cpu=None,
                    dllm_algo_state=None,
                    can_run_cuda_graph=False,
                )

            dllm_sched.SchedulerDllmMixin.process_batch_result_dllm(
                self_ns, batch, step([11, 12, 13, 14])
            )
            self.assertEqual(req.time_stats.first_token_time, 100.0)
            clock[0] = 101.0
            req.extend_range.end = 11
            req._finished = True
            dllm_sched.SchedulerDllmMixin.process_batch_result_dllm(
                self_ns, batch, step([15, 16, 17, 18])
            )
            self.assertEqual(req.time_stats.first_token_time, 100.0)
            self.assertEqual(req.time_stats.completion_time, 101.0)
            self.assertAlmostEqual(
                req.time_stats.get_decode_throughput(len(req.output_ids)), 7.0
            )


class _FakeMetric:
    instances = []

    def __init__(self, name, documentation, labelnames, buckets=None):
        self.name = name
        self.labelnames = list(labelnames)
        self.observed = []
        _FakeMetric.instances.append(self)

    def labels(self, **kwargs):
        assert set(kwargs) == set(self.labelnames), (self.name, kwargs)
        self._last = kwargs
        return self

    def observe(self, v):
        self.observed.append((self._last, v))

    def inc(self, *a):
        pass


class _Collector(TokenizerMetricsCollector):
    _counter_cls = _FakeMetric
    _histogram_cls = _FakeMetric


class TestTokenizerCollectorCompat(CustomTestCase):
    """Custom-collector API compatibility and reserved-label protection."""

    def setUp(self):
        _FakeMetric.instances = []
        self.server_args = SimpleNamespace(
            prompt_tokens_buckets=None, generation_tokens_buckets=None
        )

    def test_reserved_is_streaming_label_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "is_streaming"):
            _Collector(
                server_args=self.server_args,
                labels={"model_name": "m", "is_streaming": "x"},
            )

    def test_builtin_collector_gets_decode_throughput_labels(self):
        c = _Collector(server_args=self.server_args, labels={"model_name": "m"})
        self.assertEqual(
            c.histogram_decode_throughput.labelnames, ["model_name", "is_streaming"]
        )
        c.observe_one_finished_request(
            {"model_name": "m"},
            3,
            5,
            0,
            1.0,
            False,
            is_streaming=True,
            decode_throughput=42.0,
        )
        self.assertEqual(
            c.histogram_decode_throughput.observed,
            [({"model_name": "m", "is_streaming": "true"}, 42.0)],
        )

    def test_old_collector_signature_is_tolerated(self):
        class Old(_Collector):
            def __init__(self, server_args=None, labels=None):
                super().__init__(server_args=server_args, labels=labels)
                self.seen = []

            def observe_one_finished_request(
                self,
                labels,
                prompt_tokens,
                generation_tokens,
                cached_tokens,
                e2e_latency,
                has_grammar,
                cached_tokens_details=None,
            ):
                self.seen.append((labels, generation_tokens))

        kwargs = filter_supported_kwargs(
            Old.__init__,
            dict(bucket_decode_throughput=[1.0], labels={"model_name": "m"}),
        )
        self.assertEqual(kwargs, {"labels": {"model_name": "m"}})
        c = Old(server_args=self.server_args, **kwargs)
        flt = SupportedKwargsFilter(c.observe_one_finished_request)
        extra = flt(dict(spec_verify_ct=0, is_streaming=True, decode_throughput=1.0))
        self.assertEqual(extra, {})
        c.observe_one_finished_request(
            {"model_name": "m"}, 3, 5, 0, 1.0, False, None, **extra
        )
        self.assertEqual(c.seen, [({"model_name": "m"}, 5)])

    def test_var_kwargs_collector_receives_everything(self):
        def observe(labels, *args, **kwargs):
            pass

        payload = dict(spec_verify_ct=1, is_streaming=False, decode_throughput=2.0)
        self.assertEqual(SupportedKwargsFilter(observe)(payload), payload)


if __name__ == "__main__":
    unittest.main()
