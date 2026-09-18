"""Pure-CPU unit tests for the deployment-load gauges on the scheduler metrics path.

``sglang:prefill_queue_depth`` / ``sglang:decode_queue_depth`` split the single
non-PD ``waiting_queue`` by ``Req.is_retracted`` (computed inside the existing
``QueueCount.from_reqs`` pass), ``sglang:num_prefill_inflight_reqs`` reflects
``scheduler.chunked_req``, and ``emit_constants`` publishes the static
``sglang:max_running_requests`` / ``sglang:max_queued_requests`` gauges plus the
``sglang:replica_gpu_info{gpu_type,gpu_count}`` info gauge.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
    SchedulerStats,
)
from sglang.test.test_utils import CustomTestCase


class _FakeReq:
    def __init__(self, is_retracted: bool = False, priority=None):
        self.is_retracted = is_retracted
        self.priority = priority


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def set(self, value):
        self.metric.values[tuple(sorted(self.labels.items()))] = value

    def inc(self, value=1):
        pass

    def observe(self, value):
        pass


class _RecordingMetric:
    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.values = {}

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


def _make_reporter(scheduler) -> SchedulerMetricsReporter:
    context = SchedulerMetricsCollectorContext(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    with patch.object(SchedulerMetricsReporter, "__init__", return_value=None):
        reporter = SchedulerMetricsReporter()
    reporter.scheduler = scheduler
    reporter.metrics_collector_context = context
    reporter.metrics_collector = None
    reporter.stats = SchedulerStats()
    return reporter


class TestQueueCountRetracted(CustomTestCase):
    def test_counts_retracted_in_same_pass(self):
        reqs = [_FakeReq(), _FakeReq(is_retracted=True), _FakeReq()]
        qc = QueueCount.from_reqs(reqs, count_retracted=True)
        self.assertEqual(qc.total, 3)
        self.assertEqual(qc.num_retracted, 1)
        self.assertIsNone(qc.by_priority)

    def test_priority_breakdown_and_retracted_together(self):
        reqs = [
            _FakeReq(priority=0),
            _FakeReq(is_retracted=True, priority=1),
            _FakeReq(priority=1),
        ]
        qc = QueueCount.from_reqs(
            reqs, enable_priority_scheduling=True, count_retracted=True
        )
        self.assertEqual(qc.by_priority, {0: 1, 1: 2})
        self.assertEqual(qc.num_retracted, 1)

    def test_default_does_not_count_retracted(self):
        qc = QueueCount.from_reqs([_FakeReq(is_retracted=True)])
        self.assertEqual((qc.total, qc.num_retracted), (1, 0))

    def test_empty_queue(self):
        qc = QueueCount.from_reqs([], count_retracted=True)
        self.assertEqual((qc.total, qc.num_retracted), (0, 0))


class TestQueueDepths(CustomTestCase):
    def _scheduler(self, waiting, chunked_req=None, mode=DisaggregationMode.NULL):
        return types.SimpleNamespace(
            waiting_queue=waiting,
            chunked_req=chunked_req,
            disaggregation_mode=mode,
        )

    def test_non_pd_split_by_retraction(self):
        waiting = [_FakeReq(), _FakeReq(is_retracted=True), _FakeReq(), _FakeReq()]
        reporter = _make_reporter(self._scheduler(waiting))
        reporter.stats.num_queue_reqs = QueueCount.from_reqs(
            waiting, count_retracted=True
        )
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 3)
        self.assertEqual(reporter.stats.decode_queue_depth, 1)
        self.assertEqual(
            reporter.stats.prefill_queue_depth + reporter.stats.decode_queue_depth,
            reporter.stats.num_queue_reqs.total,
        )
        self.assertEqual(reporter.stats.num_prefill_inflight_reqs, 0)

    def test_chunked_prefill_inflight(self):
        reporter = _make_reporter(self._scheduler([], chunked_req=_FakeReq()))
        reporter.stats.num_queue_reqs = QueueCount.from_reqs([], count_retracted=True)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 0)
        self.assertEqual(reporter.stats.decode_queue_depth, 0)
        self.assertEqual(reporter.stats.num_prefill_inflight_reqs, 1)

    def test_pd_prefill_engine(self):
        scheduler = self._scheduler([_FakeReq()], mode=DisaggregationMode.PREFILL)
        scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(queue=[1, 2])
        scheduler.disagg_prefill_inflight_queue = [1]
        reporter = _make_reporter(scheduler)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 4)
        self.assertEqual(reporter.stats.decode_queue_depth, 0)

    def test_pd_decode_engine(self):
        scheduler = self._scheduler([], mode=DisaggregationMode.DECODE)
        scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[1], retracted_queue=[1, 2, 3, 4], held_rebootstrap_reqs=[1]
        )
        scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(queue=[1, 2])
        reporter = _make_reporter(scheduler)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 0)
        self.assertEqual(reporter.stats.decode_queue_depth, 8)


class TestReplicaGpuCount(CustomTestCase):
    def _count(self, **kwargs):
        defaults = dict(tp_size=1, pp_size=1, dp_size=1, enable_dp_attention=False)
        defaults.update(kwargs)
        fake = types.SimpleNamespace(server_args=types.SimpleNamespace(**defaults))
        return Scheduler._replica_gpu_count(fake)

    def test_tp8(self):
        self.assertEqual(self._count(tp_size=8), 8)

    def test_tp_pp(self):
        self.assertEqual(self._count(tp_size=4, pp_size=2), 8)

    def test_plain_dp_multiplies(self):
        self.assertEqual(self._count(tp_size=2, dp_size=4), 8)

    def test_dp_attention_is_subset_of_tp(self):
        self.assertEqual(self._count(tp_size=8, dp_size=8, enable_dp_attention=True), 8)


class _RecordingCollector(SchedulerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric
    _summary_cls = _RecordingMetric


class TestCollectorGauges(CustomTestCase):
    LABELS = {
        "model_name": "m",
        "engine_type": "unified",
        "tp_rank": 0,
        "pp_rank": 0,
        "moe_ep_rank": 0,
    }

    # Built once: the collector also registers non-DI'd metrics (GaugeHistogram)
    # on the global prometheus registry, which rejects duplicates.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        server_args = types.SimpleNamespace(
            prefill_delayer_max_delay_passes=30,
            prefill_delayer_forward_passes_buckets=None,
            prefill_delayer_wait_seconds_buckets=None,
        )
        cls.collector = _RecordingCollector(
            labels=dict(cls.LABELS), server_args=server_args
        )

    def _collector(self) -> _RecordingCollector:
        for gauge in (
            self.collector.prefill_queue_depth,
            self.collector.decode_queue_depth,
            self.collector.num_prefill_inflight_reqs,
            self.collector.max_running_requests,
            self.collector.max_queued_requests,
            self.collector.replica_gpu_info,
        ):
            gauge.values.clear()
        return self.collector

    def _value(self, gauge, **extra):
        return gauge.values[tuple(sorted({**self.LABELS, **extra}.items()))]

    def test_gauge_names(self):
        c = self._collector()
        self.assertEqual(c.prefill_queue_depth.name, "sglang:prefill_queue_depth")
        self.assertEqual(c.decode_queue_depth.name, "sglang:decode_queue_depth")
        self.assertEqual(
            c.num_prefill_inflight_reqs.name, "sglang:num_prefill_inflight_reqs"
        )
        self.assertEqual(c.max_running_requests.name, "sglang:max_running_requests")
        self.assertEqual(c.max_queued_requests.name, "sglang:max_queued_requests")
        self.assertEqual(c.replica_gpu_info.name, "sglang:replica_gpu_info")
        self.assertEqual(
            c.replica_gpu_info.labelnames,
            (*self.LABELS.keys(), "gpu_type", "gpu_count"),
        )

    def test_log_stats_emits_queue_depths(self):
        c = self._collector()
        stats = SchedulerStats(
            prefill_queue_depth=3, decode_queue_depth=1, num_prefill_inflight_reqs=1
        )
        c.log_stats(stats)
        self.assertEqual(self._value(c.prefill_queue_depth), 3)
        self.assertEqual(self._value(c.decode_queue_depth), 1)
        self.assertEqual(self._value(c.num_prefill_inflight_reqs), 1)

    def _emit(self, c, **kwargs):
        c.emit_constants(
            max_total_num_tokens=1000,
            max_running_requests_under_SLO=None,
            engine_startup_time=0.0,
            engine_load_weights_time=0.0,
            page_size=1,
            num_pages=1000,
            context_len=4096,
            startup_available_gpu_memory_gb=1.0,
            **kwargs,
        )

    def test_emit_constants_static_and_gpu_info(self):
        c = self._collector()
        self._emit(
            c,
            max_running_requests=12,
            max_queued_requests=16,
            gpu_type="NVIDIA B300",
            gpu_count=8,
        )
        self.assertEqual(self._value(c.max_running_requests), 12)
        self.assertEqual(self._value(c.max_queued_requests), 16)
        self.assertEqual(
            self._value(c.replica_gpu_info, gpu_type="NVIDIA B300", gpu_count="8"), 1
        )

    def test_emit_constants_skips_unknown_values(self):
        c = self._collector()
        self._emit(c)
        self.assertEqual(c.max_running_requests.values, {})
        self.assertEqual(c.max_queued_requests.values, {})
        self.assertEqual(c.replica_gpu_info.values, {})


if __name__ == "__main__":
    unittest.main()
