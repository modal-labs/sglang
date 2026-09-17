"""CPU tests for Fp8SatfiniteTelemetry per-layer clamp counters."""

import unittest

import torch

from sglang.kernels.ops.quantization.fp8_utils import to_fp8_satfinite
from sglang.srt.layers.attention.fp8_satfinite_telemetry import (
    Fp8SatfiniteTelemetry,
)
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _telemetry(num_layers=4, every=1):
    return Fp8SatfiniteTelemetry(num_layers=num_layers, device="cpu", every=every)


class TestFp8SatfiniteTelemetry(CustomTestCase):
    def test_rows_counted_by_kind(self):
        t = _telemetry()
        x = torch.zeros(6, 2, 4)
        x[0, 0, 0] = 100.0  # below all thresholds
        x[1, 0, 0] = 450.0  # gt448 only
        x[2, 0, 0] = 470.0  # gt448 + gt464
        x[3, 0, 0] = float("inf")
        x[4, 0, 0] = float("nan")
        x[5, 0, 0] = -460.0  # sign-insensitive, gt448 only

        self.assertTrue(t.should_record(2))
        t.record(2, x)

        self.assertEqual(
            t.drain(),
            [
                (2, "gt448", 3, 470.0),
                (2, "gt464", 1, 470.0),
                (2, "nonfinite", 2, 470.0),
            ],
        )

    def test_drain_zeroes_buffers_and_skips_sync_when_clean(self):
        t = _telemetry()
        t.record(0, torch.full((2, 4), 500.0))
        self.assertEqual(len(t.drain()), 2)  # gt448 + gt464 rows
        self.assertFalse(t._dirty)
        self.assertEqual(t.drain(), [])
        self.assertTrue((t.counts == 0).all())
        self.assertTrue((t.amax == 0).all())

    def test_sampling_every(self):
        t = _telemetry(every=4)
        results = [t.should_record(0) for _ in range(8)]
        self.assertEqual(results, [False, False, False, True] * 2)

        t_off = _telemetry(every=0)
        self.assertFalse(any(t_off.should_record(0) for _ in range(8)))

    def test_other_layers_do_not_advance_chunk_counter(self):
        t = _telemetry(every=2)
        self.assertFalse(t.should_record(0))  # chunk 1
        for _ in range(5):
            self.assertFalse(t.should_record(1))  # does not advance
        self.assertTrue(t.should_record(0))  # chunk 2

    def test_recorded_tensor_satfinite_quantizes_without_nan(self):
        t = _telemetry()
        x = torch.zeros(3, 2, 4)
        x[0, 0, 0] = 1e4
        x[1, 0, 0] = 450.0
        self.assertTrue(t.should_record(0))
        t.record(0, x)
        out = to_fp8_satfinite(x, torch.float8_e4m3fn)
        self.assertFalse(out.to(torch.float32).isnan().any().item())

    def test_metrics_collector_increment(self):
        class _StubCounter:
            def __init__(self):
                self.calls = []

            def labels(self, **kwargs):
                self.calls.append(kwargs)
                return self

            def inc(self, n):
                self.calls.append(("inc", n))

        # prometheus_client isn't installed in the CPU unit-test env; bind
        # the method onto a bare instance with a stub counter.
        collector = object.__new__(SchedulerMetricsCollector)
        collector.labels = {"dp": "0"}
        collector.fp8_satfinite_clamp_events_total = _StubCounter()
        collector.increment_fp8_satfinite_clamp_events(3, "gt448", 7)
        self.assertEqual(
            collector.fp8_satfinite_clamp_events_total.calls,
            [{"dp": "0", "layer": "3", "kind": "gt448"}, ("inc", 7)],
        )

    def test_wrapper_backends_drain_all_children(self):
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        class _Child:
            def __init__(self, events):
                self.events = events

            def drain_fp8_satfinite_telemetry(self):
                ev, self.events = self.events, []
                return ev

        primary, c0, c1 = (
            _Child([(1, "gt448", 1, 450.0)]),
            _Child([(1, "gt448", 2, 460.0)]),
            _Child([]),
        )
        tbo = object.__new__(TboAttnBackend)
        tbo.primary, tbo.children = primary, [c0, c1]
        self.assertEqual(
            tbo.drain_fp8_satfinite_telemetry(),
            [(1, "gt448", 1, 450.0), (1, "gt448", 2, 460.0)],
        )
        self.assertEqual(tbo.drain_fp8_satfinite_telemetry(), [])

        hybrid = object.__new__(HybridAttnBackend)
        hybrid.prefill_backend = _Child([(2, "gt464", 3, 470.0)])
        hybrid.decode_backend = _Child([(2, "nonfinite", 1, 0.0)])
        self.assertEqual(
            hybrid.drain_fp8_satfinite_telemetry(),
            [(2, "gt464", 3, 470.0), (2, "nonfinite", 1, 0.0)],
        )
        shared = _Child([(0, "gt448", 1, 449.0)])
        hybrid.prefill_backend = hybrid.decode_backend = shared
        self.assertEqual(
            hybrid.drain_fp8_satfinite_telemetry(), [(0, "gt448", 1, 449.0)]
        )

    def test_reporter_warn_aggregates_kinds_and_wrapper_duplicates(self):
        from sglang.srt.managers.scheduler_components import metrics_reporter as mr

        class _Backend:
            def drain_fp8_satfinite_telemetry(self):
                return [
                    (4, "gt448", 3, 500.0),
                    (4, "gt464", 1, 500.0),
                    (4, "gt448", 7, 700.0),
                    (5, "nonfinite", 2, 0.0),
                ]

        class _NS:
            pass

        reporter = object.__new__(mr.SchedulerMetricsReporter)
        reporter.scheduler = _NS()
        reporter.scheduler.tp_worker = _NS()
        reporter.scheduler.tp_worker.model_runner = _NS()
        reporter.scheduler.tp_worker.model_runner.attn_backend = _Backend()
        reporter.enable_metrics = True
        reporter.is_stats_logging_rank = True
        reporter._fp8_satfinite_last_warn = {}
        increments = []
        reporter.metrics_collector = _NS()
        reporter.metrics_collector.increment_fp8_satfinite_clamp_events = (
            lambda layer, kind, n: increments.append((layer, kind, n))
        )
        with self.assertLogs(mr.logger, level="WARNING") as cm:
            reporter._log_fp8_satfinite_telemetry()
        self.assertEqual(len(increments), 4)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("layer=4 rows=gt448=10 gt464=1 amax=700.0", cm.output[0])
        self.assertIn("layer=5 rows=nonfinite=2 amax=0.0", cm.output[1])
        # Rate-limited: a second drain within 60s emits no WARN.
        with self.assertNoLogs(mr.logger, level="WARNING"):
            reporter._log_fp8_satfinite_telemetry()
        self.assertEqual(len(increments), 8)


if __name__ == "__main__":
    unittest.main()
