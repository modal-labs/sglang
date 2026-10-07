"""Unit tests for the detokenize trace attributes — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

import sglang.srt.observability.trace as mod
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.observability.trace import (
    set_global_trace_level,
    trace_set_thread_info,
)
from sglang.test.test_utils import CustomTestCase

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _has_otel = True
except ImportError:
    _has_otel = False


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestDetokenizeTrace(CustomTestCase):
    def setUp(self):
        self.orig_initialized = mod.opentelemetry_initialized
        self.orig_tracer = mod.tracer
        self.orig_threads = mod.threads_info.copy()
        from sglang.srt.runtime_context import get_resources

        self.orig_level = get_resources().trace_level

        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        mod.opentelemetry_initialized = True
        mod.tracer = provider.get_tracer("test")
        set_global_trace_level(3)
        trace_set_thread_info("Tokenizer")

    def tearDown(self):
        mod.opentelemetry_initialized = self.orig_initialized
        mod.tracer = self.orig_tracer
        mod.threads_info.clear()
        mod.threads_info.update(self.orig_threads)
        from sglang.srt.runtime_context import get_resources

        get_resources().trace_level = self.orig_level

    def dispatched_request(self):
        """A request as the tokenizer holds it after dispatch to the scheduler."""
        stats = APIServerReqTimeStats()
        stats.init_trace_ctx(rid="req-1", bootstrap_room=None)
        stats.set_created_time()
        stats.set_tokenize_finish_time()
        stats.set_api_server_dispatch_time()
        stats.set_api_server_dispatch_finish_time()  # ends the tokenizer lane
        return stats

    def root_span(self):
        return next(s for s in self.exporter.get_finished_spans() if s.name == "Req")

    def test_detokenize_batches_sum_onto_the_root_span(self):
        stats = self.dispatched_request()
        stats.add_detokenize_duration(0.001)
        stats.add_detokenize_duration(0.002)
        stats.set_finished_time()

        spans = self.exporter.get_finished_spans()
        self.assertFalse([s for s in spans if s.name == "detokenize"])
        attrs = self.root_span().attributes
        self.assertAlmostEqual(attrs["detokenize.total_ms"], 3.0, delta=1e-6)
        self.assertEqual(attrs["detokenize.batches"], 2)

        # A request no detokenizer batch carried gets no detokenize attributes.
        self.exporter.clear()
        stats = self.dispatched_request()
        stats.set_finished_time()
        attrs = self.root_span().attributes
        self.assertNotIn("detokenize.total_ms", attrs)
        self.assertNotIn("detokenize.batches", attrs)


if __name__ == "__main__":
    unittest.main()
