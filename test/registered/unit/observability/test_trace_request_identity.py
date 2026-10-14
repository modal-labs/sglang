"""Unit tests for request IDs on root spans, trace IDs and the tracing rank."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import pickle
import unittest
from http import HTTPStatus
from types import SimpleNamespace

from starlette.datastructures import Headers

import sglang.srt.observability.trace as mod
from sglang.srt.environ import envs
from sglang.srt.observability.trace import (
    TraceCustomIdGenerator,
    TraceReqContext,
    extract_trace_headers,
    request_header_attributes,
    set_global_trace_level,
    should_trace_rank,
    trace_set_thread_info,
)
from sglang.test.test_utils import CustomTestCase

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import StatusCode

    _has_otel = True
except ImportError:
    _has_otel = False

TRACED = "x-request-id,x-upstream-request-id"
TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class TestRequestHeaders(CustomTestCase):
    def test_extracts_trace_context_and_configured_headers(self):
        headers = Headers(
            {
                "Traceparent": TRACEPARENT,
                "X-Request-Id": "6f97ea7f",
                "Authorization": "Bearer secret",
            }
        )
        with envs.SGLANG_TRACE_REQUEST_HEADERS.override(TRACED):
            extracted = extract_trace_headers(headers)

        self.assertEqual(
            extracted, {"traceparent": TRACEPARENT, "x-request-id": "6f97ea7f"}
        )

    def test_extracts_only_trace_context_by_default(self):
        with envs.SGLANG_TRACE_REQUEST_HEADERS.override(""):
            extracted = extract_trace_headers(
                {"traceparent": TRACEPARENT, "x-request-id": "6f97ea7f"}
            )

        self.assertEqual(extracted, {"traceparent": TRACEPARENT})

    def test_attributes_follow_otel_naming_and_are_capped(self):
        """Values are cut to SGLANG_TRACE_REQUEST_HEADER_MAX_LEN, which the
        gateway also reads: a hardcoded cap would ignore the setting and let the
        engine and gateway record the same header differently. Empty values are
        left out."""
        headers = {"X-Request-Id": "a" * 1000, "x-upstream-request-id": ""}
        with (
            envs.SGLANG_TRACE_REQUEST_HEADERS.override(TRACED),
            envs.SGLANG_TRACE_REQUEST_HEADER_MAX_LEN.override(8),
        ):
            attrs = request_header_attributes(headers)

        self.assertEqual(attrs, {"http.request.header.x-request-id": "a" * 8})


class TestTraceIds(CustomTestCase):
    def test_trace_ids_use_all_128_bits(self):
        generator = TraceCustomIdGenerator()
        ids = [generator.generate_trace_id() for _ in range(64)]

        self.assertTrue(all(0 < i < 2**128 for i in ids))
        self.assertTrue(any(i >= 2**64 for i in ids))
        self.assertEqual(len(set(ids)), len(ids))


def server_args(tp_size=8, dp_size=1, enable_dp_attention=False, attn_cp_size=1):
    return SimpleNamespace(
        tp_size=tp_size,
        dp_size=dp_size,
        enable_dp_attention=enable_dp_attention,
        attn_cp_size=attn_cp_size,
    )


class TestTracingRank(CustomTestCase):
    def tracing_ranks(self, args, attn_cp_rank=0):
        return [
            tp_rank
            for tp_rank in range(args.tp_size)
            if should_trace_rank(args, tp_rank, attn_cp_rank)
        ]

    def test_only_the_first_tp_rank_traces(self):
        with envs.SGLANG_TRACE_ALL_RANKS.override(False):
            self.assertEqual(self.tracing_ranks(server_args()), [0])

    def test_each_attention_dp_group_keeps_a_tracing_rank(self):
        args = server_args(dp_size=2, enable_dp_attention=True)
        with envs.SGLANG_TRACE_ALL_RANKS.override(False):
            self.assertEqual(self.tracing_ranks(args), [0, 4])

    def test_only_the_first_context_parallel_rank_traces(self):
        args = server_args(attn_cp_size=2)
        with envs.SGLANG_TRACE_ALL_RANKS.override(False):
            self.assertEqual(self.tracing_ranks(args, attn_cp_rank=0), [0, 4])
            self.assertEqual(self.tracing_ranks(args, attn_cp_rank=1), [])

    def test_every_rank_traces_when_asked(self):
        with envs.SGLANG_TRACE_ALL_RANKS.override(True):
            self.assertEqual(self.tracing_ranks(server_args()), list(range(8)))


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestRootSpan(CustomTestCase):
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
        set_global_trace_level(2)
        trace_set_thread_info("Tokenizer")

    def tearDown(self):
        mod.opentelemetry_initialized = self.orig_initialized
        mod.tracer = self.orig_tracer
        mod.threads_info.clear()
        mod.threads_info.update(self.orig_threads)
        from sglang.srt.runtime_context import get_resources

        get_resources().trace_level = self.orig_level

    def _root_span(self, role="unified", headers=None, finish_reason=None):
        ctx = TraceReqContext("a1b2c3d4e5f6", role=role, external_trace_header=headers)
        ctx.trace_req_start()
        ctx.trace_set_root_error(finish_reason)
        ctx.trace_req_finish()
        spans = self.exporter.get_finished_spans()
        (root,) = [s for s in spans if "rid" in s.attributes]
        return root

    def test_root_span_carries_request_ids_under_a_fixed_name(self):
        for role, name in [("decode", "decode Req"), ("unified", "Req")]:
            self.exporter.clear()
            with envs.SGLANG_TRACE_REQUEST_HEADERS.override(TRACED):
                root = self._root_span(
                    role, {"x-request-id": "6f97ea7f", "traceparent": TRACEPARENT}
                )

            self.assertEqual(root.name, name)
            self.assertEqual(root.attributes["rid"], "a1b2c3d4e5f6")
            self.assertEqual(
                root.attributes["http.request.header.x-request-id"], "6f97ea7f"
            )
            self.assertEqual(
                format(root.context.trace_id, "032x"), TRACEPARENT.split("-")[1]
            )

    def test_lane_span_names_leave_out_host_and_pid(self):
        mod.threads_info.clear()
        trace_set_thread_info("Scheduler", tp_rank=0, dp_rank=1)
        self._root_span()

        names = [s.name for s in self.exporter.get_finished_spans()]
        self.assertIn("Scheduler [TP 0] [DP 1]", names)

    def test_root_span_records_the_request_id_unchanged(self):
        ctx = TraceReqContext("req_abc_123")
        ctx.trace_req_start()
        ctx.trace_req_finish()

        root = self.exporter.get_finished_spans()[-1]
        self.assertEqual(root.attributes["rid"], "req_abc_123")

    def test_health_checks_are_not_traced(self):
        self.assertTrue(TraceReqContext("a1b2c3d4e5f6").is_tracing_enabled())
        self.assertFalse(TraceReqContext("HEALTH_CHECK_1f2e").is_tracing_enabled())

    def test_server_side_abort_marks_the_root_as_an_error(self):
        root = self._root_span(
            finish_reason={
                "type": "abort",
                "message": "timeout",
                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
            }
        )

        self.assertEqual(root.attributes["error.type"], "503")
        self.assertEqual(root.status.status_code, StatusCode.ERROR)

    def test_caller_errors_and_plain_aborts_are_recorded_but_not_errors(self):
        for finish_reason, error_type in [
            ({"type": "abort", "status_code": HTTPStatus.BAD_REQUEST}, "400"),
            ({"type": "abort", "status_code": None}, "abort"),
            ({"type": "abort", "status_code": "unavailable"}, "abort"),
            ({"type": "stop"}, None),
        ]:
            self.exporter.clear()
            root = self._root_span(finish_reason=finish_reason)

            self.assertEqual(root.attributes.get("error.type"), error_type)
            self.assertEqual(root.status.status_code, StatusCode.UNSET)

    def test_lane_spans_follow_the_root_error_rule(self):
        """An aborted lane is an error only for a server fault (5xx), as on the
        root span. A rejected request (4xx) or a client/user abort (no status
        code) must not show up as a failed span under a root that does not."""
        for abort_info, status in [
            (
                {"reason": "too long", "status_code": HTTPStatus.BAD_REQUEST},
                StatusCode.UNSET,
            ),
            ({"reason": "Aborted"}, StatusCode.UNSET),
            (
                {
                    "reason": "KV transfer failed",
                    "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                },
                StatusCode.ERROR,
            ),
        ]:
            self.exporter.clear()
            ctx = TraceReqContext("a1b2c3d4e5f6")
            ctx.trace_req_start()
            ctx.abort(abort_info=abort_info)
            ctx.trace_req_finish()

            (lane,) = [
                s for s in self.exporter.get_finished_spans() if s.name == "Tokenizer"
            ]
            self.assertEqual(lane.status.status_code, status)
            self.assertEqual(lane.attributes["reason"], abort_info["reason"])

    def _discard(self, rids, error, **kwargs):
        """Discard pending requests through the tokenizer's failure path and
        return their exported root spans, keyed by rid."""
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

        states = {}
        for rid in rids:
            stats = APIServerReqTimeStats()
            stats.init_trace_ctx(rid=rid, bootstrap_room=None)
            stats.set_created_time()
            states[rid] = SimpleNamespace(time_stats=stats)
        manager = SimpleNamespace(rid_to_state=states)
        if len(rids) == 1:
            obj = SimpleNamespace(is_single=True, rid=rids[0])
        else:
            obj = SimpleNamespace(is_single=False, rid=list(rids))

        TokenizerManager._discard_pending_req_states(manager, obj, error, **kwargs)

        self.assertEqual(manager.rid_to_state, {})
        return {
            s.attributes["rid"]: s
            for s in self.exporter.get_finished_spans()
            if "rid" in s.attributes
        }

    def test_server_faults_before_scheduling_are_errors_with_their_status(self):
        """The status recorded must be the one the HTTP layer returns: an
        exception's own status_code (e.g. HTTPException 503) wins, and any other
        unexpected exception is a 500. Both are server faults, so the root is
        marked as an error; a sampler keeping error traces depends on that."""
        import fastapi

        for error, error_type in [
            (fastapi.HTTPException(status_code=503, detail="no encoder"), "503"),
            (RuntimeError("unexpected"), "500"),
        ]:
            self.exporter.clear()
            roots = self._discard(["a1b2c3d4e5f6"], error)

            root = roots["a1b2c3d4e5f6"]
            self.assertEqual(root.attributes["error.type"], error_type)
            self.assertEqual(root.status.status_code, StatusCode.ERROR)

    def test_discarding_a_batch_ends_every_sub_request_trace(self):
        """A batch rejected before scheduling has one root span per sub-request;
        each must be ended and exported, not only the first."""
        rids = ["batch-0", "batch-1", "batch-2"]

        roots = self._discard(rids, ValueError("one prompt too long"))

        self.assertEqual(sorted(roots), rids)
        for rid in rids:
            self.assertEqual(roots[rid].attributes["error.type"], "400")

    def test_client_disconnect_is_not_recorded_as_a_400(self):
        """A client disconnect raises ValueError, like a rejected input, but the
        client gets no response: recording it as a 400 would count every
        disconnected stream as a bad request."""
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

        stats = APIServerReqTimeStats()
        stats.init_trace_ctx(rid="a1b2c3d4e5f6", bootstrap_room=None)
        stats.set_created_time()
        manager = SimpleNamespace(
            rid_to_state={"a1b2c3d4e5f6": SimpleNamespace(time_stats=stats)}
        )
        obj = SimpleNamespace(is_single=True, rid="a1b2c3d4e5f6")

        TokenizerManager._discard_pending_req_states(
            manager,
            obj,
            ValueError("Request is disconnected from the client side"),
            client_disconnected=True,
        )

        (root,) = [
            s for s in self.exporter.get_finished_spans() if "rid" in s.attributes
        ]
        self.assertEqual(root.attributes["error.type"], "abort")
        self.assertEqual(root.status.status_code, StatusCode.UNSET)

    def test_generate_request_checks_for_a_disconnect_before_labeling(self):
        """generate_request must ask the HTTP request whether the client left;
        only then is a ValueError recorded as an abort instead of a 400."""
        import asyncio
        import contextlib
        from unittest.mock import AsyncMock, Mock

        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

        for disconnected, error_type in [(True, "abort"), (False, "400")]:
            self.exporter.clear()
            stats = APIServerReqTimeStats()
            stats.init_trace_ctx(rid="a1b2c3d4e5f6", bootstrap_room=None)
            stats.set_created_time()

            manager = TokenizerManager.__new__(TokenizerManager)
            manager.rid_to_state = {}
            manager._init_req_state = lambda obj, request: manager.rid_to_state.update(
                {obj.rid: SimpleNamespace(time_stats=stats)}
            )
            manager.auto_create_handle_loop = Mock()
            manager._set_default_priority = Mock()
            manager.server_args = SimpleNamespace(language_only=False)
            manager.request_logger = Mock()
            manager.tokenizer = None
            manager.is_pause = False
            manager.is_pause_cond = asyncio.Condition()
            manager.model_update_lock = SimpleNamespace(
                reader_lock=contextlib.nullcontext()
            )
            manager._validate_and_resolve_lora = AsyncMock()
            manager._tokenize_one_request = AsyncMock(side_effect=ValueError("failed"))
            obj = SimpleNamespace(
                rid="a1b2c3d4e5f6",
                is_single=True,
                routed_dp_rank=None,
                normalize_batch_and_arguments=Mock(),
            )
            request = Mock(is_disconnected=AsyncMock(return_value=disconnected))

            async def drive():
                await manager.generate_request(obj, request).__anext__()

            with self.assertRaises(ValueError):
                asyncio.run(drive())
            (root,) = [
                s for s in self.exporter.get_finished_spans() if "rid" in s.attributes
            ]
            self.assertEqual(root.attributes["error.type"], error_type)

    def test_context_sent_to_a_rank_that_does_not_trace_is_disabled(self):
        ctx = TraceReqContext("a1b2c3d4e5f6")
        ctx.trace_req_start()
        state = pickle.dumps(ctx)

        mod.opentelemetry_initialized = False
        received = pickle.loads(state)
        received.trace_slice_start("prefill", level=1)
        received.trace_slice_end("prefill", level=1)

        self.assertFalse(received.is_tracing_enabled())
        mod.opentelemetry_initialized = True
        ctx.trace_req_finish()
        self.assertEqual(
            [s.name for s in self.exporter.get_finished_spans() if s.name == "prefill"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
