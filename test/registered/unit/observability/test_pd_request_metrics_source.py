"""CPU regression tests executing real metrics methods with Prometheus collectors."""

import math
from types import MethodType
from types import SimpleNamespace as NS

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram
from test_first_output_flush_source import SRT, load_node

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

COLLECT = load_node(
    SRT / "managers/tokenizer_manager.py",
    "TokenizerManager",
    "collect_metrics",
    DisaggregationMode=NS(PREFILL="prefill"),
)
METRICS = SRT / "observability/metrics_collector.py"
LABELS = load_node(
    SRT / "managers/tokenizer_manager.py", "TokenizerManager", "_request_metric_labels"
)
ABORT = load_node(
    SRT / "managers/tokenizer_manager.py",
    "TokenizerManager",
    "_handle_abort_req",
    is_health_check_generate_req=lambda recv_obj: False,
    logger=NS(info=lambda *args, **kwargs: None),
)


def collector(role):
    registry = CollectorRegistry()
    obj = NS(
        RESERVED_LABELS=(),
        _counter_cls=lambda **kw: Counter(registry=registry, **kw),
        _histogram_cls=lambda **kw: Histogram(registry=registry, **kw),
    )
    init = load_node(
        METRICS,
        "TokenizerMetricsCollector",
        "__init__",
        generate_buckets=lambda supplied, default: supplied or default,
        # Label validation is covered elsewhere; these tests exercise observation.
        check_reserved_metric_labels=lambda *args, **kwargs: None,
    )
    init(
        obj,
        server_args=NS(prompt_tokens_buckets=None, generation_tokens_buckets=None),
        labels={"engine_type": role},
    )
    for method in (
        "observe_time_to_first_token",
        "observe_inter_token_latency",
        "observe_request_tpot",
        "observe_finished_outcome",
        "observe_one_finished_request",
    ):
        setattr(
            obj,
            method,
            MethodType(load_node(METRICS, "TokenizerMetricsCollector", method), obj),
        )
    return registry, obj


def request(role, *, stream=False, tokens=101, reason="length", finished=True):
    registry, metrics = collector(role)
    state = NS(
        obj=NS(stream=stream),
        ttft_observed=False,
        last_completion_tokens=1,
        finished=finished,
        time_stats=NS(
            get_first_token_latency=lambda: 0.1,
            get_interval=lambda: 0.1,
            get_e2e_latency=lambda: 1.0,
            get_decode_throughput=lambda *args: 0.0,
            set_last_time=lambda: None,
        ),
    )
    manager = NS(
        metrics_collector=metrics,
        disaggregation_mode=role,
        enable_priority_scheduling=False,
        _request_has_grammar=lambda obj: False,
        _finished_request_kwargs_filter=lambda kwargs: kwargs,
    )
    manager._request_metric_labels = MethodType(LABELS, manager)
    recv = NS(
        completion_tokens=[tokens],
        finished_reasons=[{"type": reason}],
        prompt_tokens=[100],
        cached_tokens=[60],
        time_stats=None,
    )
    return registry, manager, state, recv


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "throughput", [200.0, 0.0, -1.0, float("inf"), float("nan"), None]
)
@pytest.mark.parametrize(
    "role,reason,tokens",
    [
        ("null", "length", 101),
        ("decode", "stop", 101),
        ("prefill", "length", 101),
        ("decode", "abort", 101),
        ("decode", "other", 101),
        ("null", "length", 1),
    ],
)
def test_request_tpot_excludes_invalid_or_incomplete_decode(
    stream, throughput, role, reason, tokens
):
    registry, manager, state, recv = request(
        role, stream=stream, tokens=tokens, reason=reason, finished=False
    )
    meta = {"decode_throughput": throughput}
    COLLECT(manager, state, recv, 0, meta)
    labels = {"engine_type": role, "stream": str(stream).lower()}
    metric = "sglang:request_time_per_output_token_seconds"
    assert registry.get_sample_value(metric + "_count", labels) is None
    state.finished = True
    COLLECT(manager, state, recv, 0, meta)
    expected = (
        role != "prefill"
        and reason in ("stop", "length")
        and tokens > 1
        and throughput is not None
        and math.isfinite(throughput)
        and throughput > 0
    )
    assert registry.get_sample_value(metric + "_count", labels) == (
        1 if expected else None
    )
    if expected:
        assert registry.get_sample_value(metric + "_sum", labels) == pytest.approx(
            0.005
        )


@pytest.mark.parametrize("role", ["null", "decode", "prefill"])
def test_abort_without_output_does_not_observe_first_token(role):
    registry, manager, state, recv = request(role, tokens=0, reason="abort")
    COLLECT(manager, state, recv, 0, {})
    assert not state.ttft_observed
    assert (
        registry.get_sample_value(
            "sglang:time_to_first_token_seconds_count",
            {"engine_type": role, "stream": "false"},
        )
        is None
    )
    assert (
        registry.get_sample_value(
            "sglang:finished_requests_by_outcome_total",
            {"engine_type": role, "outcome": "abort"},
        )
        == 1
    )


def test_prefill_never_observes_decode_intervals():
    registry, manager, state, recv = request("prefill", tokens=0, finished=False)
    for count in (0, 1, 8, 0):
        recv.completion_tokens = [count]
        COLLECT(manager, state, recv, 0, {})
    assert (
        registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"engine_type": "prefill"}
        )
        is None
    )


def test_token_count_reset_changes_baseline_without_negative_histograms():
    registry, manager, state, recv = request("decode", tokens=8, finished=False)
    COLLECT(manager, state, recv, 0, {})
    for count in (0, 0, 2):
        recv.completion_tokens = [count]
        COLLECT(manager, state, recv, 0, {})
    labels = {"engine_type": "decode"}
    assert state.last_completion_tokens == 2
    assert (
        registry.get_sample_value("sglang:inter_token_latency_seconds_count", labels)
        == 2
    )
    assert registry.get_sample_value(
        "sglang:inter_token_latency_seconds_sum", labels
    ) == pytest.approx(0.1)


@pytest.mark.parametrize("delta", [-10, -1, 0])
def test_collector_rejects_nonpositive_token_weights(delta):
    registry, metrics = collector("decode")
    metrics.observe_inter_token_latency({"engine_type": "decode"}, 0.1, delta)
    assert (
        registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"engine_type": "decode"}
        )
        is None
    )


@pytest.mark.parametrize(
    "reason,outcome",
    [
        ("stop", "success"),
        ("length", "success"),
        ("abort", "abort"),
        ("other", "other"),
    ],
)
def test_terminal_outcomes_and_tokens_are_separated(reason, outcome):
    registry, manager, state, recv = request("decode", reason=reason)
    COLLECT(manager, state, recv, 0, {})
    labels = {"engine_type": "decode", "outcome": outcome}
    for metric, expected in [
        ("finished_requests_by_outcome", 1),
        ("finished_prompt_tokens_by_outcome", 100),
        ("finished_cached_tokens_by_outcome", 60),
    ]:
        assert (
            registry.get_sample_value("sglang:" + metric + "_total", labels) == expected
        )


def test_abort_echo_counts_an_abort_outcome_without_first_token():
    """A request aborted in the waiting queue (or by the API) before any output
    batch finishes through _handle_abort_req, not collect_metrics."""
    registry, manager, state, _ = request("null", stream=False)
    events = []
    state.finished = False
    state.obj = NS(stream=False, log_metrics=True, return_logprob=False)
    state.time_stats.set_finished_time = lambda: None
    state.output_ids = []
    state.get_text = lambda: ""
    state.out_list = []
    state.event = NS(set=lambda: events.append("set"))
    manager.enable_metrics = True
    manager.server_args = NS(weight_version="default")
    manager.rid_to_state = {"r": state}
    recv = NS(rid="r", abort_message=None, finished_reason=None)

    ABORT(manager, recv)
    labels = {"engine_type": "null", "outcome": "abort"}
    assert (
        registry.get_sample_value("sglang:finished_requests_by_outcome_total", labels)
        == 1
    )
    assert (
        registry.get_sample_value(
            "sglang:time_to_first_token_seconds_count",
            {"engine_type": "null", "stream": "false"},
        )
        is None
    )
    assert events == ["set"] and "r" not in manager.rid_to_state

    # A second echo for the same rid (already finished) records nothing.
    ABORT(manager, recv)
    assert (
        registry.get_sample_value("sglang:finished_requests_by_outcome_total", labels)
        == 1
    )


def test_embedding_output_without_completion_tokens_is_collected():
    """BatchEmbeddingOutput has no completion_tokens; collect_metrics reads it
    through getattr and every other field it touches exists on that type."""
    registry, manager, state, _ = request("null")
    recv = NS(
        finished_reasons=[{"type": "stop"}],
        prompt_tokens=[20],
        cached_tokens=[0],
        time_stats=None,
    )
    COLLECT(manager, state, recv, 0, {})
    assert (
        registry.get_sample_value(
            "sglang:finished_requests_by_outcome_total",
            {"engine_type": "null", "outcome": "success"},
        )
        == 1
    )
