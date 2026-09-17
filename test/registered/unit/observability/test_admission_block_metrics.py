from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.admission_block import (
    AdmissionBlockCause,
    req_slot_block_cause,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingMetric:
    def __init__(self):
        self.calls = []

    def labels(self, **labels):
        self._labels = labels
        return self

    def inc(self, value):
        self.calls.append((self._labels, value))

    def set(self, value):
        self.calls.append((self._labels, value))


def _adder(
    *,
    has_capacity=True,
    remaining_total=100,
    remaining_current=100,
    is_hybrid_swa=False,
    rem_mamba_slots=None,
    rem_input_tokens=100,
    rem_chunk_tokens=None,
):
    """A PrefillAdder with only the state budget_state() and add_one_req()'s
    first gate read, bypassing __init__ (which needs a real allocator)."""
    adder = object.__new__(PrefillAdder)
    adder.memory_budget = SimpleNamespace(
        has_capacity=lambda: has_capacity,
        remaining_total=remaining_total,
        remaining_current=remaining_current,
    )
    adder.is_hybrid_swa = is_hybrid_swa
    adder.rem_mamba_slots = rem_mamba_slots
    adder.rem_input_tokens = rem_input_tokens
    adder.rem_chunk_tokens = rem_chunk_tokens
    adder.dllm_config = None
    adder.can_run_list = []
    adder.prefill_max_requests = None
    adder.stop_cause = "stale"
    return adder


def test_every_cause_is_a_distinct_label_value():
    assert len(set(AdmissionBlockCause.ALL)) == len(AdmissionBlockCause.ALL)
    assert AdmissionBlockCause.MAMBA_SLOTS in AdmissionBlockCause.ALL
    assert AdmissionBlockCause.BATCH_FULL in AdmissionBlockCause.ALL


def test_req_slot_cause_blames_the_smaller_operand_and_ties_go_to_the_pool():
    assert (
        req_slot_block_cause(pp_budget=4, available_req_slots=0)
        == AdmissionBlockCause.MAX_RUNNING_REQUESTS
    )
    assert (
        req_slot_block_cause(pp_budget=0, available_req_slots=0)
        == AdmissionBlockCause.MAX_RUNNING_REQUESTS
    )
    assert (
        req_slot_block_cause(pp_budget=0, available_req_slots=7)
        == AdmissionBlockCause.PP_MICRO_BATCH
    )


def test_budget_state_kv_exhausted():
    adder = _adder(has_capacity=False, remaining_total=0)
    assert adder.budget_state() == AddReqResult.NO_TOKEN
    assert adder.stop_cause == AdmissionBlockCause.KV_TOKENS


def test_budget_state_swa_exhausted_while_full_has_room():
    adder = _adder(has_capacity=False, remaining_total=100, is_hybrid_swa=True)
    assert adder.budget_state() == AddReqResult.NO_TOKEN
    assert adder.stop_cause == AdmissionBlockCause.SWA_TOKENS


def test_budget_state_non_swa_model_never_blames_swa():
    adder = _adder(has_capacity=False, remaining_total=100, is_hybrid_swa=False)
    assert adder.budget_state() == AddReqResult.NO_TOKEN
    assert adder.stop_cause == AdmissionBlockCause.KV_TOKENS


def test_budget_state_mamba_slots_exhausted():
    adder = _adder(rem_mamba_slots=0)
    assert adder.budget_state() == AddReqResult.NO_TOKEN
    assert adder.stop_cause == AdmissionBlockCause.MAMBA_SLOTS


def test_budget_state_kv_wins_over_mamba_when_both_are_out():
    adder = _adder(has_capacity=False, remaining_total=0, rem_mamba_slots=0)
    assert adder.budget_state() == AddReqResult.NO_TOKEN
    assert adder.stop_cause == AdmissionBlockCause.KV_TOKENS


def test_budget_state_per_pass_compute_budgets():
    adder = _adder(rem_input_tokens=0)
    assert adder.budget_state() == AddReqResult.OTHER
    assert adder.stop_cause == AdmissionBlockCause.MAX_PREFILL_TOKENS

    adder = _adder(rem_chunk_tokens=0)
    assert adder.budget_state() == AddReqResult.OTHER
    assert adder.stop_cause == AdmissionBlockCause.CHUNKED_PREFILL_SIZE


def test_budget_state_continue_clears_stale_cause():
    adder = _adder(rem_mamba_slots=3, rem_chunk_tokens=8)
    assert adder.budget_state() == AddReqResult.CONTINUE
    assert adder.stop_cause is None


def test_memory_budget_cause_for_a_request_that_does_not_fit():
    adder = _adder(is_hybrid_swa=True, remaining_total=50, remaining_current=50)
    # Needs more FULL tokens than remain: FULL is binding.
    assert adder._memory_budget_block_cause(60) == AdmissionBlockCause.KV_TOKENS
    # FULL could cover it, so the SWA check was what rejected it.
    assert adder._memory_budget_block_cause(40) == AdmissionBlockCause.SWA_TOKENS


def test_add_one_req_prefill_max_requests_gate():
    adder = _adder()
    adder.prefill_max_requests = 0
    res = adder.add_one_req(
        MagicMock(), has_chunked_req=False, truncation_align_size=None
    )
    assert res == AddReqResult.OTHER
    assert adder.stop_cause == AdmissionBlockCause.PREFILL_MAX_REQUESTS


def test_collector_increments_both_counters_with_cause_label():
    collector = object.__new__(SchedulerMetricsCollector)
    collector.labels = {"model_name": "test"}
    collector.prefill_admission_blocked_passes_total = _RecordingMetric()
    collector.prefill_admission_blocked_requests_total = _RecordingMetric()

    collector.increment_admission_blocked(AdmissionBlockCause.MAMBA_SLOTS, 3)
    collector.increment_admission_blocked(AdmissionBlockCause.KV_TOKENS, 0)

    assert collector.prefill_admission_blocked_passes_total.calls == [
        ({"model_name": "test", "cause": "mamba_slots"}, 1),
        ({"model_name": "test", "cause": "kv_tokens"}, 1),
    ]
    assert collector.prefill_admission_blocked_requests_total.calls == [
        ({"model_name": "test", "cause": "mamba_slots"}, 3),
    ]


def test_collector_emits_max_running_requests_with_cap_source():
    collector = object.__new__(SchedulerMetricsCollector)
    collector.labels = {"model_name": "test"}
    collector.max_running_requests = _RecordingMetric()
    for name in (
        "max_total_num_tokens",
        "weight_memory_usage_gb",
        "kv_cache_memory_usage_gb",
        "page_size",
        "num_pages",
        "context_len",
        "startup_available_gpu_memory_gb",
    ):
        setattr(collector, name, _RecordingMetric())
    collector.graph_memory_usage_gb = _RecordingMetric()

    collector.emit_constants(
        max_total_num_tokens=1,
        max_total_num_tokens_swa=None,
        weight_memory_usage_gb=0.0,
        kv_cache_memory_usage_gb=0.0,
        graph_memory_usage_gb={},
        max_running_requests_under_SLO=None,
        page_size=1,
        num_pages=1,
        context_len=1,
        startup_available_gpu_memory_gb=0.0,
        max_running_requests=70,
        max_running_requests_cap_source="mamba_pool",
    )

    assert collector.max_running_requests.calls == [
        ({"model_name": "test", "cap_source": "mamba_pool"}, 70)
    ]


def test_reporter_forwards_only_when_scheduler_metrics_enabled():
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.metrics_collector = MagicMock()

    reporter.current_scheduler_metrics_enabled = False
    reporter.record_admission_block(AdmissionBlockCause.KV_TOKENS, 2)
    reporter.metrics_collector.increment_admission_blocked.assert_not_called()

    reporter.current_scheduler_metrics_enabled = True
    reporter.record_admission_block(AdmissionBlockCause.KV_TOKENS, 2)
    reporter.metrics_collector.increment_admission_blocked.assert_called_once_with(
        AdmissionBlockCause.KV_TOKENS, 2
    )
