"""Label vocabulary for the prefill admission-block metrics.

Kept free of heavy imports so the scheduler side (``schedule_policy``,
``scheduler``) and the observability side (``metrics_collector``) can both
import it without a cycle.
"""

from __future__ import annotations


class AdmissionBlockCause:
    """First binding constraint that ended a prefill admission pass.

    Each scheduler step the prefill adder admits waiting requests until one
    gate says stop. That gate is what caps concurrency for the step: the FULL
    KV token budget, the SWA budget, the Mamba state pool, request slots, one
    of the per-pass compute budgets, or a policy that asked to wait. These are
    the ``cause`` label values of ``sglang:prefill_admission_blocked_*``.
    """

    # Memory pools.
    KV_TOKENS = "kv_tokens"  # FULL KV token budget (available + evictable)
    SWA_TOKENS = "swa_tokens"  # sliding-window KV budget (hybrid SWA models)
    MAMBA_SLOTS = "mamba_slots"  # Mamba / linear-attention state slots
    # Request slots.
    MAX_RUNNING_REQUESTS = "max_running_requests"  # req_to_token_pool rows
    PP_MICRO_BATCH = "pp_micro_batch"  # pipeline-parallel micro-batch size
    DISAGG_PREFILL_REQ_POOL = (
        "disagg_prefill_req_pool"  # PD prefill: rows held by prealloc / transfer queues
    )
    # Per-pass compute budgets. These stop a pass, not concurrency: the rest of
    # the queue is admitted on the next pass once the budget resets.
    MAX_PREFILL_TOKENS = "max_prefill_tokens"
    CHUNKED_PREFILL_SIZE = "chunked_prefill_size"
    PREFILL_MAX_REQUESTS = "prefill_max_requests"
    PREFILL_TILE_BUDGET = "prefill_tile_budget"  # AMD-only tile budget
    DLLM_BUDGET = "dllm_budget"
    # Policies that asked to wait.
    MIN_FREE_SLOTS_DELAY = "min_free_slots_delay"
    PREFILL_DELAYER = "prefill_delayer"
    # No HiCache load-back cause: on this tree a failed init_load_back drops the
    # host prefix and recomputes it (PrefillAdder.add_one_req), so it never
    # stops a pass.
    # The batch was already marked full by an earlier pass whose cause is unknown.
    BATCH_FULL = "batch_full"
    OTHER = "other"

    ALL = (
        KV_TOKENS,
        SWA_TOKENS,
        MAMBA_SLOTS,
        MAX_RUNNING_REQUESTS,
        PP_MICRO_BATCH,
        DISAGG_PREFILL_REQ_POOL,
        MAX_PREFILL_TOKENS,
        CHUNKED_PREFILL_SIZE,
        PREFILL_MAX_REQUESTS,
        PREFILL_TILE_BUDGET,
        DLLM_BUDGET,
        MIN_FREE_SLOTS_DELAY,
        PREFILL_DELAYER,
        BATCH_FULL,
        OTHER,
    )


def req_slot_block_cause(
    *, pp_budget: int, available_req_slots: int, pp_budget_is_running_cap: bool = False
) -> str:
    """Which request-count limit binds when no request slot can be allocated.

    ``Scheduler.get_num_allocatable_reqs`` is ``min(pp_budget, available)``, so
    the smaller operand is the binding one. When ``pp_max_micro_batch_size`` is
    at least ``max_running_requests`` (the default without pipeline
    parallelism) the "pp budget" is the running-request cap itself; the
    request pool can hold spare rows beyond that cap (K3 does), so the operands
    need not tie and the cause is ``max_running_requests`` either way. A smaller
    micro-batch limit (pipeline parallelism, or set explicitly) is
    ``pp_micro_batch`` when it is the smaller operand; ties go to the request
    pool, the knob an operator actually tunes.
    """
    if pp_budget_is_running_cap or available_req_slots <= pp_budget:
        return AdmissionBlockCause.MAX_RUNNING_REQUESTS
    return AdmissionBlockCause.PP_MICRO_BATCH
