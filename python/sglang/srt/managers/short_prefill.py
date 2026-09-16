"""Conditional chunk budget for an eligible short queued prefill.

The default preserves queue ordering and one continuing request.
The canary scan mode promotes eligible short waiters only while a long request
continues; that request still receives its guaranteed chunk every pass. It does not guarantee
admission when another resource budget prevents the waiting request from fitting.
"""

import os

SCAN_WAITING = os.environ.get("SGLANG_SHORT_PREFILL_SCAN_WAITING") == "1"


def add_chunk_with_short_prefill_budget(
    adder,
    chunked_req,
    waiting_queue,
    *,
    threshold,
    chunk_size,
    batch_size,
    scan_waiting=SCAN_WAITING,
):
    if (
        threshold <= 0
        or not waiting_queue
        or adder.rem_chunk_tokens is None
        or adder.dllm_config is not None
        or adder.rem_chunk_tokens <= chunk_size
    ):
        return adder.add_chunked_req(chunked_req)
    promoted = None
    if scan_waiting:
        eligible = [
            req
            for req in waiting_queue
            if 0
            < len(req.origin_input_ids) - req.num_matched_prefix_tokens
            <= threshold
        ]
        if not eligible:
            return adder.add_chunked_req(chunked_req)
        eligible_ids = {id(req) for req in eligible}
        promoted = eligible + [
            req for req in waiting_queue if id(req) not in eligible_ids
        ]
    head = promoted[0] if promoted is not None else waiting_queue[0]
    uncached = max(0, len(head.origin_input_ids) - head.num_matched_prefix_tokens)
    if not 0 < uncached <= threshold:
        return adder.add_chunked_req(chunked_req)

    original_budget = adder.rem_chunk_tokens
    adder.rem_chunk_tokens = chunk_size
    try:
        continuing = adder.add_chunked_req(chunked_req)
    except BaseException:
        adder.rem_chunk_tokens = original_budget
        raise
    if promoted is not None:
        waiting_queue[:] = promoted
    consumed = chunk_size - adder.rem_chunk_tokens
    adder.rem_chunk_tokens = max(
        0, min(original_budget - consumed, batch_size - consumed)
    )
    return continuing
