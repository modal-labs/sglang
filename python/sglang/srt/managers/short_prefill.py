"""Adaptive chunk sharing; copied over the engine's short_prefill module.

Reserve only the estimated work of whole waiting prefills that fit. A continuing
request retains at least chunk_size tokens when the normal budget permits it.
The engine still owns cache rematching, memory admission, and the single-chunk
invariant; estimates do not guarantee admission.
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
    original_budget = adder.rem_chunk_tokens
    if (
        threshold <= 0
        or not waiting_queue
        or original_budget is None
        or adder.dllm_config is not None
        or original_budget <= chunk_size
    ):
        return adder.add_chunked_req(chunked_req)

    total_budget = min(original_budget, batch_size)
    spare = max(0, total_budget - chunk_size)
    # Round reservations up so the continuing chunk remains page aligned.
    page_size = adder.page_size
    selected = []
    reserved = 0
    for req in waiting_queue:
        uncached = max(0, len(req.origin_input_ids) - req.num_matched_prefix_tokens)
        cost = (uncached + page_size - 1) // page_size * page_size
        if 0 < uncached <= threshold and cost <= spare - reserved:
            selected.append(req)
            reserved += cost
        elif not scan_waiting:
            break
        if reserved == spare:
            break

    if not selected:
        return adder.add_chunked_req(chunked_req)

    continuing_budget = total_budget - reserved
    adder.rem_chunk_tokens = continuing_budget
    try:
        continuing = adder.add_chunked_req(chunked_req)
    except BaseException:
        adder.rem_chunk_tokens = original_budget
        raise
    consumed = continuing_budget - adder.rem_chunk_tokens
    selected_ids = {id(req) for req in selected}
    waiting_queue[:] = selected + [
        req for req in waiting_queue if id(req) not in selected_ids
    ]
    # If the continuing request finishes early, return the unused space too.
    adder.rem_chunk_tokens = max(0, total_budget - consumed)
    return continuing
