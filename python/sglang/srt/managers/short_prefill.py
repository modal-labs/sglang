"""Conditional chunk budget for an eligible short queued prefill.

This preserves queue ordering and one continuing request. It does not guarantee
admission when another resource budget prevents the waiting request from fitting.
"""


def add_chunk_with_short_prefill_budget(
    adder, chunked_req, waiting_queue, *, threshold, chunk_size, batch_size
):
    if (
        threshold <= 0
        or not waiting_queue
        or adder.rem_chunk_tokens is None
        or adder.dllm_config is not None
        or adder.rem_chunk_tokens <= chunk_size
    ):
        return adder.add_chunked_req(chunked_req)
    head = waiting_queue[0]
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
    consumed = chunk_size - adder.rem_chunk_tokens
    adder.rem_chunk_tokens = max(
        0, min(original_budget - consumed, batch_size - consumed)
    )
    return continuing
