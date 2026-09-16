"""Opt-in canary queue diagnostics. CPU metadata only; never prompt contents.

This is a temporary local qualification aid, not a scheduling policy. Reading
tensor lengths does not read device values or synchronize CUDA.
"""

from collections import deque
import os
import time

ENABLED = os.environ.get("SGLANG_K3_QUEUE_DIAGNOSTICS") == "1"
LIMIT = 256
TIMES = (
    "wait_queue_entry_time",
    "forward_entry_time",
    "prefill_finished_time",
    "prefill_bootstrap_queue_entry_time",
    "prefill_transfer_queue_entry_time",
    "decode_prealloc_queue_entry_time",
    "decode_transfer_queue_entry_time",
    "bootstrap_done_time",
)


def request_metadata(entry):
    req = getattr(entry, "req", entry)
    stats = getattr(req, "time_stats", None)
    origin = getattr(req, "origin_input_ids", ())
    matched = getattr(req, "num_matched_prefix_tokens", 0)
    return {
        "rid": str(getattr(req, "rid", "")),
        "room": str(getattr(req, "bootstrap_room", "")),
        "input_tokens": len(origin),
        "output_tokens": len(getattr(req, "output_ids", ())),
        "matched_tokens": matched,
        "host_hit_tokens": getattr(req, "host_hit_length", 0),
        "prefix_tokens": len(getattr(req, "prefix_indices", ())),
        "extend_tokens": getattr(req, "extend_input_len", 0),
        "min_uncached_seen": getattr(req, "min_uncached_seen", None),
        "arrival_stamp": getattr(req, "arrival_stamp", None),
        "waiting_for_input": getattr(entry, "waiting_for_input", None),
        "times": {name: getattr(stats, name, 0.0) for name in TIMES},
    }


def request_list(entries):
    return {
        "count": len(entries),
        "truncated": len(entries) > LIMIT,
        "requests": [request_metadata(req) for req in entries[:LIMIT]],
    }


def enabled(scheduler):
    return ENABLED and scheduler.ps.attn_tp_rank == 0 and scheduler.ps.pp_rank == 0


def record_prefill(scheduler, selected, result):
    if not enabled(scheduler):
        return
    history = getattr(scheduler, "_queue_diagnostic_history", None)
    if history is None:
        history = scheduler._queue_diagnostic_history = deque(maxlen=64)
    history.append(
        {
            "monotonic": time.perf_counter(),
            "wall_time": time.time(),
            "result": str(result),
            "selected": request_list(selected),
            "waiting": request_list(scheduler.waiting_queue),
            "continuing": (
                request_metadata(scheduler.chunked_req)
                if scheduler.chunked_req is not None
                else None
            ),
        }
    )


def snapshot(scheduler):
    now = time.perf_counter()
    result = {
        "monotonic": now,
        "wall_time": time.time(),
        "waiting": request_list(scheduler.waiting_queue),
        "running": request_list(scheduler.running_batch.reqs),
        "history": list(getattr(scheduler, "_queue_diagnostic_history", ())),
    }
    for name in (
        "disagg_prefill_bootstrap_queue",
        "disagg_prefill_inflight_queue",
        "disagg_decode_prealloc_queue",
        "disagg_decode_transfer_queue",
    ):
        queue = getattr(scheduler, name, None)
        if queue is not None:
            result[name] = request_list(
                queue if isinstance(queue, list) else queue.queue
            )
    result["continuing"] = (
        request_metadata(scheduler.chunked_req)
        if scheduler.chunked_req is not None
        else None
    )
    return result
