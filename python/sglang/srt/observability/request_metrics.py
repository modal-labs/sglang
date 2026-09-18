"""Customer-facing per-request metrics."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
    convert_time_to_realtime,
)

logger = logging.getLogger(__name__)


def _ms(start: float, end: float) -> Optional[int]:
    if start <= 0.0 or end <= 0.0:
        return None
    return max(0, round((end - start) * 1000))


def _rfc3339_ms(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class RequestMetrics:
    """Builds the customer-facing per-request metrics dict (all durations ms ints).

    Decode boundary: the stream=True boundary — the tokenizer's first-chunk
    yield, the closest observable point before the socket write — is the
    primary contract; the stream=False boundary (the scheduler's
    first_token_time) is the fallback.
    """

    def __init__(
        self,
        enabled: bool,
        deployment_id: Optional[str],
        replica_id: Optional[str],
    ):
        self.enabled = enabled
        self.deployment_id = deployment_id
        self.replica_id = replica_id

    @classmethod
    def from_env(cls) -> RequestMetrics:
        return cls(
            enabled=envs.SGLANG_ENABLE_REQUEST_METRICS.get(),
            deployment_id=os.environ.get("MODAL_APP_ID"),
            replica_id=os.environ.get("MODAL_TASK_ID"),
        )

    def build(
        self,
        rid: str,
        api_stats: APIServerReqTimeStats,
        scheduler_stats: Optional[SchedulerReqTimeStats],
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
        stream: bool,
        cached_tokens_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decode_start = api_stats.response_sent_to_client_time
        if not stream and scheduler_stats is not None:
            decode_start = scheduler_stats.first_token_time

        if scheduler_stats is None:
            prefill_queue_ms = None
            prefill_ms = None
            decode_queue_ms = None
            first_token_generated_at_ms_offset = None
        else:
            prefill_queue_ms = _ms(
                api_stats.created_time, scheduler_stats.forward_entry_time
            )
            prefill_ms = _ms(
                scheduler_stats.forward_entry_time,
                scheduler_stats.prefill_finished_time,
            )
            decode_queue_ms = _ms(
                scheduler_stats.prefill_finished_time,
                decode_start,
            )
            first_token_generated_at_ms_offset = _ms(
                api_stats.created_time, scheduler_stats.first_token_time
            )

        if cached_tokens_details is None:
            cached_device, cached_host, cached_storage = cached_tokens, 0, 0
        else:
            cached_host = cached_tokens_details.get("host", 0)
            cached_storage = cached_tokens_details.get("storage") or 0
            cached_device = max(0, cached_tokens - cached_host - cached_storage)

        return {
            "rid": rid,
            "accepted_at": _rfc3339_ms(
                convert_time_to_realtime(api_stats.created_time)
            ),
            "deployment_id": self.deployment_id,
            "replica_id": self.replica_id,
            "stream": stream,
            "prefill_tokens_uncached": max(0, prompt_tokens - cached_tokens),
            "prefill_tokens_cached": cached_tokens,
            "prefill_tokens_cached_device": cached_device,
            "prefill_tokens_cached_host": cached_host,
            "prefill_tokens_cached_storage": cached_storage,
            "output_tokens": completion_tokens,
            "prefill_queue_ms": prefill_queue_ms,
            "prefill_ms": prefill_ms,
            "decode_queue_ms": decode_queue_ms,
            "decode_ms": _ms(decode_start, api_stats.finished_time),
            "first_token_generated_at_ms_offset": first_token_generated_at_ms_offset,
        }

    def log(self, metrics: Dict[str, Any]) -> None:
        logger.info("REQUEST_METRICS %s", json.dumps(metrics, separators=(",", ":")))
