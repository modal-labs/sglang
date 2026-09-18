"""Tests for per-request metrics and OpenAI usage propagation."""

import asyncio
import json
import os
import pickle
import unittest
from unittest import mock

from sglang.srt.entrypoints.openai.protocol import UsageInfo
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ReqState
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
    convert_time_to_realtime,
)
from sglang.srt.observability.request_metrics import (
    RequestMetrics,
    _rfc3339_ms,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRequestMetrics(CustomTestCase):
    def test_generate_req_input_logs_metrics_by_default(self):
        self.assertTrue(GenerateReqInput(text="x").log_metrics)

    def test_req_state_request_metrics_skip_defaults_false(self):
        state = ReqState([], False, asyncio.Event(), object(), APIServerReqTimeStats())
        self.assertFalse(state.request_metrics_skip)

    def _api_stats(self) -> APIServerReqTimeStats:
        return APIServerReqTimeStats(
            created_time=100.0,
            response_sent_to_client_time=100.35,
            finished_time=100.95,
        )

    def _scheduler_stats(self) -> SchedulerReqTimeStats:
        return SchedulerReqTimeStats(
            forward_entry_time=100.05,
            prefill_finished_time=100.25,
            first_token_time=100.25,
        )

    def test_disabled_by_default(self):
        with envs.SGLANG_ENABLE_REQUEST_METRICS.override(False):
            self.assertFalse(RequestMetrics.from_env().enabled)

    def test_from_env_modal_ids(self):
        with temp_set_env(MODAL_APP_ID="ap-test", MODAL_TASK_ID="ta-test"):
            metrics = RequestMetrics.from_env()
            self.assertEqual(metrics.deployment_id, "ap-test")
            self.assertEqual(metrics.replica_id, "ta-test")

    def test_from_env_ids_none_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False) as environ:
            environ.pop("MODAL_APP_ID", None)
            environ.pop("MODAL_TASK_ID", None)
            metrics = RequestMetrics.from_env()
            self.assertIsNone(metrics.deployment_id)
            self.assertIsNone(metrics.replica_id)

    def test_build_full_timeline(self):
        metrics = RequestMetrics(True, "deployment", "replica").build(
            "request-id",
            self._api_stats(),
            self._scheduler_stats(),
            prompt_tokens=1000,
            cached_tokens=600,
            completion_tokens=40,
            stream=True,
        )
        self.assertEqual(
            list(metrics),
            [
                "rid",
                "accepted_at",
                "deployment_id",
                "replica_id",
                "stream",
                "prefill_tokens_uncached",
                "prefill_tokens_cached",
                "prefill_tokens_cached_device",
                "prefill_tokens_cached_host",
                "prefill_tokens_cached_storage",
                "output_tokens",
                "prefill_queue_ms",
                "prefill_ms",
                "decode_queue_ms",
                "decode_ms",
                "first_token_generated_at_ms_offset",
            ],
        )
        self.assertEqual(metrics["prefill_queue_ms"], 50)
        self.assertEqual(metrics["prefill_ms"], 200)
        self.assertEqual(metrics["decode_queue_ms"], 100)
        self.assertEqual(metrics["decode_ms"], 600)
        self.assertEqual(metrics["first_token_generated_at_ms_offset"], 250)
        self.assertEqual(metrics["prefill_tokens_uncached"], 400)
        self.assertEqual(metrics["prefill_tokens_cached"], 600)
        self.assertEqual(metrics["prefill_tokens_cached_device"], 600)
        self.assertEqual(metrics["prefill_tokens_cached_host"], 0)
        self.assertEqual(metrics["prefill_tokens_cached_storage"], 0)
        self.assertEqual(metrics["output_tokens"], 40)
        self.assertEqual(metrics["deployment_id"], "deployment")
        self.assertEqual(metrics["replica_id"], "replica")
        self.assertTrue(metrics["stream"])
        self.assertRegex(
            metrics["accepted_at"],
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
        )
        self.assertEqual(
            metrics["accepted_at"],
            _rfc3339_ms(convert_time_to_realtime(100.0)),
        )

        non_stream = RequestMetrics(True, None, None).build(
            "request-id",
            self._api_stats(),
            self._scheduler_stats(),
            prompt_tokens=1,
            cached_tokens=0,
            completion_tokens=1,
            stream=False,
        )
        self.assertFalse(non_stream["stream"])

    def test_build_non_streaming_decode_boundary(self):
        api_stats = APIServerReqTimeStats(
            created_time=100.0,
            response_sent_to_client_time=100.95,
            finished_time=100.95,
        )
        scheduler_stats = SchedulerReqTimeStats(
            forward_entry_time=100.05,
            prefill_finished_time=100.25,
            first_token_time=100.25,
        )
        metrics = RequestMetrics(True, None, None).build(
            "request-id",
            api_stats,
            scheduler_stats,
            prompt_tokens=1,
            cached_tokens=0,
            completion_tokens=1,
            stream=False,
        )
        self.assertEqual(metrics["decode_queue_ms"], 0)
        self.assertEqual(metrics["decode_ms"], 700)

    def test_build_without_scheduler_stats(self):
        metrics = RequestMetrics(True, None, "replica").build(
            "request-id",
            self._api_stats(),
            None,
            prompt_tokens=4,
            cached_tokens=1,
            completion_tokens=2,
            stream=True,
        )
        self.assertIsNone(metrics["prefill_queue_ms"])
        self.assertIsNone(metrics["prefill_ms"])
        self.assertIsNone(metrics["decode_queue_ms"])
        self.assertIsNone(metrics["first_token_generated_at_ms_offset"])
        self.assertEqual(metrics["decode_ms"], 600)
        self.assertEqual(metrics["prefill_tokens_uncached"], 3)

    def _build_with_cached(
        self,
        cached_tokens: int,
        cached_tokens_details=None,
    ):
        return RequestMetrics(True, None, None).build(
            "request-id",
            self._api_stats(),
            self._scheduler_stats(),
            prompt_tokens=10,
            cached_tokens=cached_tokens,
            completion_tokens=1,
            stream=True,
            cached_tokens_details=cached_tokens_details,
        )

    def test_cached_token_tier_split(self):
        metrics = self._build_with_cached(10, {"device": 6, "host": 3, "storage": 1})
        self.assertEqual(metrics["prefill_tokens_cached_device"], 6)
        self.assertEqual(metrics["prefill_tokens_cached_host"], 3)
        self.assertEqual(metrics["prefill_tokens_cached_storage"], 1)
        self.assertEqual(
            metrics["prefill_tokens_cached_device"]
            + metrics["prefill_tokens_cached_host"]
            + metrics["prefill_tokens_cached_storage"],
            metrics["prefill_tokens_cached"],
        )

    def test_cached_token_tier_split_no_details(self):
        metrics = self._build_with_cached(7, None)
        self.assertEqual(metrics["prefill_tokens_cached_device"], 7)
        self.assertEqual(metrics["prefill_tokens_cached_host"], 0)
        self.assertEqual(metrics["prefill_tokens_cached_storage"], 0)

    def test_cached_token_tier_split_no_storage(self):
        metrics = self._build_with_cached(5, {"device": 3, "host": 2})
        self.assertEqual(metrics["prefill_tokens_cached_device"], 3)
        self.assertEqual(metrics["prefill_tokens_cached_host"], 2)
        self.assertEqual(metrics["prefill_tokens_cached_storage"], 0)

    def test_cached_token_tier_split_mismatched_sum(self):
        metrics = self._build_with_cached(5, {"device": 2, "host": 2})
        self.assertEqual(metrics["prefill_tokens_cached_device"], 3)
        self.assertEqual(metrics["prefill_tokens_cached_host"], 2)
        self.assertEqual(metrics["prefill_tokens_cached_storage"], 0)

    def test_build_unstamped_and_negative_clamped(self):
        api_stats = self._api_stats()
        api_stats.response_sent_to_client_time = 0.0
        scheduler_stats = self._scheduler_stats()
        scheduler_stats.forward_entry_time = 99.99
        metrics = RequestMetrics(True, None, None).build(
            "request-id",
            api_stats,
            scheduler_stats,
            prompt_tokens=1,
            cached_tokens=0,
            completion_tokens=1,
            stream=True,
        )
        self.assertIsNone(metrics["decode_queue_ms"])
        self.assertIsNone(metrics["decode_ms"])
        self.assertEqual(metrics["prefill_queue_ms"], 0)

    def test_scheduler_stats_pickle_roundtrip_keeps_math(self):
        api_stats = self._api_stats()
        scheduler_stats = self._scheduler_stats()
        scheduler_stats.enable_metrics = True
        roundtrip = pickle.loads(pickle.dumps(scheduler_stats))
        before = RequestMetrics(True, None, None).build(
            "request-id",
            api_stats,
            scheduler_stats,
            prompt_tokens=1,
            cached_tokens=0,
            completion_tokens=1,
            stream=True,
        )
        after = RequestMetrics(True, None, None).build(
            "request-id",
            api_stats,
            roundtrip,
            prompt_tokens=1,
            cached_tokens=0,
            completion_tokens=1,
            stream=True,
        )
        for key in (
            "prefill_queue_ms",
            "prefill_ms",
            "decode_queue_ms",
            "first_token_generated_at_ms_offset",
        ):
            self.assertAlmostEqual(before[key], after[key], delta=1)

    def test_log_line_is_single_json(self):
        metrics = {"rid": "request-id", "stream": True}
        with self.assertLogs(
            "sglang.srt.observability.request_metrics", level="INFO"
        ) as logs:
            RequestMetrics(True, None, None).log(metrics)
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertTrue(message.startswith("REQUEST_METRICS "))
        self.assertEqual(json.loads(message.split(" ", 1)[1]), metrics)

    def test_usage_info_passthrough(self):
        metrics = {"rid": "x"}
        usage = UsageProcessor.calculate_response_usage(
            [
                {
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "request_metrics": metrics,
                    }
                }
            ]
        )
        self.assertEqual(usage.request_metrics, metrics)
        self.assertIsNone(
            UsageProcessor.calculate_response_usage(
                [{"meta_info": {"prompt_tokens": 3, "completion_tokens": 2}}]
            ).request_metrics
        )
        streaming_usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 3},
            reasoning_tokens={0: 0},
            completion_tokens={0: 2},
            cached_tokens={0: 0},
            n_choices=1,
            request_metrics={"rid": "y"},
        )
        self.assertEqual(streaming_usage.request_metrics, {"rid": "y"})
        self.assertIsNone(UsageInfo().model_dump()["request_metrics"])


if __name__ == "__main__":
    unittest.main()
