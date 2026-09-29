"""_handle_batch_output takes non-streaming first_token_time from the scheduler."""

import asyncio
import dataclasses
import pickle
import time
import unittest
from unittest.mock import MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchStrOutput,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
)

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_tokenizer_manager() -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.speculative_algorithm = None
    tm.server_args.incremental_streaming_output = False
    tm.server_args.skip_tokenizer_init = False
    tm.server_args.batch_notify_size = 1
    tm.server_args.dp_size = 1
    tm.rid_to_state = {}
    tm.enable_metrics = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.send_to_scheduler = MagicMock()
    return tm


def _make_req_state(rid: str, stream: bool) -> ReqState:
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = stream
    obj.return_logprob = False
    obj.lora_path = None
    obj.log_metrics = False
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _make_unfinished_batch_str_output(rid: str) -> BatchStrOutput:
    kwargs = {"rids": [rid], "finished_reasons": [None], "output_strs": ["hello"]}
    for f in dataclasses.fields(BatchStrOutput):
        if (
            f.name in kwargs
            or f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        ):
            continue
        if f.name in ("output_ids", "spec_correct_drafts_histogram"):
            kwargs[f.name] = [[]]
        else:
            kwargs[f.name] = [0]
    return BatchStrOutput(**kwargs)


class TestNonStreamingFirstTokenTime(CustomTestCase):
    def _run(self, *, stream: bool, prefill_finished_time: float):
        tm = _make_tokenizer_manager()
        rid = "ttft_rid"
        state = _make_req_state(rid, stream)
        state.time_stats.created_time = time.perf_counter() - 1.0
        tm.rid_to_state[rid] = state
        sched_stats = SchedulerReqTimeStats(
            enable_metrics=True, prefill_finished_time=prefill_finished_time
        )
        batch_output = _make_unfinished_batch_str_output(rid)
        batch_output.time_stats = [pickle.loads(pickle.dumps(sched_stats))]
        arrival = time.perf_counter()
        asyncio.run(tm._handle_batch_output(batch_output))
        return state.time_stats, arrival

    def test_non_streaming_uses_scheduler_time(self):
        produced = time.perf_counter() - 0.5
        stats, arrival = self._run(stream=False, prefill_finished_time=produced)
        self.assertAlmostEqual(stats.first_token_time, produced, delta=1e-3)
        self.assertGreaterEqual(stats.last_time, arrival)

    def test_streaming_uses_arrival_time(self):
        produced = time.perf_counter() - 0.5
        stats, arrival = self._run(stream=True, prefill_finished_time=produced)
        self.assertGreaterEqual(stats.first_token_time, arrival)

    def test_falls_back_to_arrival_without_scheduler_time(self):
        stats, arrival = self._run(stream=False, prefill_finished_time=0.0)
        self.assertGreaterEqual(stats.first_token_time, arrival)


if __name__ == "__main__":
    unittest.main()
