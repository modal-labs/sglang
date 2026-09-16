"""CPU-only privacy, boundedness, wrapper and mutation checks."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

MODULE = Path(__file__).parents[2] / "python/sglang/srt/managers/queue_diagnostics.py"
spec = importlib.util.spec_from_file_location("queue_diagnostics", MODULE)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class MetadataOnly:
    def __len__(self):
        return 17

    def __iter__(self):
        raise AssertionError("Must not inspect token/device contents")

    def item(self):
        raise AssertionError("Must not synchronize CUDA")


class TestQueueDiagnostics(unittest.TestCase):
    def req(self):
        return NS(
            rid="test",
            bootstrap_room=42,
            origin_input_ids=MetadataOnly(),
            prefix_indices=MetadataOnly(),
            output_ids=[],
            time_stats=NS(),
            num_matched_prefix_tokens=12,
        )

    def scheduler(self):
        return NS(
            ps=NS(attn_tp_rank=0, pp_rank=0),
            waiting_queue=[self.req()],
            running_batch=NS(reqs=[]),
            chunked_req=None,
            disagg_prefill_inflight_queue=[self.req()],
            disagg_decode_prealloc_queue=NS(queue=[NS(req=self.req())]),
        )

    def test_metadata_does_not_read_contents(self):
        result = diag.request_metadata(NS(req=self.req(), waiting_for_input=True))
        self.assertEqual(result["input_tokens"], 17)
        self.assertTrue(result["waiting_for_input"])
        self.assertNotIn("origin_input_ids", json.dumps(result))

    def test_bounded_snapshot(self):
        s = self.scheduler()
        s.waiting_queue = [self.req() for _ in range(300)]
        result = diag.snapshot(s)
        self.assertEqual(len(result["waiting"]["requests"]), 256)
        self.assertTrue(result["waiting"]["truncated"])
        self.assertEqual(result["disagg_prefill_inflight_queue"]["count"], 1)
        json.dumps(result, allow_nan=False)

    def test_history_is_bounded_and_immutable(self):
        s = self.scheduler()
        old = diag.ENABLED
        try:
            diag.ENABLED = True
            for _ in range(100):
                diag.record_prefill(s, [s.waiting_queue[0]], "CONTINUE")
            self.assertEqual(len(s._queue_diagnostic_history), 64)
            s.waiting_queue[0].rid = "changed"
            self.assertEqual(
                s._queue_diagnostic_history[-1]["selected"]["requests"][0]["rid"],
                "test",
            )
        finally:
            diag.ENABLED = old

    def test_disabled_and_other_rank_have_no_history(self):
        s = self.scheduler()
        old = diag.ENABLED
        try:
            diag.ENABLED = False
            diag.record_prefill(s, [], "none")
            self.assertFalse(hasattr(s, "_queue_diagnostic_history"))
            diag.ENABLED = True
            s.ps.attn_tp_rank = 1
            diag.record_prefill(s, [], "none")
            self.assertFalse(hasattr(s, "_queue_diagnostic_history"))
        finally:
            diag.ENABLED = old


if __name__ == "__main__":
    unittest.main()
