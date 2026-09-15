"""Unit tests for APIServerReqTimeStats decode-throughput behavior."""

import unittest

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGetDecodeThroughput(CustomTestCase):
    def _finished_stats(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.first_token_time = 1.5
        stats.finished_time = 2.5
        return stats

    def test_excludes_first_token_and_ttft(self):
        # decode window = 2.5 - 1.5 = 1.0s; (5 - 1) tokens / 1.0s = 4.0 tok/s
        stats = self._finished_stats()
        self.assertAlmostEqual(stats.get_decode_throughput(5), 4.0)
        self.assertAlmostEqual(
            stats.convert_to_output_meta_info(completion_tokens=5)["decode_throughput"],
            4.0,
        )

    def test_single_token_request_is_undefined(self):
        stats = self._finished_stats()
        self.assertEqual(stats.get_decode_throughput(1), 0.0)
        self.assertNotIn(
            "decode_throughput",
            stats.convert_to_output_meta_info(completion_tokens=1),
        )

    def test_unstamped_first_token_time_is_undefined(self):
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.finished_time = 2.5
        self.assertEqual(stats.get_decode_throughput(5), 0.0)
        self.assertNotIn(
            "decode_throughput",
            stats.convert_to_output_meta_info(completion_tokens=5),
        )


if __name__ == "__main__":
    unittest.main()
