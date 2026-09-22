"""Exercise the deployed acceptance watchdog with completed-request counters.

AST-loading avoids building a Modal image. Timing and metric bodies are controlled,
while the production parser, window logic, thresholds and diagnostics execute.
"""

from __future__ import annotations

import ast
import http.client
import math
import re
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVE_FILE = REPO_ROOT / "deploy/instinct/serve.py"
WATCHDOG_NAMES = (
    "MINUTES",
    "PORT",
    "SPEC_ACCEPT_METRICS_URL",
    "SPEC_ACCEPT_WINDOW_SECONDS",
    "SPEC_ACCEPT_COLLAPSE_THRESHOLD",
    "SPEC_ACCEPT_MIN_VERIFY_CALLS",
    "SPEC_ACCEPT_POLL_SECONDS",
    "SPEC_ACCEPT_SUSTAINED_POLLS",
    "SPEC_ACCEPT_MAX_SAMPLE_GAP_SECONDS",
    "_SPEC_ACCEPT_COUNTERS",
    "_SPEC_ACCEPT_COUNTER_RE",
    "SpecAcceptWatchdog",
)


def _load_watchdog_namespace():
    tree = ast.parse(SERVE_FILE.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") in WATCHDOG_NAMES
        )
        or (isinstance(node, ast.ClassDef) and node.name in WATCHDOG_NAMES)
    ]
    ns = {
        "re": re,
        "math": math,
        "time": time,
        "urllib": urllib,
        "http": http,
        "deque": deque,
        "Callable": Callable,
        "print": lambda *_a, **_kw: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SERVE_FILE), "exec"), ns)
    return ns


def _metrics_body(generation_tokens, verify_calls):
    return (
        f'sglang:generation_tokens_total{{model_name="K3"}} {generation_tokens}\n'
        f'sglang:spec_verify_calls_total{{model_name="K3"}} {verify_calls}\n'
    )


class _Scripted:
    def __init__(self, ns, **kwargs):
        self.now = 1000.0
        self.generation_tokens = 0.0
        self.verify_calls = 0.0
        self.body = None
        self.fail_with = None
        self.watchdog = ns["SpecAcceptWatchdog"](
            read_metrics=self._read_metrics, clock=lambda: self.now, **kwargs
        )

    def _read_metrics(self):
        if self.fail_with is not None:
            raise self.fail_with
        if self.body is not None:
            return self.body
        return _metrics_body(self.generation_tokens, self.verify_calls)

    def poll(self, dt, *, verify_calls=0.0, accept_length=4.0):
        self.now += dt
        self.verify_calls += verify_calls
        self.generation_tokens += verify_calls * accept_length
        return self.watchdog.check()


class TestDeploySpecAcceptWatchdog(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_watchdog_namespace()
        cls.window = cls.ns["SPEC_ACCEPT_WINDOW_SECONDS"]
        cls.threshold = cls.ns["SPEC_ACCEPT_COLLAPSE_THRESHOLD"]
        cls.min_verify = cls.ns["SPEC_ACCEPT_MIN_VERIFY_CALLS"]
        cls.poll = cls.ns["SPEC_ACCEPT_POLL_SECONDS"]

    def _run(self, s, seconds, **kwargs):
        return [s.poll(self.poll, **kwargs) for _ in range(int(seconds // self.poll))]

    def test_production_thresholds_are_preserved(self):
        self.assertEqual(
            (self.window, self.threshold, self.min_verify), (300, 1.5, 200)
        )
        self.assertEqual((self.poll, self.ns["SPEC_ACCEPT_SUSTAINED_POLLS"]), (10, 30))
        self.assertEqual(self.ns["SPEC_ACCEPT_MAX_SAMPLE_GAP_SECONDS"], 20)

    def test_healthy_acceptance_never_fires(self):
        s = _Scripted(self.ns)
        verdicts = self._run(s, 3 * self.window, verify_calls=50, accept_length=3.8)
        self.assertEqual(set(verdicts), {None})

    def test_no_verdict_until_window_is_full(self):
        s = _Scripted(self.ns)
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertEqual(set(verdicts), {None})
        self.assertIsNotNone(s.poll(self.poll, verify_calls=50, accept_length=1.0))

    def test_idle_and_low_traffic_replica_gives_no_verdict(self):
        s = _Scripted(self.ns)
        self._run(s, 2 * self.window)
        self.assertIsNone(s.poll(self.poll))
        # Below the per-window verify-call floor even at accept 1.0.
        per_poll = (self.min_verify - 1) / (self.window / self.poll + 1)
        verdicts = self._run(
            s, 2 * self.window, verify_calls=per_poll, accept_length=1.0
        )
        self.assertEqual(set(verdicts), {None})

    def test_sustained_collapse_fires_with_diagnostics_and_recovers(self):
        s = _Scripted(self.ns)
        self._run(s, 2 * self.window, verify_calls=50, accept_length=4.0)
        # Collapse: every poll from here on rejects every draft token.
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertIsNone(
            verdicts[0], "one low poll inside a healthy window is not a verdict"
        )
        self.assertIsNotNone(verdicts[-1])
        self.assertIn("spec accept collapse: 1.00 tokens/verify", verdicts[-1])
        self.assertIn("verify calls) <= 1.5", verdicts[-1])
        for _ in range(3):
            self.assertIsNotNone(s.poll(self.poll, verify_calls=50, accept_length=1.0))
        # Recovery clears the verdict once the window is healthy again.
        verdicts = self._run(
            s, self.window + self.poll, verify_calls=50, accept_length=4.0
        )
        self.assertIsNone(verdicts[-1])

    def test_mixed_window_uses_ratio_of_counter_deltas(self):
        s = _Scripted(self.ns)
        self._run(s, self.window, verify_calls=50, accept_length=4.0)
        # Half the window at 4.0 and half at 1.0 averages 2.5 > 1.5: no verdict.
        verdicts = self._run(s, self.window / 2, verify_calls=50, accept_length=1.0)
        self.assertEqual(set(verdicts), {None})
        # Once the window is entirely collapsed it fires.
        verdicts = self._run(
            s, self.window / 2 + self.poll, verify_calls=50, accept_length=1.0
        )
        self.assertIsNotNone(verdicts[-1])

    def test_transient_dip_shorter_than_window_does_not_fire(self):
        s = _Scripted(self.ns)
        self._run(s, self.window, verify_calls=50, accept_length=4.0)
        verdicts = self._run(s, 120, verify_calls=50, accept_length=1.0)
        verdicts += self._run(s, self.window, verify_calls=50, accept_length=4.0)
        self.assertEqual(set(verdicts), {None})

    def test_counter_reset_discards_window(self):
        s = _Scripted(self.ns)
        self._run(s, 2 * self.window, verify_calls=50, accept_length=1.0)
        self.assertIsNotNone(s.poll(self.poll, verify_calls=50, accept_length=1.0))
        s.generation_tokens = 0.0
        s.verify_calls = 0.0
        # The reset poll seeds a fresh window that has to fill again.
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertEqual(set(verdicts), {None})
        self.assertIsNotNone(s.poll(self.poll, verify_calls=50, accept_length=1.0))

    def test_metrics_outage_drops_window_not_extends_it(self):
        # Review 4065862021: counters cannot be observed across a /metrics
        # outage, so stale samples must not extend the effective window after
        # recovery.
        s = _Scripted(self.ns)
        self._run(s, self.window, verify_calls=50, accept_length=4.0)
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertIsNotNone(verdicts[-1])
        s.fail_with = urllib.error.URLError("outage")
        self.assertIsNone(s.poll(self.window))
        s.fail_with = None
        # The pre-outage samples are gone: the first post-recovery poll has no
        # full window, and a verdict needs a freshly filled one.
        self.assertIsNone(
            s.poll(self.poll, verify_calls=50, accept_length=1.0),
            "recovery must not produce an outage-extended verdict",
        )
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertIsNotNone(verdicts[-1], "a refilled window fires again")

    def test_delayed_poll_drops_pre_gap_samples(self):
        # A poll delayed past the gap tolerance breaks window continuity the
        # same way an outage does.
        s = _Scripted(self.ns)
        self._run(s, self.window, verify_calls=50, accept_length=4.0)
        self._run(s, self.window, verify_calls=50, accept_length=1.0)
        gap = 6 * self.poll
        self.assertIsNone(
            s.poll(gap, verify_calls=50, accept_length=1.0),
            "the first poll after the gap has no contiguous window",
        )
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertIsNotNone(verdicts[-1])

    def test_short_gap_within_tolerance_preserves_window(self):
        s = _Scripted(self.ns)
        self._run(s, self.window, verify_calls=50, accept_length=4.0)
        self._run(s, self.window, verify_calls=50, accept_length=1.0)
        # A single delayed poll within the 2-poll tolerance keeps the window.
        verdict = s.poll(self.poll * 1.5, verify_calls=50, accept_length=1.0)
        self.assertIsNotNone(verdict, "a <=2-poll gap must not reset the window")
        # A single failed scrape also stays within the tolerance: the window
        # survives one missed poll and the verdict continues.
        s.fail_with = urllib.error.URLError("hiccup")
        self.assertIsNone(s.poll(self.poll))
        s.fail_with = None
        self.assertIsNotNone(
            s.poll(self.poll, verify_calls=50, accept_length=1.0),
            "a single missed poll must not reset the window",
        )

    def test_metrics_fetch_failure_is_not_a_verdict(self):
        s = _Scripted(self.ns)
        self._run(s, 2 * self.window, verify_calls=50, accept_length=1.0)
        for error in (
            urllib.error.URLError("connection refused"),
            TimeoutError("timed out"),
            ConnectionResetError(),
            ValueError("bad float"),
        ):
            s.fail_with = error
            self.assertIsNone(s.poll(self.poll, verify_calls=50, accept_length=1.0))
        s.fail_with = None
        # Review 4065862021: the outage outlasted the 2-poll sample-gap
        # tolerance, so the first restored poll has no contiguous window and
        # must not produce an outage-extended verdict; the verdict returns
        # only after a fresh window fills.
        self.assertIsNone(s.poll(self.poll, verify_calls=50, accept_length=1.0))
        verdicts = self._run(s, self.window, verify_calls=50, accept_length=1.0)
        self.assertIsNotNone(verdicts[-1])

    def test_missing_or_invalid_counters_are_not_healthy_zero(self):
        for body in (
            "",
            "sglang:spec_verify_calls_total 400\n",
            "sglang:generation_tokens_total 0\n",
            _metrics_body("NaN", 400),
            _metrics_body("Inf", 400),
            _metrics_body(1, -1),
            _metrics_body("invalid", 400),
        ):
            with self.subTest(body=body):
                s = _Scripted(self.ns)
                self._run(s, self.window + self.poll, verify_calls=50, accept_length=1)
                self.assertGreater(len(s.watchdog._samples), 1)
                s.body = body
                with patch("builtins.print") as log, patch.dict(
                    self.ns, {"print": log}
                ):
                    self.assertIsNone(s.watchdog.check())
                self.assertEqual(len(s.watchdog._samples), 0)
                self.assertIn("metrics unavailable; no verdict", log.call_args.args[0])

    def test_truncated_http_response_does_not_kill_watchdog(self):
        s = _Scripted(self.ns)
        self._run(s, self.window + self.poll, verify_calls=50, accept_length=1)
        s.fail_with = http.client.IncompleteRead(b"partial metrics", 100)
        self.assertIsNone(s.poll(self.poll))
        s.fail_with = None
        self.assertIsNotNone(s.poll(self.poll, verify_calls=50, accept_length=1))

    def test_counters_sum_labels_and_ignore_unrelated_metrics(self):
        s = _Scripted(self.ns)
        s.body = (
            'sglang:generation_tokens_total{model_name="a"} 100\n'
            'sglang:generation_tokens_total{model_name="b"} 20\n'
            'sglang:spec_verify_calls_total{model_name="a"} 40\n'
            'sglang:spec_verify_calls_total{model_name="b"} 60\n'
            "sglang:generation_tokens_created 1700000000\n"
            "sglang:spec_accept_length 1.0\n"
        )
        self.assertEqual(s.watchdog._read_counters(), (120, 100))
        s.body = _metrics_body(0, 0)
        self.assertEqual(s.watchdog._read_counters(), (0, 0))

    def test_threshold_and_verify_floor_are_inclusive(self):
        for verifies, ratio, should_fire in (
            (200, 1.5, True),
            (199, 1.0, False),
            (200, 1.51, False),
        ):
            with self.subTest(verifies=verifies, ratio=ratio):
                s = _Scripted(self.ns)
                s.poll(0)
                for _ in range(29):
                    s.poll(self.poll)
                verdict = s.poll(self.poll, verify_calls=verifies, accept_length=ratio)
                self.assertEqual(verdict is not None, should_fire)

    def test_zero_verify_completions_can_mask_collapse(self):
        # The pinned engine counts all completion tokens in the numerator.
        # Ten speculative requests each produce 21 tokens over 20 verifies;
        # additional one-token requests have no verify calls. This pins the
        # calibrated aggregate arithmetic, including its accepted blind spot.
        for short_tokens, should_fire in (
            (0, True),
            (90, True),
            (91, False),
            (200, False),
        ):
            with self.subTest(short_tokens=short_tokens):
                s = _Scripted(self.ns)
                s.poll(0)
                for _ in range(29):
                    s.poll(self.poll)
                s.generation_tokens = 210 + short_tokens
                s.verify_calls = 200
                verdict = s.poll(self.poll)
                self.assertEqual(verdict is not None, should_fire)

    def test_fetch_uses_local_metrics_endpoint_and_timeout(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self):
                return _metrics_body(4, 1).encode()

        with patch.object(urllib.request, "urlopen", return_value=Response()) as fetch:
            watchdog = self.ns["SpecAcceptWatchdog"](request_timeout=7)
            self.assertEqual(watchdog._read_counters(), (4, 1))
            fetch.assert_called_once_with("http://127.0.0.1:8000/metrics", timeout=7)


if __name__ == "__main__":
    import unittest

    unittest.main()
