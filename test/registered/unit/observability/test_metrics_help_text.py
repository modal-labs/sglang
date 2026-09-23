"""HELP text must stay parseable by strict Prometheus text-format parsers.

The PD router merges each worker's /metrics with openmetrics-parser, which
rejects braces in HELP text and then drops that worker's whole payload. One
selector-style example in a docstring hid every prefill metric in production.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang.srt.observability.metrics_collector as metrics_collector


def _documentation_strings():
    tree = ast.parse(Path(metrics_collector.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "documentation" and isinstance(
                keyword.value, ast.Constant
            ):
                yield node.lineno, keyword.value.value


class TestMetricsHelpText(unittest.TestCase):
    def test_finds_documentation_strings(self):
        self.assertGreater(len(list(_documentation_strings())), 50)

    def test_help_text_has_no_braces(self):
        offenders = [
            (lineno, text)
            for lineno, text in _documentation_strings()
            if "{" in text or "}" in text
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
