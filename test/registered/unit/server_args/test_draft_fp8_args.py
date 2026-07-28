import argparse

import pytest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _parser():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser


def test_draft_fp8_activation_scheme_defaults_dynamic():
    args = _parser().parse_args(["--model", "dummy"])
    assert args.speculative_draft_fp8_activation_scheme == "dynamic"


def test_draft_fp8_activation_scheme_accepts_static():
    args = _parser().parse_args(
        [
            "--model",
            "dummy",
            "--speculative-draft-fp8-activation-scheme",
            "static",
        ]
    )
    assert args.speculative_draft_fp8_activation_scheme == "static"


def test_draft_fp8_activation_scheme_rejects_unknown_value():
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--model",
                "dummy",
                "--speculative-draft-fp8-activation-scheme",
                "calibrated",
            ]
        )
