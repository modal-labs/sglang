"""Regression tests for the private K3 SiTU MoE PDL launch policy."""

from __future__ import annotations

import ast
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_PATH = _REPO_ROOT / "python/sglang/kernels/ops/moe/trtllm_gen_moe.py"


def test_private_situ_launches_gate_pdl_by_token_count():
    source = _SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_TRTLLM_MOE_PDL_MAX_TOKENS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert (
        ast.unparse(assignments[0].value)
        == "envs.SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS.get()"
    )

    assert source.count("num_tokens <= _TRTLLM_MOE_PDL_MAX_TOKENS,  # enable_pdl") == 2
    assert "True,  # enable_pdl" not in source
