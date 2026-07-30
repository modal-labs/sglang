from __future__ import annotations

import gc
import weakref
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _bare_method(*, workspace=None, max_tile_n=None):
    method = object.__new__(Mxfp4MoEMethod)
    method._trtllm_gen_eager_workspace = workspace
    method._trtllm_gen_eager_workspace_max_tile_n = max_tile_n
    return method


def test_eager_workspace_is_passed_outside_capture():
    workspace = object()
    method = _bare_method(workspace=workspace, max_tile_n=256)

    with patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=False
    ):
        kwargs = method._trtllm_gen_eager_workspace_kwargs()

    assert kwargs == {"workspace": workspace, "max_tile_n": 256}


def test_capture_uses_graph_private_workspace_but_keeps_tactic_cap():
    method = _bare_method(workspace=object(), max_tile_n=256)

    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
        kwargs = method._trtllm_gen_eager_workspace_kwargs()

    assert kwargs == {"workspace": None, "max_tile_n": 256}


def test_unbound_workspace_does_not_change_launcher_kwargs():
    method = _bare_method()

    with patch.object(
        torch.cuda, "is_current_stream_capturing",
        side_effect=AssertionError("capture state must not be queried"),
    ):
        assert method._trtllm_gen_eager_workspace_kwargs() == {}


def test_deferred_finalize_output_anchors_workspace_lifetime():
    workspace = torch.empty(1, dtype=torch.uint8)
    workspace_ref = weakref.ref(workspace)
    tensor = torch.empty(0)
    output = FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=tensor,
        expert_weights=tensor,
        expanded_idx_to_permuted_idx=tensor,
        top_k=1,
        _workspace_owner=workspace,
    )

    del workspace
    gc.collect()
    assert workspace_ref() is output._workspace_owner

    del output
    gc.collect()
    assert workspace_ref() is None
