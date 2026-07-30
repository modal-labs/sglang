from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.attention.fla import (
    chunk_delta_h,
    chunk_intra,
    chunk_intra_token_parallel,
)
from sglang.kernels.ops.attention.fla import kda as kda_ops
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _unwrap_autotuner(kernel):
    while not hasattr(kernel, "configs") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    return kernel


def test_k3_prefill_autotune_regimes_cross_exact_fusion_boundary():
    chunk_counts = kda_ops._k3_prefill_autotune_chunk_counts(12)

    assert chunk_counts == (1, 22)
    assert chunk_counts[0] * 12 <= kda_ops.KDA_FUSED_INTRA_MAX_CTAS
    assert chunk_counts[1] * 12 > kda_ops.KDA_FUSED_INTRA_MAX_CTAS


def test_k3_prefill_autotune_results_are_persistent():
    kernels = (
        kda_ops.kda_gate_chunk_cumsum_vector_kernel,
        chunk_intra.chunk_kda_fwd_kernel_inter_solve_fused,
        chunk_intra_token_parallel.chunk_kda_fwd_kernel_intra_token_parallel,
        kda_ops.chunk_gla_fwd_kernel_o,
    )

    assert all(_unwrap_autotuner(kernel).cache_results for kernel in kernels)


def test_k3_state_update_has_no_token_count_jit_bucket():
    autotuner = _unwrap_autotuner(
        chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    )

    assert "NT_BUCKET" not in autotuner.keys
    assert "NT_BUCKET" not in autotuner.arg_names


def test_text_model_precompiles_kda_before_ready(monkeypatch):
    class FakeKDA:
        def __init__(self):
            self.local_num_heads = 12
            self.head_k_dim = 128
            self.head_v_dim = 128
            self.config = SimpleNamespace(dtype=torch.bfloat16)
            self.A_log = torch.empty(1, 1, 12, 1, dtype=torch.float32)
            self.dt_bias = torch.empty(12 * 128, dtype=torch.float32)
            self.attn = SimpleNamespace(lower_bound=-5.0)

    model = object.__new__(kimi_k3.KimiK3LinearForCausalLM)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(layers=[SimpleNamespace(self_attn=FakeKDA())]),
    )
    object.__setattr__(model, "config", SimpleNamespace())

    precompile = Mock(return_value=())
    marker = Mock()
    tp_group = Mock()
    monkeypatch.setattr(kimi_k3, "KimiK3DeltaAttention", FakeKDA)
    monkeypatch.setattr(
        kimi_k3,
        "get_server_args",
        lambda: SimpleNamespace(
            linear_attn_prefill_backend="triton",
            linear_attn_backend="triton",
        ),
    )
    monkeypatch.setattr(
        kimi_k3,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_size=1),
    )
    monkeypatch.setattr(kimi_k3, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(kimi_k3, "rank0_log", marker)
    monkeypatch.setattr(
        kda_ops,
        "precompile_k3_triton_prefill_kernels",
        precompile,
    )
    monkeypatch.setattr(
        "sglang.srt.configs.mamba_utils.mamba2_state_dtype",
        lambda config: SimpleNamespace(temporal=torch.bfloat16),
    )

    model.precompile_kernels_after_loading()

    precompile.assert_called_once_with(
        num_heads=12,
        head_dim=128,
        value_dim=128,
        activation_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        a_log_dtype=torch.float32,
        dt_bias_dtype=torch.float32,
        lower_bound=-5.0,
        device=torch.device("cpu"),
    )
    tp_group.barrier.assert_not_called()
    assert marker.call_args.args[0].startswith("K3_KDA_TRITON_PREFILL_READY ")


def test_text_model_skips_kda_precompile_for_non_triton_prefill(monkeypatch):
    model = object.__new__(kimi_k3.KimiK3LinearForCausalLM)
    get_tp_group = Mock()
    precompile = Mock()
    monkeypatch.setattr(
        kimi_k3,
        "get_server_args",
        lambda: SimpleNamespace(
            linear_attn_prefill_backend="cutedsl",
            linear_attn_backend="triton",
        ),
    )
    monkeypatch.setattr(kimi_k3, "get_tp_group", get_tp_group)
    monkeypatch.setattr(
        kda_ops,
        "precompile_k3_triton_prefill_kernels",
        precompile,
    )

    model.precompile_kernels_after_loading()

    get_tp_group.assert_not_called()
    precompile.assert_not_called()


def test_kda_precompile_broadcasts_rank_zero_failure(monkeypatch):
    class FakeKDA:
        local_num_heads = 12
        head_k_dim = 128
        head_v_dim = 128
        config = SimpleNamespace(dtype=torch.bfloat16)
        A_log = torch.empty(1, 1, 12, 1, dtype=torch.float32)
        dt_bias = torch.empty(12 * 128, dtype=torch.float32)
        attn = SimpleNamespace(lower_bound=-5.0)

    model = object.__new__(kimi_k3.KimiK3LinearForCausalLM)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(layers=[SimpleNamespace(self_attn=FakeKDA())]),
    )
    object.__setattr__(model, "config", SimpleNamespace())
    tp_group = Mock()
    tp_group.broadcast_object.side_effect = lambda value, src: value
    monkeypatch.setattr(kimi_k3, "KimiK3DeltaAttention", FakeKDA)
    monkeypatch.setattr(
        kimi_k3,
        "get_server_args",
        lambda: SimpleNamespace(
            linear_attn_prefill_backend="triton",
            linear_attn_backend="triton",
        ),
    )
    monkeypatch.setattr(
        kimi_k3,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_size=8),
    )
    monkeypatch.setattr(kimi_k3, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        kda_ops,
        "precompile_k3_triton_prefill_kernels",
        Mock(side_effect=RuntimeError("synthetic autotune failure")),
    )
    monkeypatch.setattr(
        "sglang.srt.configs.mamba_utils.mamba2_state_dtype",
        lambda config: SimpleNamespace(temporal=torch.bfloat16),
    )

    with pytest.raises(
        RuntimeError,
        match="K3 KDA Triton precompile failed on TP rank 0.*synthetic",
    ):
        model.precompile_kernels_after_loading()

    tp_group.broadcast_object.assert_called_once()
    tp_group.all_gather_object.assert_not_called()


def test_vl_wrapper_delegates_text_precompile_before_vision():
    events = []
    model = object.__new__(kimi_k3.KimiK3ForConditionalGeneration)
    object.__setattr__(
        model,
        "language_model",
        SimpleNamespace(precompile_kernels_after_loading=lambda: events.append("text")),
    )
    object.__setattr__(model, "config", SimpleNamespace(language_only=False))
    object.__setattr__(
        model,
        "vision_tower",
        SimpleNamespace(
            precompile_fused_rope=lambda: events.append("rope") or False,
            precompile_attention_backend=lambda: events.append("attention") or False,
        ),
    )

    model.precompile_kernels_after_loading()

    assert events == ["text", "rope", "attention"]
