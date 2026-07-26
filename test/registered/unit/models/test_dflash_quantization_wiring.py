from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models import dflash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _module_result(**attrs):
    module = torch.nn.Module()
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def test_dflash_attention_keeps_qkv_bf16_and_quantizes_output_projection():
    quant_config = object()
    qkv_ctor = Mock(side_effect=lambda *args, **kwargs: _module_result())
    o_proj_ctor = Mock(side_effect=lambda *args, **kwargs: _module_result())
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
    )

    with (
        patch.object(dflash, "get_parallel", return_value=SimpleNamespace(tp_size=1)),
        patch.object(dflash, "QKVParallelLinear", qkv_ctor),
        patch.object(dflash, "RowParallelLinear", o_proj_ctor),
        patch.object(dflash, "get_rope_config", return_value=(10000.0, None)),
        patch.object(dflash, "get_rope", return_value=_module_result()),
        patch.object(dflash, "RadixAttention", return_value=_module_result()),
        patch.object(
            dflash,
            "_get_dflash_layer_attention_params",
            return_value=(-1, AttentionType.ENCODER_ONLY),
        ),
    ):
        dflash.DFlashAttention(
            config,
            layer_id=0,
            quant_config=quant_config,
            prefix="layers.0.self_attn",
        )

    assert qkv_ctor.call_args.kwargs["quant_config"] is None
    assert qkv_ctor.call_args.kwargs["prefix"] == "layers.0.self_attn.qkv_proj"
    assert o_proj_ctor.call_args.kwargs["quant_config"] is quant_config
    assert o_proj_ctor.call_args.kwargs["prefix"] == "layers.0.self_attn.o_proj"


def test_dflash_mlp_quantizes_gate_up_and_down_projections():
    quant_config = object()
    gate_up_ctor = Mock(side_effect=lambda *args, **kwargs: _module_result())
    down_ctor = Mock(side_effect=lambda *args, **kwargs: _module_result())
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        hidden_act="silu",
    )

    with (
        patch.object(dflash, "MergedColumnParallelLinear", gate_up_ctor),
        patch.object(dflash, "RowParallelLinear", down_ctor),
        patch.object(dflash, "SiluAndMul", return_value=_module_result()),
    ):
        dflash.DFlashMLP(
            config,
            quant_config=quant_config,
            prefix="layers.0.mlp",
        )

    assert gate_up_ctor.call_args.kwargs["quant_config"] is quant_config
    assert gate_up_ctor.call_args.kwargs["prefix"] == "layers.0.mlp.gate_up_proj"
    assert down_ctor.call_args.kwargs["quant_config"] is quant_config
    assert down_ctor.call_args.kwargs["prefix"] == "layers.0.mlp.down_proj"


def test_dflash_fc_is_a_quantized_replicated_linear():
    quant_config = object()
    fc_ctor = Mock(
        side_effect=lambda input_size, *args, **kwargs: _module_result(
            input_size=input_size
        )
    )

    class FakeDecoderLayer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    draft_config = SimpleNamespace(
        num_target_layers=6,
        resolve_target_layer_ids=lambda **kwargs: [0, 1, 2, 3, 4, 5],
        resolve_block_size=lambda default: 8,
    )
    config = SimpleNamespace(
        hidden_size=16,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
    )

    with (
        patch.object(
            dflash.DFlashDraftModel,
            "decoder_layer_cls",
            FakeDecoderLayer,
        ),
        patch.object(dflash, "ReplicatedLinear", fc_ctor),
        patch.object(
            dflash,
            "parse_dflash_draft_config",
            return_value=draft_config,
        ),
    ):
        model = dflash.DFlashDraftModel(config, quant_config=quant_config)

    assert model.quant_config is quant_config
    assert fc_ctor.call_args.kwargs["quant_config"] is quant_config
    assert fc_ctor.call_args.kwargs["prefix"] == "fc"
    assert model.fc.input_size == 6 * config.hidden_size
