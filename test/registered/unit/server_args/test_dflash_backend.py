from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.speculative_hook import (
    _resolve_dflash_draft_attention_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_draft_attention_backend="trtllm_mha",
        speculative_draft_model_path="/draft",
        speculative_draft_model_revision=None,
        trust_remote_code=False,
        json_model_override_args="{}",
    )


def _patch_draft_config(monkeypatch, config, layer_types):
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_utils.get_dflash_layer_types",
        lambda _: layer_types,
    )


def test_dflash_keeps_trtllm_mha_for_all_sliding_draft(monkeypatch):
    config = SimpleNamespace(num_hidden_layers=5, is_causal=False)
    _patch_draft_config(monkeypatch, config, ["sliding_attention"] * 5)
    args = _args()

    _resolve_dflash_draft_attention_backend(args)

    assert args.speculative_draft_attention_backend == "trtllm_mha"


def test_dflash_keeps_trtllm_mha_for_explicitly_causal_draft(monkeypatch):
    text_config = SimpleNamespace(num_hidden_layers=5, is_causal=True)
    config = SimpleNamespace(text_config=text_config)
    _patch_draft_config(monkeypatch, config, ["full_attention"] * 5)
    args = _args()

    _resolve_dflash_draft_attention_backend(args)

    assert args.speculative_draft_attention_backend == "trtllm_mha"


@pytest.mark.parametrize(
    "layer_types",
    [
        ["full_attention"] * 5,
        ["sliding_attention"] * 4 + ["full_attention"],
        ["sliding_attention"] * 4,
    ],
)
def test_dflash_rejects_trtllm_mha_for_unsafe_draft(monkeypatch, layer_types):
    config = SimpleNamespace(num_hidden_layers=5, is_causal=False)
    _patch_draft_config(monkeypatch, config, layer_types)
    args = _args()

    _resolve_dflash_draft_attention_backend(args)

    assert args.speculative_draft_attention_backend == "flashinfer"
