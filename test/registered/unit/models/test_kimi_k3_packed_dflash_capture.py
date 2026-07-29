from types import SimpleNamespace

import torch

from sglang.srt.layers.attn_residual import aggregate_stream_torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.kimi_k3 import KimiK3LinearModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_aggregate_stream_reference_writes_row_strided_out():
    class IdentityNorm:
        def __call__(self, value):
            return value

    class SumProjection:
        def __call__(self, value):
            return (value.sum(dim=-1, keepdim=True), None)

    prefix = torch.randn(3, 4)
    bank = torch.randn(3, 2, 4)
    expected = aggregate_stream_torch(prefix, bank, 2, SumProjection(), IdentityNorm())
    packed = torch.empty(3, 12)
    slot = packed[:, 4:8]

    actual = aggregate_stream_torch(
        prefix,
        bank,
        2,
        SumProjection(),
        IdentityNorm(),
        out=slot,
    )

    assert actual is slot
    assert not slot.is_contiguous()
    assert torch.equal(actual, expected)


def test_kimi_k3_packed_taps_match_legacy_concat_order():
    num_tokens = 3
    hidden_size = 4
    num_taps = 6
    packed = torch.empty((num_tokens, num_taps * hidden_size))
    legacy_taps = []
    fake_model = SimpleNamespace()

    for tap_idx in range(num_taps):
        hidden = torch.full((num_tokens, hidden_size), float(tap_idx))
        residual = torch.full((num_tokens, hidden_size), float(10 * tap_idx))
        expected = hidden + residual
        legacy_taps.append(expected)
        slot = packed[:, tap_idx * hidden_size : (tap_idx + 1) * hidden_size]

        returned = KimiK3LinearModel._dspark_capture_stream(
            fake_model,
            layer_idx=tap_idx,
            hidden_states=hidden,
            residual=residual,
            attn_res=None,
            out=slot,
        )

        assert returned is slot

    assert torch.equal(packed, torch.cat(legacy_taps, dim=-1))


def test_logits_processor_accepts_prepacked_aux_without_copy():
    packed = torch.randn(3, 24)
    legacy = list(packed.split(4, dim=-1))

    assert LogitsProcessor._pack_aux_hidden_states(packed) is packed
    assert torch.equal(LogitsProcessor._pack_aux_hidden_states(legacy), packed)


def test_logits_processor_keeps_packed_aux_as_one_tensor_during_verify():
    hidden = torch.randn(2, 4)
    final_pre_norm = torch.randn(2, 4)
    packed = torch.randn(2, 24)
    metadata = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        draft_extend_select_index=None,
    )

    result = LogitsProcessor._get_pruned_states(
        None,
        hidden,
        final_pre_norm,
        packed,
        metadata,
    )

    assert result[0] is hidden
    assert result[1] is final_pre_norm
    assert result[2] is packed
