import torch

from sglang.srt.speculative.dflash_worker_v2 import (
    _get_dflash_sampling_tp_group,
    _sync_dflash_sampling_results,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTPGroup:
    def __init__(self, canonical):
        self.world_size = 2
        self.canonical = canonical
        self.broadcast_calls = []

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))
        tensor.copy_(self.canonical)


def test_accept_and_bonus_use_one_packed_broadcast():
    accept_len = torch.tensor([0, 7], dtype=torch.int32)
    bonus = torch.tensor([11, 12], dtype=torch.int64)
    canonical = torch.tensor([[3, 101], [1, 202]], dtype=torch.int64)
    outcome = torch.empty((4, 2), dtype=torch.int64)
    tp_group = _FakeTPGroup(canonical)

    _sync_dflash_sampling_results(
        accept_len,
        bonus,
        tp_group=tp_group,
        outcome_buffer=outcome,
    )

    torch.testing.assert_close(accept_len, canonical[:, 0].to(torch.int32))
    torch.testing.assert_close(bonus, canonical[:, 1])
    assert len(tp_group.broadcast_calls) == 1
    assert tp_group.broadcast_calls[0][1] == 0
    broadcast_tensor = tp_group.broadcast_calls[0][0]
    assert broadcast_tensor.shape == (2, 2)
    assert broadcast_tensor.dtype == torch.int64
    assert broadcast_tensor.data_ptr() == outcome.data_ptr()


def test_single_rank_is_a_noop():
    accept_len = torch.tensor([2], dtype=torch.int32)
    bonus = torch.tensor([99], dtype=torch.int64)
    outcome = torch.empty((1, 2), dtype=torch.int64)
    tp_group = _FakeTPGroup(torch.empty((0, 2), dtype=torch.int64))
    tp_group.world_size = 1

    _sync_dflash_sampling_results(
        accept_len,
        bonus,
        tp_group=tp_group,
        outcome_buffer=outcome,
    )

    torch.testing.assert_close(accept_len, torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(bonus, torch.tensor([99], dtype=torch.int64))
    assert tp_group.broadcast_calls == []


def test_sampling_sync_uses_attention_tp_group_with_dp_attention(monkeypatch):
    attn_tp_group = object()
    monkeypatch.setattr(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled", lambda: True
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: type("_Parallel", (), {"attn_tp_group": attn_tp_group})(),
    )

    assert _get_dflash_sampling_tp_group() is attn_tp_group


def test_sampling_sync_uses_model_tp_group_without_dp_attention(monkeypatch):
    tp_group = object()
    monkeypatch.setattr(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled", lambda: False
    )
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_worker_v2.get_tp_group", lambda: tp_group
    )

    assert _get_dflash_sampling_tp_group() is tp_group
