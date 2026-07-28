from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dflash_worker_v2 import (
    _assert_dflash_tp_tensor_consensus,
    _stable_dflash_state_hash,
    _sync_dflash_sampling_results,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTPGroup:
    def __init__(self, canonical, gathered=None):
        self.world_size = 2
        self.canonical = canonical
        self.gathered = gathered
        self.broadcast_calls = []
        self.gather_calls = []

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))
        tensor.copy_(self.canonical)

    def all_gather_into_tensor(self, output, local):
        self.gather_calls.append((output, local))
        output.copy_(self.gathered)


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
    assert tp_group.gather_calls == []


def test_single_rank_is_a_noop():
    accept_len = torch.tensor([2], dtype=torch.int32)
    bonus = torch.tensor([99], dtype=torch.int64)
    tp_group = _FakeTPGroup(torch.empty((0, 2), dtype=torch.int64))
    tp_group.world_size = 1

    _sync_dflash_sampling_results(accept_len, bonus, tp_group=tp_group)

    torch.testing.assert_close(accept_len, torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(bonus, torch.tensor([99], dtype=torch.int64))
    assert tp_group.broadcast_calls == []
    assert tp_group.gather_calls == []


def test_debug_assertion_reports_prebroadcast_divergence():
    accept_len = torch.tensor([3, 1], dtype=torch.int32)
    bonus = torch.tensor([101, 999], dtype=torch.int64)
    canonical = torch.tensor([[3, 101], [1, 202]], dtype=torch.int64)
    gathered = torch.tensor(
        [
            [3, 101],
            [1, 202],
            [3, 101],
            [1, 999],
        ],
        dtype=torch.int64,
    )
    tp_group = _FakeTPGroup(canonical, gathered)

    with pytest.raises(
        RuntimeError,
        match=r"rank=1, row=1, local=\[1, 999\], rank0=\[1, 202\]",
    ):
        _sync_dflash_sampling_results(
            accept_len,
            bonus,
            tp_group=tp_group,
            outcome_buffer=torch.empty((2, 2), dtype=torch.int64),
            local_outcome_buffer=torch.empty((2, 2), dtype=torch.int64),
            gathered_outcome_buffer=torch.empty((4, 2), dtype=torch.int64),
            assert_consensus=True,
        )

    assert len(tp_group.broadcast_calls) == 1
    assert len(tp_group.gather_calls) == 1


def test_debug_assertion_accepts_consensus():
    accept_len = torch.tensor([3, 1], dtype=torch.int32)
    bonus = torch.tensor([101, 202], dtype=torch.int64)
    canonical = torch.tensor([[3, 101], [1, 202]], dtype=torch.int64)
    gathered = canonical.repeat(2, 1)
    tp_group = _FakeTPGroup(canonical, gathered)

    _sync_dflash_sampling_results(
        accept_len,
        bonus,
        tp_group=tp_group,
        assert_consensus=True,
    )

    torch.testing.assert_close(accept_len, canonical[:, 0].to(torch.int32))
    torch.testing.assert_close(bonus, canonical[:, 1])


def test_tensor_consensus_reports_first_divergent_value():
    local = torch.tensor([[10, 20], [30, 40]], dtype=torch.int64)
    gathered = torch.tensor(
        [10, 20, 30, 40, 10, 20, 31, 40],
        dtype=torch.int64,
    )
    tp_group = _FakeTPGroup(canonical=None, gathered=gathered)

    with pytest.raises(
        RuntimeError,
        match=r"candidate-token divergence.*rank=1, flat_index=2, local=31, rank0=30",
    ):
        _assert_dflash_tp_tensor_consensus(
            local,
            label="candidate-token",
            tp_group=tp_group,
        )


class _GrammarState:
    def __init__(self):
        self._finished = False
        self.current_token = 7
        self.accepted_tokens = [5, 6, 7]
        self.tokens_in_think = 3
        self.tokens_after_end = -1
        self.think_end_match_len = 1
        self._match_len_history = [0, 0, 1]
        self.grammar = None


def _request_state():
    return SimpleNamespace(
        rid="req-1",
        origin_input_ids=array("q", range(32)),
        output_ids=array("q", [101, 102]),
        grammar=_GrammarState(),
    )


def test_request_state_hash_tracks_output_and_grammar_without_full_prompt_scan():
    left = _request_state()
    right = _request_state()
    assert _stable_dflash_state_hash(left) == _stable_dflash_state_hash(right)

    right.output_ids.append(103)
    assert _stable_dflash_state_hash(left)[0] != _stable_dflash_state_hash(right)[0]

    right = _request_state()
    right.grammar.think_end_match_len = 2
    assert _stable_dflash_state_hash(left)[1] != _stable_dflash_state_hash(right)[1]
