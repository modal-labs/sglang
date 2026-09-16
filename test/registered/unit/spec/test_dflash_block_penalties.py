import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
    apply_scaling_penalties,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.speculative.dflash_utils import (
    DFlashBlockPenaltyState,
    apply_dflash_verify_logits_adjustments,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(
    *,
    repetition_penalty=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    min_new_tokens=0,
):
    return SimpleNamespace(
        sampling_params=SamplingParams(
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            min_new_tokens=min_new_tokens,
        ),
        tokenizer=SimpleNamespace(
            additional_stop_token_ids=None,
            eos_token_id=2,
        ),
        penalty_cumulated_len=0,
    )


def _make_batch(reqs):
    class FakeBatch:
        pass

    batch = FakeBatch()
    batch.reqs = reqs
    batch.device = torch.device("cpu")
    return batch


def _make_orchestrator(reqs, vocab_size=16):
    return BatchedPenalizerOrchestrator(
        vocab_size,
        _make_batch(reqs),
        {
            BatchedFrequencyPenalizer,
            BatchedMinNewTokensPenalizer,
            BatchedPresencePenalizer,
            BatchedRepetitionPenalizer,
        },
    )


def _make_sampling_info(batch_size, vocab_size, **overrides):
    values = dict(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab_size,
        device="cpu",
        penalizer_orchestrator=None,
    )
    values.update(overrides)
    return SamplingBatchInfo(**values)


def _assert_logits_equal(actual, expected):
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    finite = ~torch.isinf(expected)
    torch.testing.assert_close(actual[finite], expected[finite])


def _make_reference(orchestrator, logits, candidates):
    reference = torch.empty_like(logits)
    for position in range(candidates.shape[1]):
        row = logits[:, position].clone()
        orchestrator.accumulate_additive_penalties(row)
        scaling = orchestrator.accumulate_scaling_penalties()
        if scaling is not None:
            apply_scaling_penalties(row, scaling)
        reference[:, position] = row
        if position + 1 < candidates.shape[1]:
            orchestrator.cumulate_output_tokens(candidates[:, position + 1])
    return reference


def _make_block_case():
    reqs = [
        _make_req(
            repetition_penalty=1.5,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            min_new_tokens=3,
        ),
        _make_req(
            repetition_penalty=1.5,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            min_new_tokens=6,
        ),
    ]
    committed = torch.tensor(
        [
            [5, 7],
            [4, 8],
        ],
        dtype=torch.int64,
    )
    candidates = torch.tensor(
        [
            [7, 5, 9, 5],
            [8, 1, 10, 1],
        ],
        dtype=torch.int64,
    )
    return reqs, committed, candidates


def _prepare_orchestrator(reqs, committed):
    orchestrator = _make_orchestrator(reqs)
    for token_column in committed.T:
        orchestrator.cumulate_output_tokens(token_column)
    return orchestrator


def test_block_matches_token_by_token_all_penalties():
    torch.manual_seed(0)
    bs, k, vocab_size = 2, 4, 16
    reqs, committed, candidates = _make_block_case()
    logits = torch.randn(bs, k, vocab_size, dtype=torch.float32)

    overlap_orchestrator = _prepare_orchestrator(reqs, committed)
    state = DFlashBlockPenaltyState.from_orchestrator(overlap_orchestrator)
    additive = torch.zeros(bs, vocab_size)
    overlap_orchestrator.accumulate_additive_penalties(additive)
    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        acc_additive_penalties=additive,
        acc_scaling_penalties=overlap_orchestrator.accumulate_scaling_penalties(),
        dflash_block_penalty_state=state,
    )
    overlap_logits = logits.clone().view(bs * k, vocab_size)
    apply_dflash_verify_logits_adjustments(
        next_token_logits=overlap_logits,
        sampling_info=sampling_info,
        draft_token_num=k,
        candidates=candidates,
    )
    overlap_reference = _make_reference(
        _prepare_orchestrator(reqs, committed),
        logits,
        candidates,
    )
    _assert_logits_equal(overlap_logits.view(bs, k, vocab_size), overlap_reference)

    non_overlap_orchestrator = _prepare_orchestrator(reqs, committed)
    non_overlap_info = _make_sampling_info(
        bs,
        vocab_size,
        penalizer_orchestrator=non_overlap_orchestrator,
    )
    non_overlap_logits = logits.clone().view(bs * k, vocab_size)
    apply_dflash_verify_logits_adjustments(
        next_token_logits=non_overlap_logits,
        sampling_info=non_overlap_info,
        draft_token_num=k,
        candidates=candidates,
    )
    _assert_logits_equal(non_overlap_logits.view(bs, k, vocab_size), overlap_reference)


def test_position_zero_matches_broadcast_path():
    torch.manual_seed(1)
    bs, k, vocab_size = 2, 4, 16
    reqs, committed, candidates = _make_block_case()
    orchestrator = _prepare_orchestrator(reqs, committed)
    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        penalizer_orchestrator=orchestrator,
    )
    logits = torch.randn(bs, k, vocab_size)
    broadcast = logits.clone().view(bs * k, vocab_size)
    apply_dflash_verify_logits_adjustments(
        next_token_logits=broadcast,
        sampling_info=sampling_info,
        draft_token_num=k,
    )
    block = logits.clone().view(bs * k, vocab_size)
    apply_dflash_verify_logits_adjustments(
        next_token_logits=block,
        sampling_info=sampling_info,
        draft_token_num=k,
        candidates=candidates,
    )
    _assert_logits_equal(
        block.view(bs, k, vocab_size)[:, 0], broadcast.view(bs, k, vocab_size)[:, 0]
    )
    # The broadcast (pre-fix) path ignores the preceding candidates, so it must
    # disagree with the token-by-token reference at every later position.
    reference = _make_reference(
        _prepare_orchestrator(reqs, committed), logits, candidates
    )
    for position in range(1, k):
        assert not torch.equal(
            broadcast.view(bs, k, vocab_size)[:, position], reference[:, position]
        )


def test_no_penalty_path_is_byte_identical():
    bs, k, vocab_size = 2, 4, 16
    vocab_mask = torch.ones(bs, vocab_size, dtype=torch.bool)
    vocab_mask[0, 3] = False
    vocab_mask[1, 7] = False
    logit_bias = torch.randn(bs, vocab_size)

    def apply_mask_func(*, logits, vocab_mask):
        logits.masked_fill_(~vocab_mask, float("-inf"))

    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        vocab_mask=vocab_mask,
        apply_mask_func=apply_mask_func,
        logit_bias=logit_bias,
    )
    candidates = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64)
    logits = torch.randn(bs * k, vocab_size)
    with_candidates = logits.clone()
    without_candidates = logits.clone()
    apply_dflash_verify_logits_adjustments(
        next_token_logits=with_candidates,
        sampling_info=sampling_info,
        draft_token_num=k,
        candidates=candidates,
    )
    apply_dflash_verify_logits_adjustments(
        next_token_logits=without_candidates,
        sampling_info=sampling_info,
        draft_token_num=k,
    )
    assert torch.equal(with_candidates, without_candidates)

    no_penalty_orchestrator = _make_orchestrator(
        [_make_req(), _make_req()],
        vocab_size,
    )
    assert DFlashBlockPenaltyState.from_orchestrator(no_penalty_orchestrator) is None


def test_block_path_still_applies_vocab_mask_and_logit_bias():
    torch.manual_seed(2)
    bs, k, vocab_size = 2, 4, 16
    reqs, committed, candidates = _make_block_case()
    vocab_mask = torch.ones(bs, vocab_size, dtype=torch.bool)
    vocab_mask[0, 3] = False
    vocab_mask[1, 11] = False
    logit_bias = torch.randn(bs, vocab_size)

    def apply_mask_func(*, logits, vocab_mask):
        logits.masked_fill_(~vocab_mask, float("-inf"))

    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        penalizer_orchestrator=_prepare_orchestrator(reqs, committed),
        vocab_mask=vocab_mask,
        apply_mask_func=apply_mask_func,
        logit_bias=logit_bias,
    )
    logits = torch.randn(bs, k, vocab_size)
    block = logits.clone().view(bs * k, vocab_size)
    apply_dflash_verify_logits_adjustments(
        next_token_logits=block,
        sampling_info=sampling_info,
        draft_token_num=k,
        candidates=candidates,
    )
    reference = _make_reference(
        _prepare_orchestrator(reqs, committed), logits, candidates
    )
    reference.add_(logit_bias[:, None, :])
    reference.masked_fill_(~vocab_mask[:, None, :], float("-inf"))
    _assert_logits_equal(block.view(bs, k, vocab_size), reference)


def test_forward_copy_snapshot_is_isolated():
    reqs, committed, _ = _make_block_case()
    orchestrator = _prepare_orchestrator(reqs, committed)
    state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)
    presence = state.cumulated_presence.clone()
    scaling = state.scaling_base.clone()
    lengths = state.len_output_tokens.clone()

    orchestrator.cumulate_output_tokens(torch.tensor([9, 10], dtype=torch.int64))

    assert torch.equal(state.cumulated_presence, presence)
    assert torch.equal(state.scaling_base, scaling)
    assert torch.equal(state.len_output_tokens, lengths)


def test_candidates_shape_mismatch_raises():
    reqs, committed, candidates = _make_block_case()
    orchestrator = _prepare_orchestrator(reqs, committed)
    sampling_info = _make_sampling_info(
        len(reqs),
        16,
        dflash_block_penalty_state=DFlashBlockPenaltyState.from_orchestrator(
            orchestrator
        ),
    )
    logits = torch.randn(2 * 4, 16)
    with pytest.raises(ValueError, match="candidates shape mismatch"):
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=4,
            candidates=candidates[:, :-1],
        )


def test_penalty_state_rows_mismatch_raises():
    reqs, committed, candidates = _make_block_case()
    orchestrator = _prepare_orchestrator(reqs, committed)
    state = DFlashBlockPenaltyState.from_orchestrator(orchestrator)
    sampling_info = _make_sampling_info(
        len(reqs),
        16,
        dflash_block_penalty_state=dataclasses.replace(
            state,
            additive_base=state.additive_base[:1],
        ),
    )
    with pytest.raises(ValueError, match="penalty state rows mismatch"):
        apply_dflash_verify_logits_adjustments(
            next_token_logits=torch.randn(2 * 4, 16),
            sampling_info=sampling_info,
            draft_token_num=4,
            candidates=candidates,
        )
