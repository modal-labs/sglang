import torch

from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_sampling_info(batch_size, vocab_size, **overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return SamplingBatchInfo(**defaults)


def _make_logits(batch_size, vocab_size):
    return torch.tensor(
        [
            [2.0, -3.0, 0.5, -1.5, 4.0, -2.5, 1.0, 0.0],
            [-2.0, 3.5, -0.75, 1.25, -4.0, 2.5, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )[:batch_size, :vocab_size]


def _make_batch(reqs):
    class FakeBatch:
        pass

    batch = FakeBatch()
    batch.reqs = reqs
    batch.device = "cpu"
    return batch


def _make_req(*, repetition_penalty=1.0, frequency_penalty=0.0):
    class FakeRequest:
        pass

    req = FakeRequest()
    req.sampling_params = SamplingParams(
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
    )
    return req


def test_forward_copy_penalties_match_sampling_batch_info():
    bs, draft_token_num, vocab_size = 2, 3, 8
    logits_2d = _make_logits(bs, vocab_size)
    additive = torch.zeros(bs, vocab_size)
    additive[0, 1] = -0.5
    additive[1, 4] = -0.75
    scaling = torch.ones(bs, vocab_size)
    scaling[:, 0] = 1.5
    logit_bias = torch.zeros(bs, vocab_size)
    logit_bias[:, 3] = 0.25
    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        acc_additive_penalties=additive,
        acc_scaling_penalties=scaling,
        logit_bias=logit_bias,
    )
    ref = logits_2d.clone()
    sampling_info.apply_logits_bias(ref)

    logits = logits_2d.repeat_interleave(draft_token_num, dim=0).clone()
    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    )

    expected = ref[:, None, :].expand(bs, draft_token_num, vocab_size)
    assert torch.allclose(
        logits.view(bs, draft_token_num, vocab_size),
        expected,
    )


def test_attached_penalizer_matches_sampling_batch_info():
    bs, draft_token_num, vocab_size = 2, 3, 8
    reqs = [
        _make_req(repetition_penalty=1.5, frequency_penalty=0.3),
        _make_req(repetition_penalty=1.5, frequency_penalty=0.3),
    ]
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size,
        _make_batch(reqs),
        {BatchedRepetitionPenalizer, BatchedFrequencyPenalizer},
    )
    orchestrator.cumulate_output_tokens(torch.tensor([7, 7]))
    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        penalizer_orchestrator=orchestrator,
    )
    logits_2d = _make_logits(bs, vocab_size)
    additive = torch.zeros(bs, vocab_size)
    orchestrator.accumulate_additive_penalties(additive)
    scaling = orchestrator.accumulate_scaling_penalties()
    reference_info = _make_sampling_info(
        bs,
        vocab_size,
        acc_additive_penalties=additive,
        acc_scaling_penalties=scaling,
    )
    ref = logits_2d.clone()
    reference_info.apply_logits_bias(ref)

    logits = logits_2d.repeat_interleave(draft_token_num, dim=0).clone()
    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    )

    expected = ref[:, None, :].expand(bs, draft_token_num, vocab_size)
    assert torch.allclose(
        logits.view(bs, draft_token_num, vocab_size),
        expected,
    )


def test_no_penalties_leave_verify_logits_unchanged():
    bs, draft_token_num, vocab_size = 2, 3, 8
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size,
        _make_batch([_make_req(repetition_penalty=1.0) for _ in range(bs)]),
        {BatchedRepetitionPenalizer, BatchedFrequencyPenalizer},
    )
    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        penalizer_orchestrator=orchestrator,
    )
    logits = _make_logits(bs, vocab_size).repeat_interleave(draft_token_num, dim=0)
    original = logits.clone()

    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    )

    assert torch.equal(logits, original)


def test_vocab_mask_and_logit_bias_match_sampling_batch_info():
    bs, draft_token_num, vocab_size = 2, 3, 8
    vocab_mask = torch.ones(bs, vocab_size, dtype=torch.bool)
    vocab_mask[0, 2] = False
    vocab_mask[1, 5] = False
    logit_bias = torch.zeros(bs, vocab_size)
    logit_bias[:, 6] = 0.75

    def apply_mask_func(*, logits, vocab_mask):
        logits.masked_fill_(~vocab_mask, float("-inf"))

    sampling_info = _make_sampling_info(
        bs,
        vocab_size,
        vocab_mask=vocab_mask,
        apply_mask_func=apply_mask_func,
        logit_bias=logit_bias,
    )
    logits_2d = _make_logits(bs, vocab_size)
    ref = logits_2d.clone()
    sampling_info.apply_logits_bias(ref)

    logits = logits_2d.repeat_interleave(draft_token_num, dim=0).clone()
    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    )

    expected = ref[:, None, :].expand(bs, draft_token_num, vocab_size)
    assert torch.allclose(
        logits.view(bs, draft_token_num, vocab_size),
        expected,
    )


def test_dspark_fold_gate_sees_forward_copy_penalties():
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        verify_logits_adjustments_are_noop,
    )

    bs, vocab_size = 2, 8
    assert verify_logits_adjustments_are_noop(_make_sampling_info(bs, vocab_size))
    assert not verify_logits_adjustments_are_noop(
        _make_sampling_info(
            bs, vocab_size, acc_additive_penalties=torch.zeros(bs, vocab_size)
        )
    )
    assert not verify_logits_adjustments_are_noop(
        _make_sampling_info(
            bs, vocab_size, acc_scaling_penalties=torch.ones(bs, vocab_size)
        )
    )
