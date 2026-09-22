"""DFlash sampling accept with a non-finite target row: the row must not
reach the reject sampler as NaN, the request must get exactly one
``vocab_size - 1`` sentinel at that row (drafts before it stay accepted), the
per-request degenerate mask must flag it, and a *legitimately* sampled
``vocab_size - 1`` must leave the mask clear."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_sampling,
    accept_sampling_triton,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    reject_sampling_sentinel_token,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

VOCAB_SIZE = 163840
SENTINEL = reject_sampling_sentinel_token(VOCAB_SIZE)
BLOCK = 8
GAMMA = BLOCK - 1
PEAK = 50.0
BONUS_TOKEN = 4242
ACCEPT_FNS = {"torch": accept_sampling, "triton": accept_sampling_triton}


def _peaked_inputs(bs: int, device: torch.device):
    """Target logits peaked on each draft candidate (p ~ 1) and, at the bonus
    row, on BONUS_TOKEN; draft q one-hot on the candidates."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    candidates = torch.randint(
        1, VOCAB_SIZE - 1, (bs, BLOCK), generator=gen, dtype=torch.int64
    ).to(device)
    target_logits = torch.zeros((bs * BLOCK, VOCAB_SIZE), dtype=torch.float32)
    rows = target_logits.view(bs, BLOCK, VOCAB_SIZE)
    for b in range(bs):
        for r in range(GAMMA):
            rows[b, r, int(candidates[b, r + 1])] = PEAK
        rows[b, GAMMA, BONUS_TOKEN] = PEAK
    target_logits = target_logits.to(device)
    draft_probs = torch.zeros(
        (bs, GAMMA, VOCAB_SIZE), dtype=torch.float32, device=device
    )
    draft_probs.scatter_(-1, candidates[:, 1:].unsqueeze(-1), 1.0)
    sampling_info = SimpleNamespace(
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        temperatures=torch.ones((bs, 1), dtype=torch.float32, device=device),
    )
    draft_input = SimpleNamespace(max_top_k=None, uniform_top_k_value=None)
    return candidates, target_logits, draft_probs, sampling_info, draft_input


def _run(accept_fn, candidates, target_logits, draft_probs, sampling_info, draft_input):
    return accept_fn(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=GAMMA,
        verify_num_draft_tokens=BLOCK,
        cutoff_verify_lens=None,
    )


@pytest.mark.parametrize("impl", sorted(ACCEPT_FNS))
def test_clean_rows_accept_everything(impl):
    device = torch.device("cuda")
    args = _peaked_inputs(bs=3, device=device)
    correct_len, bonus, cap_trim, degenerate = _run(ACCEPT_FNS[impl], *args)
    assert correct_len.tolist() == [GAMMA] * 3
    assert bonus.tolist() == [BONUS_TOKEN] * 3
    assert cap_trim.tolist() == [0] * 3
    assert degenerate.dtype == torch.bool
    assert degenerate.tolist() == [False] * 3


@pytest.mark.parametrize("impl", sorted(ACCEPT_FNS))
@pytest.mark.parametrize("bad_row", [0, 3, GAMMA])
@pytest.mark.parametrize("kind", ["nan", "inf", "partial_nan"])
def test_degenerate_row_yields_one_sentinel(impl, bad_row, kind):
    device = torch.device("cuda")
    candidates, target_logits, draft_probs, sampling_info, draft_input = _peaked_inputs(
        bs=3, device=device
    )
    rows = target_logits.view(3, BLOCK, VOCAB_SIZE)
    if kind == "nan":
        rows[1, bad_row] = float("nan")
    elif kind == "inf":
        rows[1, bad_row] = float("inf")
    else:
        rows[1, bad_row, 17] = float("nan")

    correct_len, bonus, _, degenerate = _run(
        ACCEPT_FNS[impl],
        candidates,
        target_logits,
        draft_probs,
        sampling_info,
        draft_input,
    )

    assert degenerate.tolist() == [False, True, False]
    # Drafts before the bad row are still accepted; the bad row itself
    # produces the sentinel and nothing after it.
    assert correct_len.tolist() == [GAMMA, bad_row, GAMMA]
    assert bonus.tolist() == [BONUS_TOKEN, SENTINEL, BONUS_TOKEN]


@pytest.mark.parametrize("impl", sorted(ACCEPT_FNS))
def test_legit_sentinel_sample_is_not_degenerate(impl):
    device = torch.device("cuda")
    candidates, target_logits, draft_probs, sampling_info, draft_input = _peaked_inputs(
        bs=2, device=device
    )
    rows = target_logits.view(2, BLOCK, VOCAB_SIZE)
    rows[0, GAMMA, BONUS_TOKEN] = 0.0
    rows[0, GAMMA, SENTINEL] = PEAK

    correct_len, bonus, _, degenerate = _run(
        ACCEPT_FNS[impl],
        candidates,
        target_logits,
        draft_probs,
        sampling_info,
        draft_input,
    )

    assert correct_len.tolist() == [GAMMA, GAMMA]
    assert bonus.tolist() == [SENTINEL, BONUS_TOKEN]
    assert degenerate.tolist() == [False, False]


@pytest.mark.parametrize("impl", sorted(ACCEPT_FNS))
def test_sampler_never_sees_non_finite_target(impl, monkeypatch):
    import sglang.kernels.ops.speculative.dspark.dspark_accept as mod

    seen = {}
    real = mod.chain_speculative_sampling_triton

    def spy(**kwargs):
        seen["finite"] = bool(torch.isfinite(kwargs["target_probs"]).all().item())
        return real(**kwargs)

    monkeypatch.setattr(mod, "chain_speculative_sampling_triton", spy)
    device = torch.device("cuda")
    candidates, target_logits, draft_probs, sampling_info, draft_input = _peaked_inputs(
        bs=2, device=device
    )
    target_logits.view(2, BLOCK, VOCAB_SIZE)[1, 2] = float("nan")
    _run(
        ACCEPT_FNS[impl],
        candidates,
        target_logits,
        draft_probs,
        sampling_info,
        draft_input,
    )
    assert seen["finite"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
