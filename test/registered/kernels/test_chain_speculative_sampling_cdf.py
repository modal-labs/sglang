"""Final chain sampling must stay within finite positive probability support."""

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

VOCAB_SIZE = 163840
NUM_SLOTS = 8
# Exact float32 bits from normalized finite p/q rows in minimal-fixtures.pt.
# SHA256: 373bb1883128c3d39ce41ffc2bf95d3ac45a531b4ba94b728d566fbaa163a0eb
# Both stored target rows are bit-identical. Preserve their values without any
# normalization: the regression is reduction-vs-prefix-scan rounding.
FIXTURES = {
    "accepted": {
        "p": [
            1025976679,
            1054508789,
            1026433692,
            1016248278,
            0,
            1032908235,
            1030618853,
            1020671206,
            1021438819,
            0,
            1024878477,
            1045231174,
            0,
            0,
            1029109054,
            0,
        ],
        "q": [
            1025553404,
            1053955451,
            1025992759,
            1015876769,
            1015667769,
            1032541256,
            1030016218,
            1020128807,
            1020866762,
            1007506979,
            1024497634,
            1044712185,
            1007778145,
            999095931,
            1028564754,
            0,
        ],
        "recorded_proposal": 3,
    },
    "rejected": {
        "p": [
            1043277081,
            0,
            0,
            1060456737,
            0,
            1015906023,
            0,
            0,
            1035398411,
            1013514304,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "q": [
            1042841720,
            982619329,
            994949983,
            1060006113,
            992096530,
            1015554306,
            989800112,
            993697664,
            1034943710,
            1012935133,
            1002704410,
            985631390,
            992883196,
            1012596660,
            994097359,
            0,
        ],
        "recorded_proposal": 15,
    },
}


def _support_indices(layout):
    if layout == "first16":
        return torch.arange(16, device="cuda")
    if layout == "last16":
        return torch.arange(VOCAB_SIZE - 16, VOCAB_SIZE, device="cuda")
    if layout == "last_block_start":
        return torch.arange(VOCAB_SIZE - 4096, VOCAB_SIZE - 4096 + 16, device="cuda")
    return torch.arange(16, device="cuda") * 8192 + 17


def _chain_inputs(p, q, proposal, final_uniform):
    return dict(
        predicts=torch.full((NUM_SLOTS,), -1, device="cuda", dtype=torch.int32),
        accept_index=torch.full((1, NUM_SLOTS), -1, device="cuda", dtype=torch.int64),
        accept_token_num=torch.full((1,), -1, device="cuda", dtype=torch.int32),
        candidates=torch.full(
            (1, NUM_SLOTS), proposal, device="cuda", dtype=torch.int64
        ),
        retrive_index=torch.arange(NUM_SLOTS, device="cuda", dtype=torch.int64)[None],
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.full((1, NUM_SLOTS - 1), 0.5, device="cuda"),
        uniform_samples_for_final_sampling=final_uniform,
        target_probs=p[None, None].repeat(1, NUM_SLOTS, 1),
        draft_probs=q[None, None].repeat(1, NUM_SLOTS - 1, 1),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )


def _run_chain(kwargs, use_graph):
    if use_graph:
        chain_speculative_sampling_triton(**kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain_speculative_sampling_triton(**kwargs)
        for _ in range(2):
            kwargs["predicts"].fill_(-1)
            kwargs["accept_token_num"].fill_(-1)
            graph.replay()
    else:
        chain_speculative_sampling_triton(**kwargs)
    torch.cuda.synchronize()


@pytest.mark.parametrize("case", ["accepted", "rejected"])
@pytest.mark.parametrize(
    "layout", ["first16", "last16", "last_block_start", "scattered"]
)
@pytest.mark.parametrize(
    "uniform_bits",
    [0, 0x3F000000, 0x3F7FFFFF, 0x3F7FFFFE],
    ids=["zero", "half", "nextafter-one", "twice-nextafter-one"],
)
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_final_cdf_stays_in_positive_support(case, layout, uniform_bits, use_graph):
    fixture = FIXTURES[case]
    p = torch.zeros(VOCAB_SIZE, device="cuda")
    q = torch.zeros_like(p)
    indices = _support_indices(layout)
    p[indices] = torch.tensor(fixture["p"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    q[indices] = torch.tensor(fixture["q"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    # The recorded rejected proposal was offset 15 (p=q=0). Use a genuine
    # positive-q, zero-p proposal instead to exercise first-step rejection with
    # a valid draft token while retaining every original probability bit.
    proposal_offset = fixture["recorded_proposal"] if case == "accepted" else 1
    proposal = int(indices[proposal_offset].item())
    final_uniform = torch.tensor([uniform_bits], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    kwargs = _chain_inputs(p, q, proposal, final_uniform)
    _run_chain(kwargs, use_graph)

    expected_accept = NUM_SLOTS - 1 if case == "accepted" else 0
    assert kwargs["accept_token_num"].item() == expected_accept
    bonus = int(kwargs["predicts"][expected_accept].item())
    assert 0 <= bonus < VOCAB_SIZE
    sampled_mass = p if case == "accepted" else (p - q).clamp_min(0)
    assert torch.isfinite(sampled_mass).all().item()
    assert sampled_mass.sum().item() > 0
    assert sampled_mass[bonus].item() > 0, (case, layout, uniform_bits, bonus)
    assert torch.equal(
        kwargs["accept_index"][0, : expected_accept + 1],
        torch.arange(expected_accept + 1, device="cuda"),
    )


@pytest.mark.parametrize("case", ["accepted", "rejected"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_final_cdf_partial_last_block(case, use_graph):
    vocab_size = 4097
    fixture = FIXTURES[case]
    p = torch.zeros(vocab_size, device="cuda")
    q = torch.zeros_like(p)
    p[-16:] = torch.tensor(fixture["p"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    q[-16:] = torch.tensor(fixture["q"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    offset = fixture["recorded_proposal"] if case == "accepted" else 1
    uniform = torch.tensor([0x3F7FFFFF], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    kwargs = _chain_inputs(p, q, vocab_size - 16 + offset, uniform)
    _run_chain(kwargs, use_graph)
    accepted = NUM_SLOTS - 1 if case == "accepted" else 0
    assert kwargs["accept_token_num"].item() == accepted
    bonus = int(kwargs["predicts"][accepted].item())
    assert 0 <= bonus < vocab_size
    mass = p if case == "accepted" else (p - q).clamp_min(0)
    assert mass[bonus].item() > 0


@pytest.mark.parametrize(
    "invalid",
    [
        "zero_residual",
        "nan_mass",
        "inf_mass",
        "negative_total",
        "uniform_one",
        "nan_uniform",
    ],
)
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_final_cdf_no_match_invalid_inputs_keep_sentinel(invalid, use_graph):
    # These fixtures deliberately have no ordinary CDF match. This verifies the
    # rare roundoff fallback does not manufacture a supported token for invalid
    # mass/uniforms; it does not claim the kernel validates arbitrary bad inputs.
    p = torch.zeros(VOCAB_SIZE, device="cuda")
    p[17] = 1.0
    q = p.clone()
    uniform = torch.tensor([0.5], device="cuda")
    proposal = 0 if invalid == "zero_residual" else 17
    kwargs = _chain_inputs(p, q, proposal, uniform)
    final_row = kwargs["target_probs"][0, -1]
    if invalid == "nan_mass":
        final_row[0] = float("nan")
    elif invalid == "inf_mass":
        final_row[0] = float("inf")
    elif invalid == "negative_total":
        final_row[0] = -0.2
        final_row[17] = 0.1
    elif invalid == "uniform_one":
        uniform.fill_(1.0)
    elif invalid == "nan_uniform":
        uniform.fill_(float("nan"))
    _run_chain(kwargs, use_graph)

    accepted = 0 if invalid == "zero_residual" else NUM_SLOTS - 1
    assert kwargs["accept_token_num"].item() == accepted
    assert kwargs["predicts"][accepted].item() == VOCAB_SIZE - 1


@pytest.mark.parametrize("invalid", ["nan_q", "negative_p", "inf_q"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_final_cdf_fallback_checks_invalid_earlier_blocks(invalid, use_graph):
    # This saved residual has a positive finite mass but a final-CDF roundoff
    # gap at max-uniform. Invalid target p must disable fallback, while
    # non-probability q follows the same zero clamp as both residual passes.
    fixture = FIXTURES["rejected"]
    p = torch.zeros(VOCAB_SIZE, device="cuda")
    q = torch.zeros_like(p)
    p[-16:] = torch.tensor(fixture["p"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    q[-16:] = torch.tensor(fixture["q"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    if invalid == "nan_q":
        q[7] = float("nan")
    elif invalid == "negative_p":
        p[7] = -1.0
    else:
        q[7] = float("inf")
    uniform = torch.tensor([0x3F7FFFFF], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    kwargs = _chain_inputs(p, q, VOCAB_SIZE - 16 + 1, uniform)
    _run_chain(kwargs, use_graph)

    assert kwargs["accept_token_num"].item() == 0
    expected = VOCAB_SIZE - 1 if invalid == "negative_p" else VOCAB_SIZE - 7
    assert kwargs["predicts"][0].item() == expected


@pytest.mark.parametrize("invalid_q", [1.5, float("nan")], ids=["q>1", "q=NaN"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_final_cdf_fallback_uses_guarded_residual(invalid_q, use_graph):
    fixture = FIXTURES["rejected"]
    p = torch.zeros(VOCAB_SIZE, device="cuda")
    q = torch.zeros_like(p)
    p[-16:] = torch.tensor(fixture["p"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    q[-16:] = torch.tensor(fixture["q"], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    # Preserve the saved endpoint-gap residual exactly. The last positive
    # lane now obtains its mass from the q-range guard rather than subtraction.
    last_positive = VOCAB_SIZE - 7
    p[last_positive] -= q[last_positive]
    q[last_positive] = invalid_q
    guarded_q = torch.where((q >= 0) & (q <= 1), q, 0)
    residual = (p - guarded_q).clamp_min(0)
    expected = int(torch.nonzero(residual > 0)[-1].item())
    uniform = torch.tensor([0x3F7FFFFF], dtype=torch.int32, device="cuda").view(
        torch.float32
    )
    kwargs = _chain_inputs(p, q, VOCAB_SIZE - 15, uniform)
    _run_chain(kwargs, use_graph)

    assert kwargs["accept_token_num"].item() == 0
    assert expected == last_positive
    assert kwargs["predicts"][0].item() == expected


@pytest.mark.parametrize(
    "q_kind",
    ["zero_q", "neg_inf_q", "pos_inf_q", "nan_q"],
    ids=["q=0", "q=-inf", "q=+inf", "q=NaN"],
)
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_acceptance_requires_probability_q(q_kind, use_graph):
    # Upstream #37134 acceptance half: a draft token must carry positive q mass
    # before coin * q < p can accept it. Before the guard, q = 0 and q = -inf
    # made the comparison true for any positive target mass (0 < p, -inf < p),
    # so the draft was accepted unconditionally; q = NaN / +inf only happened
    # to reject because the comparison itself went non-finite. After a
    # rejection the final pass resamples from the target row, whose only
    # positive-mass lane here is token 17.
    p = torch.zeros(VOCAB_SIZE, device="cuda")
    q = torch.zeros_like(p)
    p[17] = 0.5
    if q_kind == "zero_q":
        q[17] = 0.0
    elif q_kind == "neg_inf_q":
        q[17] = float("-inf")
    elif q_kind == "pos_inf_q":
        q[17] = float("inf")
    else:
        q[17] = float("nan")
    uniform = torch.tensor([0.5], device="cuda")
    kwargs = _chain_inputs(p, q, 17, uniform)
    _run_chain(kwargs, use_graph)

    assert kwargs["accept_token_num"].item() == 0
    bonus = int(kwargs["predicts"][0].item())
    assert bonus == 17


@pytest.mark.parametrize(
    "q_kind",
    ["neg_inf_q", "over_one_q"],
    ids=["q=-inf", "q=1.5"],
)
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_residual_non_probability_q_resamples_target(q_kind, use_graph):
    # Upstream #37134 residual half: after a genuine first-step rejection
    # (valid q = 1 at the proposal, zero target mass there), a non-probability
    # q on another lane used to corrupt the residual -- +inf mass for q = -inf
    # (norm_sum = inf, no CDF lane, fallback disabled) and clipped-to-zero mass
    # for q = 1.5 (norm_sum = 0, no lane, fallback skipped) -- so the final pass
    # emitted the VOCAB_SIZE - 1 sentinel. Clamped to 0, the residual keeps the
    # row's genuine 0.5 target mass and the sample lands on support.
    p = torch.zeros(VOCAB_SIZE, device="cuda")
    q = torch.zeros_like(p)
    q[5] = 1.0  # proposal: valid draft mass, zero target mass -> reject
    p[19] = 0.5
    if q_kind == "neg_inf_q":
        q[19] = float("-inf")
    else:
        q[19] = 1.5
    uniform = torch.tensor([0.5], device="cuda")
    kwargs = _chain_inputs(p, q, 5, uniform)
    _run_chain(kwargs, use_graph)

    assert kwargs["accept_token_num"].item() == 0
    bonus = int(kwargs["predicts"][0].item())
    assert bonus == 19


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
