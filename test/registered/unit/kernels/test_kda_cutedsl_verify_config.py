import pytest

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import use_cutedsl_bf16_gemm
from sglang.kernels.ops.kimi_k3.kda_decode_mtp import _p2_lanes_k
from sglang.srt.layers.attention.linear.kda_backend import (
    _split_cutedsl_mtp_value_tiles,
    _supports_cutedsl_mtp_width,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-test-cpu")


@pytest.mark.parametrize("width", range(2, 9))
def test_standard_verify_widths_remain_supported(width):
    assert _supports_cutedsl_mtp_width(width)


@pytest.mark.parametrize("width", [1, 9, 15, 17])
def test_unoptimized_verify_widths_fall_back(width):
    assert not _supports_cutedsl_mtp_width(width)


def test_block16_uses_split_value_specialization_only_at_batch_one():
    assert _supports_cutedsl_mtp_width(16)
    assert _split_cutedsl_mtp_value_tiles(draft_token_num=16, batch_size=1)
    assert not _split_cutedsl_mtp_value_tiles(draft_token_num=16, batch_size=2)
    assert not _split_cutedsl_mtp_value_tiles(draft_token_num=8, batch_size=1)


def test_block16_uses_wider_recurrence_butterfly():
    assert _p2_lanes_k(N=1, num_spec=15) == 16
    assert _p2_lanes_k(N=1, num_spec=7) == 8
    assert _p2_lanes_k(N=2, num_spec=15) == 8


def test_k3_tgv_fast_path_covers_block8_and_block16():
    shape = (6144, 7168)
    assert use_cutedsl_bf16_gemm(8, *shape)
    assert use_cutedsl_bf16_gemm(16, *shape)
    assert not use_cutedsl_bf16_gemm(17, *shape)
