"""CPU regression coverage for distinct target-DCP and draft-TP slot domains."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.speculative.dflash_kv_layout import (
    draft_kv_bytes_per_target_token,
    draft_kv_transfer_start,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("tp_size", [1, 2, 8, 16])
@pytest.mark.parametrize("dtype,element_size", [("bf16", 2), ("fp8_e4m3", 1)])
def test_draft_budget_matches_physical_pool(dcp_size, tp_size, dtype, element_size):
    physical_target_tokens = 4096
    draft_tokens = physical_target_tokens * dcp_size
    expected = draft_tokens * 2 * 4 * max(1, 8 // tp_size) * 128 * element_size
    got = draft_kv_bytes_per_target_token(
        num_layers=4,
        total_kv_heads=8,
        head_dim=128,
        tp_size=tp_size,
        dcp_size=dcp_size,
        dtype=dtype,
    )
    assert got * physical_target_tokens == expected


@pytest.mark.parametrize("seq_len", [0, 1, 63, 64, 65, 4095, 4096, 4097, 20003])
@pytest.mark.parametrize("window", [None, 4096])
def test_draft_tail_covers_attention_window(seq_len, window):
    start = draft_kv_transfer_start(seq_len, window, 64)
    required_start = max(0, seq_len - window) if window else 0
    assert start % 64 == 0
    assert start <= required_start < start + 64


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_draft_padding_covers_final_logical_page(dcp_size):
    # Allocator widens both capacity and the reserved first page. The draft
    # shares these logical locations, even though its physical page is only64.
    usable_tokens, page_size = 128 * dcp_size, 64
    pool = MHATokenToKVPool(
        size=usable_tokens,
        page_size=page_size,
        padding_size=page_size * dcp_size,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    assert pool.k_buffer[0].shape[0] == usable_tokens + page_size * dcp_size
    final_page = torch.arange(usable_tokens, usable_tokens + page_size * dcp_size)
    pool.k_buffer[0][final_page] = 11
    pool.v_buffer[0][final_page] = 12
    assert (pool.k_buffer[0][final_page] == 11).all()
    assert (pool.v_buffer[0][final_page] == 12).all()
    _, sizes, item_sizes = pool.get_contiguous_buf_infos()
    assert all(size == (usable_tokens + page_size * dcp_size) * 8 for size in sizes)
    assert all(size == page_size * 8 for size in item_sizes)


def test_draft_default_padding_is_one_page():
    pool = MHATokenToKVPool(
        size=128,
        page_size=64,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    assert pool.k_buffer[0].shape[0] == 192


def test_invalid_draft_geometry_is_rejected():
    with pytest.raises(ValueError, match="divide evenly"):
        draft_kv_bytes_per_target_token(
            num_layers=4,
            total_kv_heads=7,
            head_dim=128,
            tp_size=8,
            dcp_size=8,
            dtype="bf16",
        )


def test_full_draft_registers_as_independent_state_component():
    from sglang.srt.disaggregation.base.conn import KVArgs, StateType
    from sglang.srt.disaggregation.utils import setup_state_kv_args

    args = KVArgs()
    draft = SimpleNamespace(
        _pd_dflash_full_kv=True,
        get_contiguous_buf_infos=lambda: ([101, 102], [8192, 8192], [512, 512]),
    )
    setup_state_kv_args(args, object(), draft)
    assert args.state_types == [StateType.DFLASH_KV]
    assert args.state_data_ptrs == [[101, 102]]
    assert args.state_item_lens == [[512, 512]]


@pytest.mark.parametrize("role", ["prefill", "decode", "null"])
@pytest.mark.parametrize("replay", [False, True])
def test_prefill_budget_does_not_charge_verify_scratch(role, replay):
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    args = SimpleNamespace(
        disaggregation_mode=role,
        enable_linear_replayssm_spec=replay,
        linear_replayssm_cache_len=16,
        speculative_num_draft_tokens=8,
        max_running_requests=64,
        max_mamba_cache_size=256,
        disable_radix_cache=False,
    )
    args.override = lambda reason, **kw: args.__dict__.update(kw)
    per_req = 1024 * 1024
    ring = 128 * 1024
    cache = SimpleNamespace(
        mamba_cache_per_req=per_req, replayssm_ring_bytes_per_req=lambda **kw: ring
    )
    kvc = SimpleNamespace(
        server_args=args,
        mambaish_config=SimpleNamespace(mamba2_cache_params=cache),
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        hybrid_gdn_config=object(),
        model_config=None,
        ps=SimpleNamespace(attn_dp_size=1),
        _calculate_mamba_ratio=lambda: 2,
    )
    remaining = KVCacheConfigurator._handle_max_mamba_cache(kvc, 100.0)
    expected = (256 + 1) * per_req
    if role != "prefill":
        expected += (256 + 1) * ring if replay else (64 + 1) * 8 * per_req
    assert remaining == 100.0 - expected / (1 << 30)


@pytest.mark.parametrize("overlap", [False, True])
def test_dflash_pd_input_publishes_transferred_bonus_tokens(overlap):
    from unittest.mock import MagicMock

    from sglang.srt.speculative.dflash_disaggregation import (
        build_dflash_disagg_draft_input,
    )

    batch = SimpleNamespace(
        seq_lens=torch.tensor([513, 901]),
        req_pool_indices=torch.tensor([2, 7]),
        enable_overlap=overlap,
    )
    tokens = torch.tensor([101, 102])
    future = MagicMock()
    spec = build_dflash_disagg_draft_input(batch, None, tokens, future)
    torch.testing.assert_close(spec.bonus_tokens, tokens)
    torch.testing.assert_close(spec.new_seq_lens, batch.seq_lens)
    if overlap:
        future.publish.assert_called_once()
        future.stash.assert_called_once()
        assert future.publish.call_args.args[0] is batch.req_pool_indices
    else:
        future.publish.assert_not_called()
        future.stash.assert_not_called()
