"""CPU layout regression only; GPU retraction/generation is a release gate."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt import runtime_context as rc
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("world,rank", [(n, r) for n in [1, 2, 8] for r in range(n)])
@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 23, 64])
@pytest.mark.parametrize("has_draft", [False, True])
@pytest.mark.parametrize("has_mamba", [False, True])
def test_offload_relocates_target_draft_and_mamba(
    monkeypatch, world, rank, length, has_draft, has_mamba
):
    monkeypatch.setattr(memory_pool.current_platform, "synchronize", lambda: None)
    target = MLATokenToKVPool.__new__(MLATokenToKVPool)
    target.kv_buffer = [torch.arange(320 // world * 2).reshape(-1, 2)]
    target.layer_num = 1
    target.cpu_offloading_chunk_size = 3
    draft = MHATokenToKVPool(
        size=320,
        page_size=64,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    draft.k_buffer[0].copy_(torch.arange(384 * 2).reshape(384, 1, 2))
    draft.v_buffer[0].copy_(draft.k_buffer[0] + 1000)
    mamba_data = torch.arange(40).reshape(20, 2)
    mamba = SimpleNamespace(
        get_cpu_copy=lambda idx: mamba_data[idx].clone(),
        load_cpu_copy=lambda value, idx: mamba_data.index_copy_(0, idx, value),
    )
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.full_kv_pool = target
    pool.mamba_pool = mamba
    pool._mamba_translate = lambda idx: idx + 2
    if has_draft:
        pool._pd_dflash_draft_kv_pool = draft
    src = torch.arange(64, 64 + length)
    dst = torch.arange(192, 192 + length)
    expected_target = target.kv_buffer[0][src[src % world == rank] // world].clone()
    expected_k = draft.k_buffer[0][src].clone()
    expected_v = draft.v_buffer[0][src].clone()
    expected_mamba = mamba_data[3].clone()
    with rc.get_parallel().override(
        dcp_enabled=world > 1, dcp_size=world, dcp_rank=rank
    ):
        copied = pool.get_cpu_copy(src, torch.tensor([1]) if has_mamba else None)
        target.kv_buffer[0].fill_(-1)
        draft.k_buffer[0].fill_(-2)
        draft.v_buffer[0].fill_(-3)
        mamba_data.fill_(-4)
        pool.load_cpu_copy(copied, dst, torch.tensor([5]) if has_mamba else None)
    torch.testing.assert_close(
        target.kv_buffer[0][dst[dst % world == rank] // world], expected_target
    )
    if has_draft:
        torch.testing.assert_close(draft.k_buffer[0][dst], expected_k)
        torch.testing.assert_close(draft.v_buffer[0][dst], expected_v)
    if has_mamba:
        torch.testing.assert_close(mamba_data[7], expected_mamba)
