"""Every GPU KV writer must leave reserved row 0 (the padded-token sink that
padded CUDA-graph rows target via ``out_cache_loc == 0``) untouched when
``reserved_skip_index=0``, and write every other row exactly as before. Also
checks the paged allocator zeroes recycled page envelopes on ``alloc_extend`` /
``alloc_decode`` hand-out."""

import pytest
import torch

from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q_fp8,
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
    can_use_set_mla_kv_buffer,
    set_mla_kv_buffer,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    _set_kv_buffer_prefix_valid_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
NOPE, ROPE, TOTAL = 512, 64, 576
PAGES = 512
BS = 37


def _loc_with_reserved(bs=BS, n_reserved=3, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    real = (
        torch.randperm(PAGES - 1, generator=gen, device=DEVICE)[: bs - n_reserved] + 1
    )
    loc = torch.cat([real, torch.zeros(n_reserved, dtype=real.dtype, device=DEVICE)])
    return loc[torch.randperm(bs, generator=gen, device=DEVICE)]


def _assert_only_real_rows_written(pool, loc, expected_rows):
    assert torch.count_nonzero(pool[0].view(torch.uint8)) == 0
    real = loc != 0
    assert torch.equal(
        pool[loc[real]].view(torch.uint8),
        expected_rows[real].view(torch.uint8),
    )
    others = torch.ones(PAGES, dtype=torch.bool, device=DEVICE)
    others[loc] = False
    assert torch.count_nonzero(pool[others].view(torch.uint8)) == 0


def _run_bf16_mla_writer(writer):
    loc = _loc_with_reserved()
    k_nope = torch.randn(BS, 1, NOPE, dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn(BS, 1, ROPE, dtype=torch.bfloat16, device=DEVICE)
    pool = torch.zeros(PAGES, 1, TOTAL, dtype=torch.bfloat16, device=DEVICE)
    writer(pool, loc, k_nope, k_rope)
    torch.cuda.synchronize()
    _assert_only_real_rows_written(
        pool.view(PAGES, TOTAL),
        loc,
        torch.cat([k_nope, k_rope], dim=-1).view(BS, TOTAL),
    )


def test_set_mla_kv_buffer_triton_skips_reserved():
    _run_bf16_mla_writer(
        lambda p, l, n, r: set_mla_kv_buffer_triton(p, l, n, r, reserved_skip_index=0)
    )


def test_set_mla_kv_buffer_tma_skips_reserved():
    if not can_use_set_mla_kv_buffer(NOPE * 2, ROPE * 2):
        pytest.skip("TMA set_mla_kv_buffer unsupported on this arch")
    _run_bf16_mla_writer(
        lambda p, l, n, r: set_mla_kv_buffer(p, l, n, r, reserved_skip_index=0)
    )


def test_set_mla_kv_scale_buffer_triton_skips_reserved():
    loc = _loc_with_reserved()
    k_nope = torch.randn(BS, 1, 4, dtype=torch.float32, device=DEVICE)
    k_rope = torch.randn(BS, 1, 4, dtype=torch.float32, device=DEVICE)
    pool = torch.zeros(PAGES, 1, 8, dtype=torch.float32, device=DEVICE)
    set_mla_kv_scale_buffer_triton(pool, loc, k_nope, k_rope, reserved_skip_index=0)
    torch.cuda.synchronize()
    _assert_only_real_rows_written(
        pool.view(PAGES, 8), loc, torch.cat([k_nope, k_rope], dim=-1).view(BS, 8)
    )


def test_set_mla_kv_buffer_triton_fp8_quant_skips_reserved():
    loc = _loc_with_reserved()
    k_nope = torch.randn(BS, 1, NOPE, dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn(BS, 1, ROPE, dtype=torch.bfloat16, device=DEVICE)
    pool = torch.zeros(PAGES, 1, TOTAL, dtype=FP8, device=DEVICE)
    ref = torch.zeros_like(pool)
    set_mla_kv_buffer_triton_fp8_quant(
        pool, loc, k_nope, k_rope, FP8, reserved_skip_index=0
    )
    set_mla_kv_buffer_triton_fp8_quant(ref, loc, k_nope, k_rope, FP8)
    torch.cuda.synchronize()
    assert torch.count_nonzero(pool[0].view(torch.uint8)) == 0
    real = loc != 0
    assert torch.equal(
        pool[loc[real]].view(torch.uint8), ref[loc[real]].view(torch.uint8)
    )


def test_set_mla_kv_concat_q_fp8_skips_reserved_but_converts_q():
    if not can_use_set_mla_kv_concat_q_fp8():
        pytest.skip("fused fp8 set_mla_kv_concat_q requires SM90+")
    loc = _loc_with_reserved()
    heads = 8
    k_nope = torch.randn(BS, 1, NOPE, dtype=torch.bfloat16, device=DEVICE)
    k_rope = torch.randn(BS, 1, ROPE, dtype=torch.bfloat16, device=DEVICE)
    q_nope = torch.randn(BS, heads, NOPE, dtype=torch.bfloat16, device=DEVICE)
    q_rope = torch.randn(BS, heads, ROPE, dtype=torch.bfloat16, device=DEVICE)
    pool = torch.zeros(PAGES, TOTAL, dtype=FP8, device=DEVICE)
    ref = torch.zeros_like(pool)
    q = set_mla_kv_concat_q_fp8(
        pool, loc, k_nope, k_rope, q_nope, q_rope, reserved_skip_index=0
    )
    q_ref = set_mla_kv_concat_q_fp8(ref, loc, k_nope, k_rope, q_nope, q_rope)
    torch.cuda.synchronize()
    assert torch.count_nonzero(pool[0].view(torch.uint8)) == 0
    real = loc != 0
    assert torch.equal(
        pool[loc[real]].view(torch.uint8), ref[loc[real]].view(torch.uint8)
    )
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def test_store_cache_skips_reserved():
    loc = _loc_with_reserved()
    dim = 256
    k = torch.randn(BS, dim, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(BS, dim, dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.zeros(PAGES, dim, dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.zeros(PAGES, dim, dtype=torch.bfloat16, device=DEVICE)
    store_cache(k, v, k_cache, v_cache, loc, reserved_skip_index=0)
    torch.cuda.synchronize()
    _assert_only_real_rows_written(k_cache, loc, k)
    _assert_only_real_rows_written(v_cache, loc, v)


def test_prefix_valid_page_major_skips_reserved():
    bs, block = 4, 8
    dim = 128
    loc_2d = torch.zeros(bs, block, dtype=torch.int64, device=DEVICE)
    # Row 0 is padded (all zeros); rows 1..3 real with a trailing uncommitted tail.
    for i in range(1, bs):
        loc_2d[i] = torch.arange(i * 16, i * 16 + block, device=DEVICE)
    commit_lens = torch.tensor([block, block, 5, 1], dtype=torch.int32, device=DEVICE)
    k = torch.randn(bs * block, 1, dim, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(bs * block, 1, dim, dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.zeros(PAGES, 1, dim, dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.zeros(PAGES, 1, dim, dtype=torch.bfloat16, device=DEVICE)
    _set_kv_buffer_prefix_valid_impl(
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        row_dim=dim,
        store_dtype=torch.bfloat16,
        reserved_skip_index=0,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(k_cache[0].view(torch.uint8)) == 0
    assert torch.count_nonzero(v_cache[0].view(torch.uint8)) == 0
    for i in range(1, bs):
        n = int(commit_lens[i])
        rows = loc_2d[i, :n]
        assert torch.equal(k_cache[rows], k[i * block : i * block + n])
        assert torch.equal(v_cache[rows], v[i * block : i * block + n])
        tail = loc_2d[i, n:]
        assert torch.count_nonzero(k_cache[tail].view(torch.uint8)) == 0


@pytest.mark.parametrize("pool_kind", ["mha", "mla"])
def test_pool_set_kv_buffer_skips_reserved(pool_kind):
    page = 64
    size = 8 * page
    if pool_kind == "mha":
        pool = MHATokenToKVPool(
            size=size,
            page_size=page,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=1,
            device=DEVICE,
            enable_memory_saver=False,
        )
        bufs = [pool.k_buffer[0], pool.v_buffer[0]]
        k = torch.randn(BS, 2, 128, dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn(BS, 2, 128, dtype=torch.bfloat16, device=DEVICE)
    else:
        pool = MLATokenToKVPool(
            size=size,
            page_size=page,
            dtype=torch.bfloat16,
            kv_lora_rank=NOPE,
            qk_rope_head_dim=ROPE,
            layer_num=1,
            device=DEVICE,
            enable_memory_saver=False,
        )
        bufs = [pool.kv_buffer[0]]
        k = torch.randn(BS, 1, TOTAL, dtype=torch.bfloat16, device=DEVICE)
        v = None
    assert pool.reserved_skip_index == 0
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    real = torch.randperm(size + page - 1, generator=gen, device=DEVICE)[: BS - 2] + 1
    loc = torch.cat([torch.zeros(2, dtype=real.dtype, device=DEVICE), real])

    class _Layer:
        layer_id = 0

    pool.set_kv_buffer(_Layer(), loc, k, v)
    if pool_kind == "mla":
        pool.set_mla_kv_buffer(_Layer(), loc, k[..., :NOPE], k[..., NOPE:])
    torch.cuda.synchronize()
    for buf in bufs:
        assert torch.count_nonzero(buf[0].view(torch.uint8)) == 0
    if pool_kind == "mha":
        assert torch.equal(pool.k_buffer[0][real], k[2:])
        assert torch.equal(pool.v_buffer[0][real], v[2:])
    else:
        assert torch.equal(pool.kv_buffer[0][real], k[2:])


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8])
def test_paged_allocator_zeroes_extend_and_decode_pages(dtype):
    page = 64
    size = 32 * page
    pool = MLATokenToKVPool(
        size=size,
        page_size=page,
        dtype=dtype,
        kv_lora_rank=NOPE,
        qk_rope_head_dim=ROPE,
        layer_num=2,
        device=DEVICE,
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size, page, dtype, DEVICE, pool, False, zero_pages_on_alloc=True
    )

    def dirty():
        for buf in pool.kv_buffer:
            buf.fill_(1.0)

    def assert_pages_zero(idx):
        pages = torch.unique(idx // page)
        rows = pool.page_rows(pages)
        for buf in pool.kv_buffer:
            assert torch.count_nonzero(buf[rows].view(torch.uint8)) == 0

    # alloc_extend from empty: 64 tokens (1 page) and 70 tokens (2 pages).
    dirty()
    prefix = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    seq = torch.tensor([page, 70], device=DEVICE)
    last_loc = torch.full((2,), -1, dtype=torch.int64, device=DEVICE)
    idx = allocator.alloc_extend(prefix, prefix.cpu(), seq, seq.cpu(), last_loc, 134)
    torch.cuda.synchronize()
    assert torch.unique(idx // page).numel() == 3
    assert_pages_zero(idx)
    # Pages still on the free list were not touched.
    free_rows = pool.page_rows(allocator.free_pages)
    assert torch.count_nonzero(pool.kv_buffer[0][free_rows].view(torch.uint8)) > 0

    # alloc_decode: request 0 crosses a page boundary (new zeroed page);
    # request 1 stays inside its partially filled page (nothing zeroed).
    dirty()
    seq_d = torch.tensor([page + 1, 71], device=DEVICE)
    last_d = torch.stack([idx[page - 1], idx[page + 69]])
    idx_d = allocator.alloc_decode(seq_d, seq_d.cpu(), last_d)
    torch.cuda.synchronize()
    assert idx_d[0] // page not in torch.unique(idx // page).tolist()
    assert_pages_zero(idx_d[:1])
    assert torch.count_nonzero(pool.kv_buffer[0][idx_d[1]].view(torch.uint8)) > 0
