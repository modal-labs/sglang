"""Compile request-path Triton kernels before a Kimi-K3 worker is ready.

These kernels sit around (rather than inside) model forward, so CUDA graph
capture cannot discover them.  Triton's ``JITFunction.warmup`` compiles the
exact specialization without launching the kernel; using the live memory
pools supplies configuration-derived strides, dtypes, and page geometry
without manufacturing HTTP requests or maintaining request-shape buckets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import triton

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _compile_on_rank_zero(
    model_runner: ModelRunner,
    compile_fn: Callable[[], tuple[str, ...]],
) -> tuple[str, ...]:
    """Compile once per TP group, then make every rank load the artifacts."""

    parallel = get_parallel()
    tp_group = model_runner.tp_group
    compiled: tuple[str, ...] = ()
    rank_zero_error = None
    if parallel.tp_rank == 0:
        try:
            compiled = compile_fn()
        except Exception as exc:
            rank_zero_error = f"{type(exc).__name__}: {exc}"
    if parallel.tp_size > 1:
        rank_zero_error = tp_group.broadcast_object(rank_zero_error, src=0)
    if rank_zero_error is not None:
        raise RuntimeError(
            f"K3 runtime Triton precompile failed on TP rank 0: {rank_zero_error}"
        )

    peer_error = None
    if parallel.tp_rank != 0:
        try:
            compile_fn()
        except Exception as exc:
            peer_error = f"rank={parallel.tp_rank} {type(exc).__name__}: {exc}"
    if parallel.tp_size > 1:
        peer_errors = tp_group.all_gather_object(peer_error)
        peer_errors = [error for error in peer_errors if error is not None]
        if peer_errors:
            raise RuntimeError(
                "K3 runtime Triton artifact load failed: " + "; ".join(peer_errors)
            )
    return compiled


def _precompile_request_table_kernels(model_runner: ModelRunner) -> tuple[str, ...]:
    """Compile the request-table kernels for this runner's live pool layout."""

    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.kernels.ops.memory.common import get_last_loc_kernel
    from sglang.srt.mem_cache.allocation import assign_req_to_token_pool

    req_to_token = model_runner.req_to_token_pool.req_to_token
    device = req_to_token.device
    max_batch_size = max(1, int(model_runner.max_running_requests))
    page_allocator = model_runner.token_to_kv_pool_allocator
    compiled: list[str] = []

    # Paged target allocation is exercised with both the int64 serving
    # ForwardBatch lengths and DFlash's compact int32 lengths.  Compile both
    # dtype specializations against one stable, configuration-derived batch
    # bound.  The allocator patch uses the same bound at runtime.
    if (
        hasattr(page_allocator, "free_pages")
        and isinstance(page_allocator.free_pages, torch.Tensor)
        and hasattr(page_allocator, "page_size")
    ):
        batch_upper = page_allocator.triton_batch_size_upper_bound(1)
        for length_dtype in (torch.int32, torch.int64):
            lengths = torch.empty((max_batch_size,), dtype=length_dtype, device=device)
            out_indices = torch.empty(
                (max_batch_size,), dtype=torch.int64, device=device
            )
            alloc_extend_kernel.warmup(
                lengths,
                lengths,
                lengths,
                page_allocator.free_pages,
                out_indices,
                batch_upper,
                int(page_allocator.page_size),
                grid=(1,),
            )
        compiled.append("alloc_extend_kernel")

    req_pool_indices = torch.empty((max_batch_size,), dtype=torch.int64, device=device)
    offsets_i32 = torch.empty((max_batch_size,), dtype=torch.int32, device=device)
    cache_locs = torch.empty((max_batch_size,), dtype=torch.int64, device=device)

    # The request-pool row pitch is different for target and windowed draft
    # tables.  This hook runs once for each runner and therefore compiles both
    # naturally, with no explicit model/configuration branch.
    assign_req_to_token_pool.warmup(
        req_pool_indices,
        req_to_token,
        offsets_i32,
        offsets_i32,
        cache_locs,
        int(req_to_token.shape[1]),
        triton.next_power_of_2(int(req_to_token.shape[0]) - 1),
        grid=(1,),
    )
    compiled.append("assign_req_to_token_pool")

    # CUDA serving currently uses int32 lengths, while the allocator also
    # accepts int64 lengths. Compile both legal get-last variants so a future
    # scheduler dtype change cannot move this JIT back onto the request path.
    for prefix_dtype in (torch.int32, torch.int64):
        prefix_lens = torch.empty((max_batch_size,), dtype=prefix_dtype, device=device)
        last_locs = torch.empty_like(prefix_lens)
        get_last_loc_kernel.warmup(
            req_to_token,
            req_pool_indices,
            prefix_lens,
            last_locs,
            max_batch_size,
            req_to_token.stride(0),
            256,
            grid=(triton.cdiv(max_batch_size, 256),),
        )
    compiled.append("get_last_loc_kernel")
    return tuple(compiled)


def _precompile_mla_fused_projection_write(
    model_runner: ModelRunner,
    mla_pool,
) -> tuple[str, ...]:
    """Compile MLA writes from both compact and fused-projection views.

    Kimi-K3's fused MLA A projection produces
    ``[q_lora, kv_lora, q_rope]`` in one row.  Its KV slices therefore retain
    the full projection row pitch, whereas conversion/capture paths can hand
    the same write kernel compact KV tensors.  The source pitch is model
    geometry, not a request-length bucket.
    """

    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_kernel

    if not hasattr(mla_pool, "kv_buffer"):
        return ()

    model_config = model_runner.model_config
    text_config = getattr(model_config, "hf_text_config", None)
    q_lora_rank = getattr(text_config, "q_lora_rank", None)
    if q_lora_rank is None:
        q_lora_rank = getattr(
            getattr(model_config, "hf_config", None), "q_lora_rank", None
        )
    if q_lora_rank is None or int(q_lora_rank) <= 0:
        return ()

    nope_dim = int(mla_pool.kv_lora_rank)
    rope_dim = int(mla_pool.qk_rope_head_dim)
    kv_buffer = mla_pool.kv_buffer[0]
    device = kv_buffer.device
    dtype = kv_buffer.dtype
    loc = torch.empty((1,), dtype=torch.int64, device=device)

    compact_nope = torch.empty((1, 1, nope_dim), dtype=dtype, device=device)
    compact_rope = torch.empty((1, 1, rope_dim), dtype=dtype, device=device)

    fused_width = int(q_lora_rank) + nope_dim + rope_dim
    fused_projection = torch.empty((1, fused_width), dtype=dtype, device=device)
    fused_latent = fused_projection[:, int(q_lora_rank) :]
    fused_nope = fused_latent[:, :nope_dim].unsqueeze(1)
    fused_rope = fused_latent[:, nope_dim:].unsqueeze(1)

    parallel = get_parallel()
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    for cache_k_nope, cache_k_rope in (
        (compact_nope, compact_rope),
        (fused_nope, fused_rope),
    ):
        set_mla_kv_buffer_kernel.warmup(
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK=triton.next_power_of_2(nope_dim + rope_dim),
            DCP_RANK=parallel.attn_dcp_rank,
            DCP_WORLD_SIZE=parallel.attn_dcp_size,
            grid=(1, 1),
            **pdl_kwargs,
        )
    return ("set_mla_kv_buffer_kernel",)


def _precompile_kimi_hybrid_kernels(
    model_runner: ModelRunner,
) -> tuple[str, ...]:
    """Compile Kimi target-only prefix, MLA-read, and Mamba COW kernels."""

    from sglang.kernels.ops.kvcache.kv_indices import (
        create_chunked_prefix_cache_kv_indices,
    )
    from sglang.kernels.ops.kvcache.mla_buffer import get_mla_kv_buffer_kernel
    from sglang.srt.mem_cache.mamba_slot_fused import (
        _BLOCK,
        _fused_slot_clear_kernel,
        _fused_slot_copy_kernel,
    )

    req_pool = model_runner.req_to_token_pool
    req_to_token = req_pool.req_to_token
    device = req_to_token.device
    compiled: list[str] = []

    req_pool_indices = torch.empty((1,), dtype=torch.int64, device=device)
    chunk_i32 = torch.empty((1,), dtype=torch.int32, device=device)
    chunk_cu_i32 = torch.empty((2,), dtype=torch.int32, device=device)
    chunk_out_i32 = torch.empty((1,), dtype=torch.int32, device=device)
    create_chunked_prefix_cache_kv_indices.warmup(
        req_to_token,
        req_pool_indices,
        chunk_i32,
        chunk_i32,
        chunk_cu_i32,
        chunk_out_i32,
        req_to_token.stride(0),
        grid=(1,),
    )
    compiled.append("create_chunked_prefix_cache_kv_indices")

    kv_pool = model_runner.token_to_kv_pool
    # K3's model-facing pool is HybridLinearKVPool; its dense MLA storage and
    # MLA geometry live on full_kv_pool.
    mla_pool = (
        kv_pool.full_kv_pool
        if getattr(kv_pool, "use_mla", False) and hasattr(kv_pool, "full_kv_pool")
        else kv_pool
    )
    if all(
        hasattr(mla_pool, name)
        for name in (
            "get_key_buffer",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "dtype",
        )
    ):
        layer_id = int(getattr(mla_pool, "start_layer", 0))
        kv_buffer = mla_pool.get_key_buffer(layer_id)
        nope_dim = int(mla_pool.kv_lora_rank)
        rope_dim = int(mla_pool.qk_rope_head_dim)
        loc = torch.empty((1,), dtype=torch.int32, device=device)
        cache_k_nope = torch.empty(
            (1, 1, nope_dim), dtype=mla_pool.dtype, device=device
        )
        cache_k_rope = torch.empty(
            (1, 1, rope_dim), dtype=mla_pool.dtype, device=device
        )
        get_mla_kv_buffer_kernel.warmup(
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            grid=(1,),
        )
        compiled.append("get_mla_kv_buffer_kernel")
        compiled.extend(_precompile_mla_fused_projection_write(model_runner, mla_pool))

    mamba_pool = getattr(req_pool, "mamba_pool", None)
    if mamba_pool is not None and mamba_pool._should_fuse_slot_ops():
        desc = mamba_pool._conv_slot_desc
        slot_indices = torch.empty((1,), dtype=torch.int64, device=device)
        _fused_slot_clear_kernel.warmup(
            desc.ptr,
            desc.feat,
            desc.layer_stride,
            desc.slot_stride,
            slot_indices,
            MAX_FEAT_BLOCKS=desc.max_feat_blocks,
            BLOCK=_BLOCK,
            grid=(1, desc.ptr.numel(), desc.num_layers),
        )
        _fused_slot_copy_kernel.warmup(
            desc.ptr,
            desc.feat,
            desc.layer_stride,
            desc.slot_stride,
            slot_indices,
            slot_indices,
            MAX_FEAT_BLOCKS=desc.max_feat_blocks,
            BLOCK=_BLOCK,
            grid=(1, desc.ptr.numel(), desc.num_layers),
        )
        compiled.append("_fused_slot_clear_kernel")
        compiled.append("_fused_slot_copy_kernel")

    return tuple(compiled)


def precompile_k3_runtime_kernels(
    model_runner: ModelRunner,
    *,
    include_kimi_hybrid: bool,
) -> tuple[str, ...]:
    """Compile all non-forward Triton paths used by this K3 target/draft."""

    def compile_all() -> tuple[str, ...]:
        compiled = list(_precompile_request_table_kernels(model_runner))
        if include_kimi_hybrid:
            compiled.extend(_precompile_kimi_hybrid_kernels(model_runner))
        return tuple(compiled)

    compiled = _compile_on_rank_zero(model_runner, compile_all)
    if compiled:
        logger.info(
            "K3_RUNTIME_TRITON_READY worker=%s factories=%s",
            "draft" if model_runner.is_draft_worker else "target",
            ",".join(compiled),
        )
    return compiled
