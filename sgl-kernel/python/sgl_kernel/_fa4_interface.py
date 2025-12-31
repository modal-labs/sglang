# Adapted from https://github.com/Dao-AILab/flash-attention/blob/54d8aa6751fc9d5f0357854079261913d5df1f9d/flash_attn/cute/interface.py

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-10-14] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.2.1.


import copy
import gc
import logging
import math
import os
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from flash_attn_origin.cute import utils
from flash_attn_origin.cute.interface import _flash_attn_fwd as _flash_attn_fwd_origin
from flash_attn_origin.cute.flash_fwd import FlashAttentionForwardSm90
from flash_attn_origin.cute.flash_fwd_sm100 import FlashAttentionForwardSm100


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Callable | None = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    buffers: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert (
            page_table.stride(-1) == 1
        ), "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (
        batch_size,
    ), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (
        batch_size,
    ), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert (
                t.dtype == torch.int32
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert (
                t.stride(0) == 1
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"
    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=device,
        )
    else:
        expected_out_shape = (*q_batch_seqlen_shape, num_head, head_dim_v)
        assert (
            out.shape == expected_out_shape
        ), f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        assert (
            out.dtype == out_torch_dtype
        ), f"out tensor dtype {out.dtype} does not match expected dtype {out_torch_dtype}"
        assert (
            out.device == device
        ), f"out tensor device {out.device} does not match input device {device}"
        assert out.is_cuda, "out tensor must be on CUDA device"

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert (
            lse.shape == lse_shape
        ), f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        assert (
            lse.dtype == torch.float32
        ), f"lse tensor dtype {lse.dtype} does not match expected dtype torch.float32"
        assert (
            lse.device == device
        ), f"lse tensor device {lse.device} does not match input device {device}"
        assert lse.is_cuda, "lse tensor must be on CUDA device"

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (q, k, v, out)
    ]
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
        if lse is not None
        else None
    )
    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
    ) = [
        (
            from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
            if t is not None
            else None
        )
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    page_table_tensor = (
        from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1
        )
        if page_table is not None
        else None
    )
    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
        else:
            causal, local = False, True
    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability in [
        9,
        10,
    ], "Unsupported compute capability. Supported: 9.x, 10.x"
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim
        # Perf heuristic from upstream: hdim=128, noncausal, non-local benefits from larger n_block
        if head_dim == head_dim_v == 128 and not causal and not local:
            n_block_size = 192
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    if score_mod is not None:
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
        if is_varlen:
            raise NotImplementedError(
                "score_mod with buffers is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    cute_buffers = None
    if buffers is not None:
        cute_buffers = [from_dlpack(buf) for buf in buffers]

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        utils.hash_callable(score_mod) if score_mod is not None else None,
        buffers is not None,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        compute_capability,
    )

    # Compute q_stage based on sequence length
    max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if compute_capability == 10:
        q_stage = 2 if seqlen_q_packgqa > m_block_size else 1
    else:
        q_stage = 1

    if compile_key not in _flash_attn_fwd.compile_cache:
        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=None,
                has_aux_tensors=False,
            )
        elif compute_capability == 10:
            assert page_size in [
                None,
                128,
            ], "Only page_size=128 is supported for paged KV on SM 10.0"
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=False,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                q_stage=q_stage,
                is_persistent=not causal
                and not local
                and cu_seqlens_q is None
                and seqused_q is None,
                score_mod=score_mod,
                mask_mod=None,
                has_aux_tensors=False,
                paged_kv_non_tma=page_size not in [None, 128],
                is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        # TODO caching for buffers; cute_buffers
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            cute_buffers,
        )
    _flash_attn_fwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        page_table_tensor,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
        cute_buffers,
    )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def warmup_flash_attn(f):
    """
    Decorator for flash_attn_varlen_func:
    - On first call, run several warmup passes with different flag combinations:
        * return_softmax_lse in {False, True}
        * global noncausal (window_size=(None,None))
        * causal (window_size=(None,0))
        * local sliding window (window_size=(64,64))
        * optionally pack_gqa=True if qheads > kvheads and allowed
    - No score_mod / softcap (not supported for varlen yet)
    - Executes sequentially to minimize peak GPU mem
    - Does not modify user tensors (clones)
    """
    disable_warmup = os.getenv("SGLANG_DISABLE_FA4_WARMUP", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if disable_warmup:
        return f

    done = False

    def _clone_args(args, kwargs):
        """Clone tensor arguments to avoid sharing storage; deepcopy for others."""

        def maybe_clone(x):
            if isinstance(x, torch.Tensor):
                return x.detach().clone()  # detach to avoid autograd edges
            return copy.deepcopy(x)

        return tuple(maybe_clone(a) for a in args), {
            k: maybe_clone(v) for k, v in kwargs.items()
        }

    def _infer_heads(args, kwargs):
        """Infer q and kv head counts from arguments."""
        # Expect signature: (q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
        q = args[0] if len(args) > 0 else kwargs.get("q")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        try:
            qh = int(q.shape[-2])
            kvh = int(k.shape[-2])
            return qh, kvh
        except Exception:
            return None, None

    def _run_warmups(args, kwargs):
        """Run warmup calls sequentially and release memory after each."""
        base_args, base_kwargs = _clone_args(args, kwargs)

        qh, kvh = _infer_heads(base_args, base_kwargs)
        can_pack_gqa = (
            qh is not None and kvh is not None and qh % kvh == 0 and qh // kvh > 1
        )
        has_page_table = (
            "page_table" in base_kwargs and base_kwargs["page_table"] is not None
        )

        # Window presets covering global, causal, and local
        window_presets = [
            (None, None),  # global noncausal
            (None, 0),  # causal
            (64, 64),  # local sliding window
        ]

        lse_flags = [False, True]

        # Base combo list
        combos = []
        for ws in window_presets:
            for return_lse_flag in lse_flags:
                combos.append(dict(window_size=ws, return_softmax_lse=return_lse_flag))

        # Optionally add a pack_gqa=True variant (FA4 may disable it internally for some varlen shapes/SMs)
        if can_pack_gqa:
            for ws in window_presets:
                combos.append(
                    dict(window_size=ws, return_softmax_lse=False, pack_gqa=True)
                )

        # If page_table is present, warm one combo with it (page_table in compile key for SM100)
        if has_page_table:
            combos.append(dict(window_size=(None, None), return_softmax_lse=False))

        # Run sequentially
        for combo in combos:
            wa, wk = _clone_args(base_args, base_kwargs)
            # Keep user-provided softcap/score_mod OUT (varlen+score_mod unsupported)
            wk.pop("score_mod", None)
            if "softcap" in wk and wk["softcap"]:
                wk["softcap"] = 0.0
            # Apply combo
            wk.update(combo)
            with torch.cuda.stream(torch.cuda.current_stream()):
                try:
                    f(*wa, **wk)
                except Exception as e:
                    # Some combos can be invalid for specific head dims / arch. Ignore and continue.
                    logger.debug("Warmup combo skipped: %s", e)
            del wa, wk
            torch.cuda.empty_cache()
            gc.collect()

    def wrapper(*args, **kwargs):
        nonlocal done
        if not done:
            logger.info(
                "Running FA4 warmup (global/causal/local, LSE on/off, optional GQA pack)..."
            )
            _run_warmups(args, kwargs)
            done = True
        return f(*args, **kwargs)

    return wrapper


@warmup_flash_attn
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    return_softmax_lse: Optional[bool] = False,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        num_splits == 0
        and max_seqlen_q == 1
        and max_seqlen_k is not None
        and torch.cuda.get_device_capability()[0] == 10
    ):
        if max_seqlen_k <= 4096:
            num_splits = 1

    is_varlen_q = cu_seqlens_q is not None or seqused_q is not None
    if (
        is_varlen_q
        and q.ndim == 3
        and cu_seqlens_q is not None
        and max_seqlen_q == 1
        and max_seqlen_k is not None
        and max_seqlen_k >= 1024
        and window_size[0] is None
        and window_size[1] is None
        and learnable_sink is None
        and pack_gqa is not False
        and torch.cuda.get_device_capability(q.device)[0] in (10, 11)
    ):
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q, num_heads, head_dim = q.shape
        if total_q == batch_size:
            num_kv_heads = k.shape[-2]
            q_heads_per_kv_head = num_heads // num_kv_heads
            if (
                q_heads_per_kv_head > 1
                and num_heads % num_kv_heads == 0
                and (128 % q_heads_per_kv_head != 0)
            ):
                head_dim_v = v.shape[-1]
                q_4d = q.view(batch_size, 1, num_heads, head_dim)
                q_grouped = q_4d.view(
                    batch_size, 1, num_kv_heads, q_heads_per_kv_head, head_dim
                )
                q_packed = q_grouped.permute(0, 1, 3, 2, 4).reshape(
                    batch_size, q_heads_per_kv_head, num_kv_heads, head_dim
                )

                out_packed, lse_packed = _flash_attn_fwd_origin(
                    q_packed,
                    k,
                    v,
                    None,
                    cu_seqlens_k,
                    None,
                    seqused_k,
                    page_table=page_table,
                    softmax_scale=softmax_scale,
                    causal=False,
                    window_size_left=None,
                    window_size_right=None,
                    learnable_sink=None,
                    softcap=softcap,
                    num_splits=num_splits,
                    pack_gqa=False,
                    return_lse=return_softmax_lse,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    max_seqlen_q=None,
                    max_seqlen_k=max_seqlen_k,
                )

                out_grouped = out_packed.view(
                    batch_size, 1, q_heads_per_kv_head, num_kv_heads, head_dim_v
                ).permute(0, 1, 3, 2, 4)
                out = out_grouped.reshape(batch_size, 1, num_heads, head_dim_v).view(
                    total_q, num_heads, head_dim_v
                )

                if return_softmax_lse:
                    lse_grouped = lse_packed.view(
                        batch_size, num_kv_heads, 1, q_heads_per_kv_head
                    ).permute(0, 2, 1, 3)
                    lse = lse_grouped.reshape(batch_size * 1, num_heads).T.contiguous()
                    return out, lse
                return out

    if (
        is_varlen_q
        and q.ndim == 3
        and max_seqlen_k is not None
        and max_seqlen_k >= 1024
        and pack_gqa is not False
        and torch.cuda.get_device_capability(q.device)[0] in (10, 11)
    ):
        num_heads = q.shape[-2]
        num_kv_heads = k.shape[-2]
        q_heads_per_kv_head = num_heads // num_kv_heads
        if (
            q_heads_per_kv_head > 1
            and num_heads % num_kv_heads == 0
            and (128 % q_heads_per_kv_head != 0)
        ):
            total_q, _, head_dim = q.shape
            head_dim_v = v.shape[-1]
            is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            out_dtype = torch.bfloat16 if is_fp8 else q.dtype
            out = torch.empty(
                (total_q, num_heads, head_dim_v), dtype=out_dtype, device=q.device
            )
            requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
            lse = (
                torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
                if requires_grad or return_softmax_lse
                else None
            )
            q_grouped = q.view(total_q, num_kv_heads, q_heads_per_kv_head, head_dim)
            out_grouped = out.view(
                total_q, num_kv_heads, q_heads_per_kv_head, head_dim_v
            )
            lse_grouped = (
                lse.view(num_kv_heads, q_heads_per_kv_head, total_q) if lse is not None else None
            )
            sink_grouped = (
                learnable_sink.view(num_kv_heads, q_heads_per_kv_head)
                if learnable_sink is not None
                else None
            )

            remaining = q_heads_per_kv_head
            start_head = 0
            while remaining > 0:
                chunk = 1 << (remaining.bit_length() - 1)
                if chunk > 128:
                    chunk = 128
                remaining -= chunk

                q_chunk = q_grouped[:, :, start_head : start_head + chunk, :].reshape(
                    total_q, num_kv_heads * chunk, head_dim
                )
                sink_chunk = (
                    sink_grouped[:, start_head : start_head + chunk].reshape(num_kv_heads * chunk)
                    if sink_grouped is not None
                    else None
                )

                out_chunk, lse_chunk = _flash_attn_fwd_origin(
                    q_chunk,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqused_q,
                    seqused_k,
                    page_table=page_table,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size_left=window_size[0],
                    window_size_right=window_size[1],
                    learnable_sink=sink_chunk,
                    softcap=softcap,
                    num_splits=num_splits,
                    pack_gqa=True,
                    return_lse=lse is not None,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )
                out_grouped[:, :, start_head : start_head + chunk, :].copy_(
                    out_chunk.view(total_q, num_kv_heads, chunk, head_dim_v)
                )
                if lse is not None:
                    lse_grouped[:, start_head : start_head + chunk, :].copy_(
                        lse_chunk.view(num_kv_heads, chunk, total_q)
                    )
                start_head += chunk

            return (out, lse) if return_softmax_lse else out
    out, lse = _flash_attn_fwd_origin(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_lse=return_softmax_lse,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )

    return (out, lse) if return_softmax_lse else out
