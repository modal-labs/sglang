# Adapted from https://github.com/Dao-AILab/flash-attention/blob/5d4c9537a1e0f1adcc3e4c3e11ae46fe94a18b11/flash_attn/cute/interface.py

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-10-14] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.2.1.


import copy
import gc
import logging
import os
from functools import lru_cache
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)


import cutlass
import torch
from cutlass.cute.runtime import from_dlpack
from flash_attn_origin.cute.interface import _flash_attn_fwd


@lru_cache(maxsize=None)
def _get_device_capability():
    """Cached device capability check."""
    return torch.cuda.get_device_capability()[0]


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert (
        t.shape == expected_shape
    ), f"{name} shape {t.shape} != expected {expected_shape}"
    assert (
        t.dtype == expected_dtype
    ), f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert (
        t.device == expected_device
    ), f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


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
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    return_softmax_lse: Optional[bool] = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = _flash_attn_fwd(
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
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )

    return (out, lse) if return_softmax_lse else out
