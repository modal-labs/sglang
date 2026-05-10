"""flash-kernels TileLang wrapper for GDN prefill.

`kernels.chunk_gated_delta_rule` is a hand-tuned TileLang kernel for
the gated-delta-rule (GDN) linear-attention forward pass.  On NVIDIA
B200 / GB200 (sm_100) it lowers to TCGEN5MMA — Blackwell's 5th-gen
tensor-core instruction with TMEM accumulators — and outperforms the
bundled FLA Triton chunk kernel by ~1.45x on Qwen3.5-style shapes.
The package ships from modal-projects/flash-kernels and is imported as
`from kernels import chunk_gated_delta_rule`.

flash-kernels takes its running state in the same `(B, H_v, V, K)`
layout as SGLang's `Mamba2StateShape.temporal = (num_v_heads,
head_v_dim, state_size=head_k_dim)`, so this wrapper does *not* need to
transpose between SGLang's pool layout and the kernel-internal layout.

Prefill-only.  Decode and target_verify raise NotImplementedError; the
dispatcher in `linear/gdn_backend.py` should pair this backend with
triton or flashinfer for those modes.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


_chunk_gated_delta_rule_cache: Optional[callable] = None


def _get_kernel():
    """Lazy-import flash-kernels so that import errors only surface when
    this backend is actually selected — non-Blackwell deployments
    shouldn't crash at module-import time."""
    global _chunk_gated_delta_rule_cache
    if _chunk_gated_delta_rule_cache is None:
        from kernels import chunk_gated_delta_rule

        _chunk_gated_delta_rule_cache = chunk_gated_delta_rule
    return _chunk_gated_delta_rule_cache


class FlashKernelsGDNKernel(LinearAttnKernelBase):
    """flash-kernels-backed GDN kernel (prefill only)."""

    def __init__(self):
        # flash-kernels' TileLang kernels emit sm_10x instruction
        # sequences; the dispatcher path would crash at first prefill on
        # older hardware.
        if not torch.cuda.is_available():
            raise RuntimeError("flash-kernels GDN backend requires CUDA.")
        major, _ = torch.cuda.get_device_capability(0)
        if major < 10:
            raise RuntimeError(
                f"flash-kernels GDN backend requires sm_100+ (Blackwell); "
                f"current GPU reports compute capability {major}.x. "
                f"Use --linear-attn-prefill-backend triton on older GPUs."
            )
        # Eagerly resolve the kernel symbol so import errors surface at
        # backend construction time rather than first prefill.
        _get_kernel()

    def decode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "flash-kernels GDN backend supports prefill only. Pair "
            "--linear-attn-prefill-backend flash-kernels with "
            "--linear-attn-decode-backend triton (or flashinfer)."
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        chunk_gated_delta_rule = _get_kernel()

        # Gather the active sequences' initial state out of the pool.
        # ssm_states is (N_slots, H_v, V, K); cache_indices is (B,).
        # FLA's chunk_gated_delta_rule supports an `initial_state_indices`
        # kwarg that lets the kernel gather inside; flash-kernels doesn't,
        # so we materialize a (B, H_v, V, K) tensor here.  At typical
        # shapes (B=1, H_v=64, V=K=128) this is ~512 KiB and the gather
        # is dominated by kernel time.
        initial_state = ssm_states[cache_indices]

        # Make `v` contiguous before handing it to the TileLang kernel.
        # SGLang produces q/k/v via `torch.split(mixed_qkv, [q_dim,
        # k_dim, v_dim], dim=-1)` followed by `.view(1, T, H, D)`, so
        # while `stride(-1) == 1` holds, the H-stride is the full fused
        # qkv width — TileLang requires dense-stride inputs.  q and k go
        # through `l2norm` inside `chunk_gated_delta_rule`, which
        # materializes them into fresh contiguous tensors via
        # torch.compile, so we don't need to copy those.  g and beta come
        # from fused_gdn_gating (fresh torch.empty tensors) so they're
        # already contiguous.
        v = v.contiguous()

        o, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )

        # Scatter the final state back to the pool, matching the FLA
        # Triton GPU behavior of in-place state update at cache_indices.
        # flash-kernels always returns final_state in fp32; cast to the
        # pool's dtype (typically bf16) on writeback.
        if final_state is not None:
            ssm_states[cache_indices] = final_state.to(
                ssm_states.dtype, copy=False
            )

        # Return tuple matches TritonGDNKernel.extend's contract:
        #   (core_attn_out, last_recurrent_state, h)
        # - core_attn_out: (1, T, H_v, V) prefill output
        # - last_recurrent_state=None: state was scattered above (mirrors
        #   the FLA Triton GPU path which is also None here)
        # - h=None: per-chunk states aren't surfaced.  This disables
        #   prefix caching of GDN state on this backend; SGLang's
        #   `_track_mamba_state_extend` becomes a no-op.  TODO: thread
        #   `output_h=True` through to enable it — would require calling
        #   `kernels.chunk_gated_delta_rule_fwd` (the lower-level entry
        #   point that exposes the per-chunk h tensor) instead of the
        #   high-level wrapper.
        return o, None, None

    def target_verify(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "flash-kernels GDN backend supports prefill only. Spec-decoding "
            "target_verify is handled by the dispatcher's verify_kernel "
            "(triton or flashinfer)."
        )
