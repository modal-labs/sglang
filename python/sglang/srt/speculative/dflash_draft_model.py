from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
from safetensors.torch import safe_open
from torch import nn
from transformers import AutoConfig, DynamicCache
from transformers.cache_utils import Cache
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    FlashAttentionKwargs,
    rotate_half,
)
from typing_extensions import Unpack

from sglang.srt.speculative.dflash_cache import DraftReqState

logger = logging.getLogger(__name__)

# Debug flag: set DFLASH_DEBUG=1 to enable verbose logging
DFLASH_DEBUG = os.environ.get("DFLASH_DEBUG", "0") == "1"

# Attention backend selection:
# DFLASH_ATTN_BACKEND: "sdpa" (default), "flashinfer", or "fa3"
# FA3 = FlashAttention 3, optimized for Hopper (H100/H200)
DFLASH_ATTN_BACKEND = os.environ.get("DFLASH_ATTN_BACKEND", "sdpa").lower()

# Legacy env var support
if os.environ.get("DFLASH_USE_FLASHINFER", "0") == "1":
    DFLASH_ATTN_BACKEND = "flashinfer"

# Check FlashInfer availability
try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    if DFLASH_ATTN_BACKEND == "flashinfer":
        logger.warning(
            "FlashInfer not available, falling back to SDPA for DFlash attention"
        )
        DFLASH_ATTN_BACKEND = "sdpa"

# Check FlashAttention 3 availability
try:
    from flash_attn import flash_attn_func

    FA3_AVAILABLE = True
except ImportError:
    FA3_AVAILABLE = False
    if DFLASH_ATTN_BACKEND == "fa3":
        logger.warning(
            "FlashAttention 3 not available, falling back to SDPA for DFlash attention"
        )
        DFLASH_ATTN_BACKEND = "sdpa"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped-query attention (GQA).

    Input shape: [batch, num_kv_heads, seq_len, head_dim]
    Output shape: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_qo_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        flashinfer_wrapper=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        if DFLASH_DEBUG and self.layer_idx == 0:
            logger.info(
                "DFlash attn L0: bsz=%d q_len=%d ctx_len=%d hidden=%s target=%s",
                bsz,
                q_len,
                ctx_len,
                hidden_states.shape,
                target_hidden.shape,
            )

        # Q projection (only from noise/hidden_states)
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # Level 2 optimization: Combine inputs before K/V projection
        # Reduces 4 linear ops (k_proj×2, v_proj×2) + 2 cats → 1 cat + 2 linear ops
        kv_input = torch.cat(
            [target_hidden, hidden_states], dim=1
        )  # [bsz, ctx_len + q_len, hidden]
        kv_len = ctx_len + q_len
        k = self.k_proj(kv_input).view(bsz, kv_len, -1, self.head_dim)
        v = self.v_proj(kv_input).view(bsz, kv_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        if DFLASH_DEBUG and self.layer_idx == 0:
            logger.info(
                "DFlash attn L0 shapes: q=%s k=%s v=%s num_kv_groups=%d",
                q.shape,
                k.shape,
                v.shape,
                self.num_key_value_groups,
            )

        # Select attention backend based on DFLASH_ATTN_BACKEND
        use_flashinfer = (
            DFLASH_ATTN_BACKEND == "flashinfer"
            and flashinfer_wrapper is not None
            and FLASHINFER_AVAILABLE
        )
        use_fa3 = DFLASH_ATTN_BACKEND == "fa3" and FA3_AVAILABLE

        if use_flashinfer:
            # FlashInfer expects [total_tokens, num_heads, head_dim]
            # q: [bsz, num_qo_heads, q_len, head_dim] -> [bsz * q_len, num_qo_heads, head_dim]
            # k: [bsz, num_kv_heads, kv_len, head_dim] -> [bsz * kv_len, num_kv_heads, head_dim]
            q_fi = q.transpose(1, 2).reshape(-1, self.num_qo_heads, self.head_dim)
            k_fi = k.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
            v_fi = v.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)

            attn_output = flashinfer_wrapper.forward(
                q_fi,
                k_fi,
                v_fi,
                causal=False,  # DFlash uses non-causal attention
                sm_scale=self.scaling,
            )
            # Output: [bsz * q_len, num_qo_heads, head_dim] -> [bsz, q_len, hidden_size]
            attn_output = attn_output.view(bsz, q_len, -1)
        elif use_fa3:
            # FlashAttention 3 expects [bsz, seq_len, num_heads, head_dim]
            # Current shape: [bsz, num_heads, seq_len, head_dim]
            q_fa = q.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, head_dim]
            k_fa = k.transpose(
                1, 2
            ).contiguous()  # [bsz, kv_len, num_kv_heads, head_dim]
            v_fa = v.transpose(
                1, 2
            ).contiguous()  # [bsz, kv_len, num_kv_heads, head_dim]

            attn_output = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                causal=False,  # DFlash uses non-causal attention
                softmax_scale=self.scaling,
            )
            # Output: [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, hidden_size]
            attn_output = attn_output.reshape(bsz, q_len, -1)
        else:
            # Fallback to SDPA - need to expand K/V for GQA
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            dropout_p = 0.0 if not self.training else self.attention_dropout
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.scaling,
            )
            # SDPA output: [bsz, num_heads, q_len, head_dim] -> [bsz, q_len, num_heads * head_dim]
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        flashinfer_wrapper=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            flashinfer_wrapper=flashinfer_wrapper,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(nn.Module):
    """Local (non-trust_remote_code) DFlash draft model implementation.

    This is adapted from the DFlash reference `modeling_dflash.py` shipped with
    the draft checkpoint, but is loaded as first-party code in SGLang.

    The model intentionally does NOT include embedding or lm_head weights; the
    DFlash algorithm uses the target model's embedding and lm_head.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # DFlash context feature projection: concat(draft_num_layers x hidden_size) -> hidden_size.
        self.fc = nn.Linear(
            config.num_hidden_layers * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.block_size = config.block_size

        # FlashInfer ragged wrapper (initialized lazily on first forward)
        self._flashinfer_workspace = None
        self._flashinfer_wrapper = None
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

    def _init_flashinfer(self, device: torch.device, dtype: torch.dtype):
        """Initialize FlashInfer workspace and wrapper."""
        if not FLASHINFER_AVAILABLE or DFLASH_ATTN_BACKEND != "flashinfer":
            return

        if self._flashinfer_wrapper is not None:
            return

        # Workspace buffer size (128MB should be enough for most cases)
        workspace_size = 128 * 1024 * 1024
        self._flashinfer_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device=device
        )
        self._flashinfer_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self._flashinfer_workspace, "NHD"
        )
        logger.info(
            "DFlash: Initialized FlashInfer ragged wrapper (workspace=%dMB)",
            workspace_size // (1024 * 1024),
        )

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        bsz = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        ctx_len = target_hidden.shape[1]

        # Initialize FlashInfer on first forward
        self._init_flashinfer(hidden_states.device, hidden_states.dtype)

        # Setup FlashInfer wrapper for this forward pass
        flashinfer_wrapper = None
        if (
            DFLASH_ATTN_BACKEND == "flashinfer"
            and self._flashinfer_wrapper is not None
            and FLASHINFER_AVAILABLE
        ):
            # Compute KV length after cache update
            # If past_key_values exists and has content, kv_len = cache_len + ctx_len + q_len
            # Otherwise, kv_len = ctx_len + q_len
            # DynamicCache uses get_seq_length() method
            if past_key_values is not None:
                cache_len = past_key_values.get_seq_length()
            else:
                cache_len = 0
            total_kv_len = cache_len + ctx_len + q_len

            # Build indptr arrays for ragged batching
            # For bsz=1: qo_indptr = [0, q_len], kv_indptr = [0, total_kv_len]
            qo_indptr = torch.tensor(
                [i * q_len for i in range(bsz + 1)],
                dtype=torch.int32,
                device=hidden_states.device,
            )
            kv_indptr = torch.tensor(
                [i * total_kv_len for i in range(bsz + 1)],
                dtype=torch.int32,
                device=hidden_states.device,
            )

            # Plan the wrapper (can be reused across layers)
            self._flashinfer_wrapper.begin_forward(
                qo_indptr,
                kv_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=hidden_states.dtype,
            )
            flashinfer_wrapper = self._flashinfer_wrapper

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                flashinfer_wrapper=flashinfer_wrapper,
                **kwargs,
            )

        # End forward for FlashInfer wrapper
        if flashinfer_wrapper is not None:
            self._flashinfer_wrapper.end_forward()

        return self.norm(hidden_states)

    def make_cache(self) -> DynamicCache:
        return DynamicCache()

    def make_req_state(
        self,
        req_id: int,
        max_committed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DraftReqState:
        """Create a commit-later cache state for a request.

        Args:
            req_id: Unique request identifier
            max_committed: Maximum committed tokens (context window)
            device: torch device
            dtype: tensor dtype

        Returns:
            DraftReqState with pre-allocated buffers
        """
        return DraftReqState(
            req_id=req_id,
            num_layers=self.config.num_hidden_layers,
            n_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_committed=max_committed,
            device=device,
            dtype=dtype,
        )

    def forward_commit_later(
        self,
        *,
        noise_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        req_state: DraftReqState,
    ) -> torch.Tensor:
        """Forward pass using commit-later cache (no DynamicCache, no crop).

        This is the commit-later version of forward() that:
        1. Reads committed K/V from req_state's persistent cache
        2. Writes noise K/V to scratch (overwritten each iteration)
        3. Never modifies committed_len (that happens after verify)

        Args:
            noise_embedding: [1, block_size, hidden] - noise token embeddings
            position_ids: [1, committed_len + block_size] - full position ids
            req_state: DraftReqState with committed cache + scratch

        Returns:
            hidden_states: [1, block_size, hidden] - output for lm_head
        """
        bsz, q_len, _ = noise_embedding.shape
        assert bsz == 1, "Commit-later currently only supports bsz=1"

        hidden_states = noise_embedding
        committed_len = req_state.committed_len

        # Initialize FlashInfer on first forward
        self._init_flashinfer(hidden_states.device, hidden_states.dtype)

        # Setup FlashInfer wrapper for this forward pass
        total_kv_len = committed_len + q_len
        flashinfer_wrapper = None
        if (
            DFLASH_ATTN_BACKEND == "flashinfer"
            and self._flashinfer_wrapper is not None
            and FLASHINFER_AVAILABLE
        ):
            qo_indptr = torch.tensor(
                [0, q_len], dtype=torch.int32, device=hidden_states.device
            )
            kv_indptr = torch.tensor(
                [0, total_kv_len], dtype=torch.int32, device=hidden_states.device
            )
            self._flashinfer_wrapper.begin_forward(
                qo_indptr,
                kv_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=hidden_states.dtype,
            )
            flashinfer_wrapper = self._flashinfer_wrapper

        # Position embeddings for noise tokens only
        # We need positions [committed_len, committed_len + q_len)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = self._forward_layer_commit_later(
                layer=layer,
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                req_state=req_state,
                position_embeddings=position_embeddings,
                flashinfer_wrapper=flashinfer_wrapper,
            )

        if flashinfer_wrapper is not None:
            self._flashinfer_wrapper.end_forward()

        return self.norm(hidden_states)

    def _forward_layer_commit_later(
        self,
        layer: Qwen3DFlashDecoderLayer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        req_state: DraftReqState,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        flashinfer_wrapper,
    ) -> torch.Tensor:
        """Forward one layer using commit-later cache.

        Key difference from original: K/V for context comes from committed cache,
        not from projecting target_hidden each time.
        """
        bsz, q_len, _ = hidden_states.shape
        attn = layer.self_attn

        # Residual connection
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # Q projection (only from noise/hidden_states)
        # Shape: [bsz, q_len, hidden] -> [bsz, num_heads, q_len, head_dim]
        q = attn.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, attn.head_dim)
        q = attn.q_norm(q).transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]

        # K/V projection for NOISE ONLY (not target_hidden)
        # Shape: [bsz, q_len, hidden] -> [bsz, n_kv_heads, q_len, head_dim]
        k_noise = attn.k_proj(hidden_states).view(bsz, q_len, -1, attn.head_dim)
        v_noise = attn.v_proj(hidden_states).view(bsz, q_len, -1, attn.head_dim)
        k_noise = attn.k_norm(k_noise).transpose(
            1, 2
        )  # [bsz, n_kv_heads, q_len, head_dim]
        v_noise = v_noise.transpose(1, 2)

        # Apply rotary to Q and K_noise using same method as original attention
        cos, sin = position_embeddings
        q, k_noise = apply_rotary_pos_emb(q, k_noise, cos, sin)

        # Write noise K/V to scratch [q_len, n_kv_heads, head_dim]
        # Transpose back: [bsz, n_kv_heads, q_len, head_dim] -> [q_len, n_kv_heads, head_dim]
        k_noise_flat = k_noise.squeeze(0).transpose(
            0, 1
        )  # [q_len, n_kv_heads, head_dim]
        v_noise_flat = v_noise.squeeze(0).transpose(
            0, 1
        )  # [q_len, n_kv_heads, head_dim]
        req_state.write_scratch(layer_idx, k_noise_flat, v_noise_flat, q_len)

        # Get combined K/V for attention: committed + scratch
        k_total, v_total = req_state.get_attention_kv(layer_idx, q_len)
        # k_total: [committed_len + q_len, n_kv_heads, head_dim]

        # Run attention based on selected backend
        use_flashinfer = (
            DFLASH_ATTN_BACKEND == "flashinfer"
            and flashinfer_wrapper is not None
            and FLASHINFER_AVAILABLE
        )
        use_fa3 = DFLASH_ATTN_BACKEND == "fa3" and FA3_AVAILABLE

        if use_flashinfer:
            # FlashInfer expects [total_tokens, num_heads, head_dim]
            q_fi = q.transpose(1, 2).reshape(-1, attn.num_qo_heads, attn.head_dim)
            # k_total/v_total already [kv_len, n_kv_heads, head_dim]
            k_fi = k_total
            v_fi = v_total

            attn_output = flashinfer_wrapper.forward(
                q_fi, k_fi, v_fi, causal=False, sm_scale=attn.scaling
            )
            attn_output = attn_output.view(bsz, q_len, -1)
        elif use_fa3:
            # FlashAttention 3 expects [bsz, seq_len, num_heads, head_dim]
            # q is [bsz, num_heads, q_len, head_dim]
            q_fa = q.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, head_dim]

            # k_total/v_total are [kv_len, n_kv_heads, head_dim]
            # -> [bsz, kv_len, n_kv_heads, head_dim]
            k_fa = k_total.unsqueeze(0)
            v_fa = v_total.unsqueeze(0)

            attn_output = flash_attn_func(
                q_fa, k_fa, v_fa, causal=False, softmax_scale=attn.scaling
            )
            attn_output = attn_output.reshape(bsz, q_len, -1)
        else:
            # SDPA path - need [bsz, num_heads, seq_len, head_dim]
            # q is already [bsz, num_heads, q_len, head_dim] after rotary
            q_sdpa = q

            # Expand K/V for SDPA: [kv_len, n_kv_heads, head_dim] -> [bsz, n_kv_heads, kv_len, head_dim]
            k_sdpa = k_total.unsqueeze(0).transpose(1, 2)
            v_sdpa = v_total.unsqueeze(0).transpose(1, 2)

            # GQA expansion
            k_sdpa = repeat_kv(k_sdpa, attn.num_key_value_groups)
            v_sdpa = repeat_kv(v_sdpa, attn.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=attn.scaling,
            )
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

        attn_output = attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def project_target_hidden_to_kv(
        self,
        target_hidden: torch.Tensor,
        position_start: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Project target hidden states to K/V for appending to committed cache.

        This is called AFTER verify to convert the verified target_hidden
        into K/V that will be appended to the committed cache.

        Args:
            target_hidden: [commit_len, hidden_size * num_layers] - from target model
            position_start: Position offset for rotary (= current committed_len)

        Returns:
            k_per_layer: List of [commit_len, n_kv_heads, head_dim] per layer
            v_per_layer: List of [commit_len, n_kv_heads, head_dim] per layer
        """
        commit_len = target_hidden.shape[0]

        # First, apply the FC projection and norm (same as original DFlash)
        # target_hidden: [commit_len, hidden * num_layers] -> [commit_len, hidden]
        ctx_feat = self.hidden_norm(self.fc(target_hidden))

        # Build position embeddings for these tokens
        position_ids = torch.arange(
            position_start,
            position_start + commit_len,
            device=target_hidden.device,
            dtype=torch.long,
        ).unsqueeze(0)  # [1, commit_len]

        # Get rotary embeddings
        # We need a dummy hidden for rotary_emb shape
        dummy = ctx_feat.unsqueeze(0)  # [1, commit_len, hidden]
        cos, sin = self.rotary_emb(dummy, position_ids)

        k_per_layer: List[torch.Tensor] = []
        v_per_layer: List[torch.Tensor] = []

        for layer in self.layers:
            attn = layer.self_attn
            # Project [commit_len, hidden] -> [1, n_kv_heads, commit_len, head_dim]
            k = attn.k_proj(ctx_feat).view(
                1, commit_len, attn.num_kv_heads, attn.head_dim
            )
            v = attn.v_proj(ctx_feat).view(
                1, commit_len, attn.num_kv_heads, attn.head_dim
            )
            k = attn.k_norm(k).transpose(1, 2)  # [1, n_kv_heads, commit_len, head_dim]
            v = v.transpose(1, 2)

            # Apply rotary to K using same method as attention
            # cos/sin from rotary_emb, apply_rotary_pos_emb expects 4D tensors
            _, k = apply_rotary_pos_emb(k, k, cos, sin)  # dummy q, we only need k

            # Reshape to [commit_len, n_kv_heads, head_dim]
            k = k.squeeze(0).transpose(0, 1)  # [commit_len, n_kv_heads, head_dim]
            v = v.squeeze(0).transpose(0, 1)

            k_per_layer.append(k)
            v_per_layer.append(v)

        return k_per_layer, v_per_layer


def load_dflash_draft_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[DFlashDraftModel, object]:
    """Load DFlash draft model weights from a local folder."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    # Set for compatibility with other code that might check this attribute.
    # We use F.scaled_dot_product_attention directly in Qwen3DFlashAttention.
    setattr(config, "_attn_implementation", "sdpa")

    model = DFlashDraftModel(config).to(device=device, dtype=dtype)

    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DFLASH draft weights not found: {weights_path}")

    model_state = model.state_dict()
    unexpected: list[str] = []
    with safe_open(weights_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            if key not in model_state:
                unexpected.append(key)
                continue
            model_state[key].copy_(f.get_tensor(key))

    if unexpected:
        logger.warning(
            "DFLASH draft checkpoint has %d unexpected keys (ignored). Example: %s",
            len(unexpected),
            unexpected[0],
        )

    model.eval()
    model.requires_grad_(False)
    return model, config
