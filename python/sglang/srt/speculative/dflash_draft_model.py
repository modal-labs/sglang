from __future__ import annotations

import logging
import os
from typing import Optional

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

logger = logging.getLogger(__name__)

# Debug flag: set DFLASH_DEBUG=1 to enable verbose logging
DFLASH_DEBUG = os.environ.get("DFLASH_DEBUG", "0") == "1"


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
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
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

        # Expand K/V for grouped-query attention (GQA)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        if DFLASH_DEBUG and self.layer_idx == 0:
            logger.info(
                "DFlash attn L0 shapes: q=%s k=%s v=%s num_kv_groups=%d",
                q.shape,
                k.shape,
                v.shape,
                self.num_key_value_groups,
            )

        # Level 1 optimization: Use SDPA instead of eager attention
        # DFlash uses non-causal attention (is_causal=False)
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
                **kwargs,
            )

        return self.norm(hidden_states)

    def make_cache(self) -> DynamicCache:
        return DynamicCache()


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
