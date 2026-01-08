"""DFlash commit-later cache with speculative scratch slots.

This module implements a cache design that enables:
1. Monotonic committed_len growth (compatible with paged KV)
2. No rollback/crop operations
3. Clear separation of committed vs speculative K/V
4. CUDA graph friendly (stable tensor addresses)

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Per-Request, Per-Layer Cache                       │
    ├─────────────────────────────────────────────────────┤
    │  K_committed: [max_committed, n_kv_heads, head_dim] │  ← Monotonic
    │  V_committed: [max_committed, n_kv_heads, head_dim] │
    │  committed_len: int                                 │
    ├─────────────────────────────────────────────────────┤
    │  K_scratch: [block_size, n_kv_heads, head_dim]      │  ← Overwritten
    │  V_scratch: [block_size, n_kv_heads, head_dim]      │
    └─────────────────────────────────────────────────────┘

Flow:
    1. Draft forward: writes K/V to scratch (never to committed)
    2. Attention: reads concat(committed[:committed_len], scratch[:q_len])
    3. After verify: project target_hidden → K/V, append to committed
    4. Repeat (scratch gets overwritten next iteration)

Key invariants:
    - committed_len is monotonically increasing
    - committed_len increases by commit_len >= 1 after each verify
    - Scratch is always overwritten, never appended
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class DraftLayerCache:
    """Per-layer committed K/V cache for a single request."""

    # Committed (verified) K/V - monotonically growing
    K_committed: Tensor  # [max_committed, n_kv_heads, head_dim]
    V_committed: Tensor  # [max_committed, n_kv_heads, head_dim]

    # Scratch for speculative tokens - overwritten each iteration
    K_scratch: Tensor  # [block_size, n_kv_heads, head_dim]
    V_scratch: Tensor  # [block_size, n_kv_heads, head_dim]

    # Current committed length (default must come last)
    committed_len: int = 0

    def get_kv_for_attention(self, scratch_len: int) -> Tuple[Tensor, Tensor]:
        """Get concatenated K/V for attention (committed + scratch).

        Args:
            scratch_len: Number of scratch tokens to include (typically q_len)

        Returns:
            K_total: [committed_len + scratch_len, n_kv_heads, head_dim]
            V_total: [committed_len + scratch_len, n_kv_heads, head_dim]
        """
        if self.committed_len == 0:
            return self.K_scratch[:scratch_len], self.V_scratch[:scratch_len]

        return (
            torch.cat(
                [self.K_committed[: self.committed_len], self.K_scratch[:scratch_len]],
                dim=0,
            ),
            torch.cat(
                [self.V_committed[: self.committed_len], self.V_scratch[:scratch_len]],
                dim=0,
            ),
        )

    def write_scratch(self, k: Tensor, v: Tensor, num_tokens: int) -> None:
        """Write K/V to scratch slots (overwrites previous values).

        Args:
            k: [num_tokens, n_kv_heads, head_dim]
            v: [num_tokens, n_kv_heads, head_dim]
            num_tokens: Number of tokens to write
        """
        self.K_scratch[:num_tokens].copy_(k)
        self.V_scratch[:num_tokens].copy_(v)

    def append_committed(self, k: Tensor, v: Tensor, num_tokens: int) -> None:
        """Append verified K/V to committed cache.

        Args:
            k: [num_tokens, n_kv_heads, head_dim] - rotary already applied
            v: [num_tokens, n_kv_heads, head_dim]
            num_tokens: Number of tokens to append

        Raises:
            RuntimeError: If append would exceed capacity
        """
        max_committed = self.K_committed.shape[0]
        new_len = self.committed_len + num_tokens

        if new_len > max_committed:
            raise RuntimeError(
                f"DraftLayerCache append would exceed capacity: "
                f"{self.committed_len} + {num_tokens} > {max_committed}"
            )

        self.K_committed[self.committed_len : new_len].copy_(k)
        self.V_committed[self.committed_len : new_len].copy_(v)
        self.committed_len = new_len


@dataclass
class DraftReqState:
    """Per-request state for DFlash commit-later cache.

    Manages the full cache lifecycle for one request across all draft layers.
    """

    req_id: int
    num_layers: int
    n_kv_heads: int
    head_dim: int
    block_size: int
    max_committed: int
    device: torch.device
    dtype: torch.dtype

    # Per-layer caches (initialized lazily or in __post_init__)
    layer_caches: List[DraftLayerCache] = field(default_factory=list)

    # Track total committed length (should be same across all layers)
    _committed_len: int = 0

    def __post_init__(self):
        """Initialize per-layer caches if not provided."""
        if not self.layer_caches:
            self.layer_caches = [
                self._make_layer_cache() for _ in range(self.num_layers)
            ]

    def _make_layer_cache(self) -> DraftLayerCache:
        """Create a new layer cache with pre-allocated buffers."""
        return DraftLayerCache(
            K_committed=torch.zeros(
                self.max_committed,
                self.n_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            ),
            V_committed=torch.zeros(
                self.max_committed,
                self.n_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            ),
            committed_len=0,
            K_scratch=torch.zeros(
                self.block_size,
                self.n_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            ),
            V_scratch=torch.zeros(
                self.block_size,
                self.n_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            ),
        )

    @property
    def committed_len(self) -> int:
        """Get committed length (consistent across all layers)."""
        return self._committed_len

    def get_attention_kv(
        self, layer_idx: int, scratch_len: int
    ) -> Tuple[Tensor, Tensor]:
        """Get K/V tensors for attention at a specific layer.

        Args:
            layer_idx: Which layer's cache to use
            scratch_len: Number of scratch tokens (typically q_len)

        Returns:
            k_total: [committed_len + scratch_len, n_kv_heads, head_dim]
            v_total: [committed_len + scratch_len, n_kv_heads, head_dim]
        """
        return self.layer_caches[layer_idx].get_kv_for_attention(scratch_len)

    def write_scratch(
        self, layer_idx: int, k: Tensor, v: Tensor, num_tokens: int
    ) -> None:
        """Write speculative K/V to scratch for a layer.

        Args:
            layer_idx: Which layer's scratch to write
            k: [num_tokens, n_kv_heads, head_dim]
            v: [num_tokens, n_kv_heads, head_dim]
            num_tokens: Number of tokens to write
        """
        self.layer_caches[layer_idx].write_scratch(k, v, num_tokens)

    def append_committed_all_layers(
        self, k_per_layer: List[Tensor], v_per_layer: List[Tensor], commit_len: int
    ) -> None:
        """Append committed K/V to all layers after verification.

        This is the core commit operation - must be called only after
        target_verify confirms which tokens are accepted.

        Args:
            k_per_layer: List of [commit_len, n_kv_heads, head_dim] per layer
            v_per_layer: List of [commit_len, n_kv_heads, head_dim] per layer
            commit_len: Number of tokens to commit (>= 1)

        Raises:
            ValueError: If commit_len < 1 (violates invariant)
        """
        if commit_len < 1:
            raise ValueError(
                f"commit_len must be >= 1 (DFlash guarantees progress), got {commit_len}"
            )

        if len(k_per_layer) != self.num_layers or len(v_per_layer) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {len(k_per_layer)} K and {len(v_per_layer)} V"
            )

        for layer_idx, (k, v) in enumerate(zip(k_per_layer, v_per_layer)):
            self.layer_caches[layer_idx].append_committed(k, v, commit_len)

        self._committed_len += commit_len

        # Verify consistency
        for lc in self.layer_caches:
            assert lc.committed_len == self._committed_len, (
                "Layer committed_len mismatch!"
            )

    def memory_bytes(self) -> int:
        """Calculate total memory usage for this request's cache."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        per_layer = (
            (self.max_committed + self.block_size)  # K + V committed + scratch
            * 2  # K and V
            * self.n_kv_heads
            * self.head_dim
            * bytes_per_element
        )
        return per_layer * self.num_layers


def make_draft_req_state(
    req_id: int,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_committed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DraftReqState:
    """Factory function to create a DraftReqState.

    Args:
        req_id: Unique request identifier
        num_layers: Number of draft model layers
        n_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        block_size: DFlash block size (scratch capacity)
        max_committed: Maximum committed tokens (context window)
        device: torch device
        dtype: tensor dtype

    Returns:
        Initialized DraftReqState with pre-allocated buffers
    """
    return DraftReqState(
        req_id=req_id,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_committed=max_committed,
        device=device,
        dtype=dtype,
    )


def compute_draft_cache_memory(
    num_requests: int,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_committed: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Compute total memory required for draft caches.

    Args:
        num_requests: Number of concurrent requests
        num_layers: Number of draft model layers
        n_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        block_size: DFlash block size
        max_committed: Maximum committed tokens per request
        dtype: tensor dtype

    Returns:
        Total memory in bytes

    Example:
        >>> # Qwen3-32B, 5-layer draft, 8 requests, 8k context
        >>> mem = compute_draft_cache_memory(
        ...     num_requests=8, num_layers=5, n_kv_heads=8,
        ...     head_dim=128, block_size=16, max_committed=8192
        ... )
        >>> print(f"{mem / 1e9:.2f} GB")  # ~1.3 GB
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    per_request_per_layer = (
        (max_committed + block_size)  # committed + scratch
        * 2  # K and V
        * n_kv_heads
        * head_dim
        * bytes_per_element
    )
    return num_requests * num_layers * per_request_per_layer
