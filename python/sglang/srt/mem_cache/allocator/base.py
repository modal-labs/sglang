"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        # Paged allocation kernels need a compile-time width for their prefix
        # reductions.  Once the request pool has been sized, the configurator
        # installs one capacity-derived width here so different serving batch
        # sizes do not create distinct Triton artifacts.
        self._triton_batch_size_upper_bound = None

    def set_triton_batch_size_upper_bound(self, max_batch_size: int) -> None:
        """Use one config-derived Triton reduction width for this allocator.

        Composite allocators delegate alloc_extend/alloc_decode to child
        allocators, so propagate the bound through the small set of allocator
        child surfaces as well. This is a compile-shape bound only; the runtime
        grid remains the actual batch size.
        """
        max_batch_size = int(max_batch_size)
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        new_bound = 1 << (max_batch_size - 1).bit_length()
        # A draft worker may share the target allocator and configure it later.
        # Never shrink a bound that another runner in the process already owns.
        self._triton_batch_size_upper_bound = max(
            self._triton_batch_size_upper_bound or 0,
            new_bound,
        )

        for child_name in (
            "full_attn_allocator",
            "swa_attn_allocator",
            "mamba_allocator",
            "logical_attn_allocator",
            "hisparse_attn_allocator",
        ):
            child = getattr(self, child_name, None)
            if (
                child is not None
                and child is not self
                and hasattr(child, "set_triton_batch_size_upper_bound")
            ):
                child.set_triton_batch_size_upper_bound(max_batch_size)

    def triton_batch_size_upper_bound(self, batch_size: int) -> int:
        """Return the stable configured width, or preserve legacy fallback."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        bound = self._triton_batch_size_upper_bound
        if bound is None:
            return 1 << (batch_size - 1).bit_length()
        if batch_size > bound:
            raise RuntimeError(
                "runtime batch exceeds the configured Triton allocation bound: "
                f"batch_size={batch_size}, bound={bound}"
            )
        return bound

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
