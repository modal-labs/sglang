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
import logging
import traceback
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class BaseTokenToKVPoolAllocator(abc.ABC):
    debug_mode: bool = False

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
        # Locations inside the reserved padding page that reached free() and
        # were dropped instead of recycled. A nonzero count means some caller
        # leaked a reserved location; the first occurrence logs its stack.
        self.reserved_location_drops = 0
        # Per-page allocated state indexed by page id (slot 0 is the reserved
        # page and stays False). Set on every hand-out, cleared on free, so a
        # free of a page that is not currently allocated (a cross-call double
        # free) can be dropped instead of putting the page in the pool twice.
        self.page_allocated: Optional[torch.Tensor] = None
        # Pages dropped by free() because they were not allocated. Accumulated
        # on the device so the free path adds no host sync; the scheduler's
        # idle tick copies it into the host mirror `double_free_page_drops`
        # (see `refresh_double_free_page_drops`), which stats readers use.
        self._double_free_page_drops_dev = torch.zeros(
            (), dtype=torch.int64, device=device
        )
        self.double_free_page_drops = 0
        self._double_free_logged = False
        # Locations dropped by free() because they lie past the end of the
        # pool. Kept apart from the double-free count (and from the
        # reserved-location count below the pool) because each points at a
        # different upstream bug. Same device accumulator / idle mirror scheme.
        self._out_of_pool_location_drops_dev = torch.zeros(
            (), dtype=torch.int64, device=device
        )
        self.out_of_pool_location_drops = 0
        self._out_of_pool_logged = False
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

    def _drop_reserved_locations(
        self, free_index: torch.Tensor, lower_bound: int
    ) -> torch.Tensor:
        """Return `free_index` without the locations below `lower_bound`.

        The reserved padding page is never handed out, so a location inside it
        can only reach free() through a caller bug; recycling it would alias
        the padding sink with a live request. On CUDA the boolean gather has a
        dynamic output shape and therefore costs one host synchronization per
        call; the drop tally reads the kept count that gather already brought
        to the host and adds no second one.
        """
        kept = free_index[free_index >= lower_bound]
        dropped = free_index.numel() - kept.numel()
        if dropped:
            self.reserved_location_drops += dropped
            if self.reserved_location_drops == dropped:
                logger.warning(
                    "%s.free() dropped %d location(s) below %d (reserved page); "
                    "caller stack:\n%s",
                    type(self).__name__,
                    dropped,
                    lower_bound,
                    "".join(traceback.format_stack(limit=12)[:-1]),
                )
        return kept

    def reserved_location_drops_total(self) -> int:
        """Reserved-page locations dropped by this allocator and its children."""
        return self.reserved_location_drops

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages, self.page_allocated = state

    def backup_state(self):
        # page_allocated is mutated in place, so the snapshot must be a copy.
        bitmap = self.page_allocated
        return (
            self.free_pages,
            self.release_pages,
            None if bitmap is None else bitmap.clone(),
        )

    def _reset_page_allocated(self, num_pages: int) -> None:
        self.page_allocated = torch.zeros(
            (num_pages + 1,), dtype=torch.bool, device=self.device
        )
        self._drop_scratch = torch.zeros_like(self.page_allocated)
        self._zero_page = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _mark_allocated(self, pages: torch.Tensor) -> None:
        self.page_allocated[pages] = True

    def _mask_unallocated_pages(self, pages: torch.Tensor) -> torch.Tensor:
        """Route pages that are not currently allocated to the reserved id 0.

        `pages` is the page id of every location being freed, already stripped
        of reserved locations, so a page that is not allocated can only be a
        cross-call double free. Such pages are rewritten to 0 (never recycled)
        and counted on the device per distinct page; nothing here syncs with
        the host. Callers then dedup with `_unique_pages`, which drops the 0.

        A page id past the end of the pool cannot be allocated, so it is folded
        onto 0 before the gather and dropped the same way, but tallied per
        location in `_out_of_pool_location_drops_dev` instead.
        """
        in_pool = pages < self.page_allocated.numel()
        pages = torch.where(in_pool, pages, 0)
        allocated = self.page_allocated[pages]
        # Every location of one page carries the same value, so the scatter is
        # deterministic and the sum counts each dropped page once. Out-of-pool
        # locations all land on index 0 and only ever write False there.
        self._drop_scratch[pages] = ~allocated & in_pool
        self._double_free_page_drops_dev += self._drop_scratch.sum()
        self._drop_scratch[pages] = False
        self._out_of_pool_location_drops_dev += (~in_pool).sum()
        if self.debug_mode:
            double_freed = int((~allocated & in_pool).sum())
            if double_freed:
                self._log_double_free_once(double_freed, stack=True)
        return torch.where(allocated, pages, 0)

    def _unique_pages(self, pages: torch.Tensor) -> torch.Tensor:
        """Sorted distinct page ids of `pages` without the reserved id 0.

        The 0 is always appended, so it sorts first and a fixed `[1:]` slice
        removes it (and every page masked to 0) with the single host sync that
        `torch.unique` already pays for its output size.
        """
        return torch.unique(torch.cat((pages, self._zero_page)))[1:]

    def _free_pages_of(self, free_index: torch.Tensor) -> torch.Tensor:
        """Distinct allocated pages of `free_index` (reserved locations already
        dropped), with their allocated bit cleared."""
        pages = self._unique_pages(
            self._mask_unallocated_pages(free_index // self.page_size)
        )
        self.page_allocated[pages] = False
        return pages

    def _log_double_free_once(self, count: int, stack: bool) -> None:
        if self._double_free_logged:
            return
        self._double_free_logged = True
        logger.warning(
            "%s.free() dropped %d page(s) that were not allocated (cross-call "
            "double free); later drops are only counted in "
            "double_free_page_drops%s",
            type(self).__name__,
            count,
            (
                "; caller stack:\n" + "".join(traceback.format_stack(limit=12)[:-2])
                if stack
                else " (set SGLANG_DEBUG_MEMORY_POOL=1 for the caller stack)"
            ),
        )

    def refresh_double_free_page_drops(self) -> int:
        """Copy the device drop accumulators into `double_free_page_drops` and
        `out_of_pool_location_drops`.

        This is the one host sync of the guard; call it from the scheduler's
        idle tick, never from the forward path.
        """
        drops = int(self._double_free_page_drops_dev.item())
        if drops:
            self._log_double_free_once(drops, stack=False)
        self.double_free_page_drops = drops
        out_of_pool = int(self._out_of_pool_location_drops_dev.item())
        if out_of_pool and not self._out_of_pool_logged:
            self._out_of_pool_logged = True
            logger.warning(
                "%s.free() dropped %d location(s) past the end of the pool "
                "(%d pages); later drops are only counted in "
                "out_of_pool_location_drops",
                type(self).__name__,
                out_of_pool,
                self.page_allocated.numel() - 1,
            )
        self.out_of_pool_location_drops = out_of_pool
        return drops

    def double_free_page_drops_total(self, refresh: bool = False) -> int:
        """Double-freed pages dropped by this allocator and its children, as of
        the last refresh (or now, with `refresh=True`, which syncs)."""
        if refresh:
            self.refresh_double_free_page_drops()
        return self.double_free_page_drops

    def out_of_pool_location_drops_total(self, refresh: bool = False) -> int:
        """Past-the-end locations dropped by this allocator and its children,
        as of the last refresh (or now, with `refresh=True`, which syncs)."""
        if refresh:
            self.refresh_double_free_page_drops()
        return self.out_of_pool_location_drops

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

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
