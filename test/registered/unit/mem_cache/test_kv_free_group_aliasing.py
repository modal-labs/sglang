"""A deferred free inside a free group must not re-read a mutated view.

batch_result_processor wraps the decode-finish loop in
free_group_begin/free_group_end and hands free() slices of the request's
KV index row; the tensor can be a view that the caller mutates before the
flush. The flush must free the values the caller freed, not whatever the
view later alias. Adapted from upstream sgl-project/sglang #34067
(cfb354bcfc), test_group_owns_deferred_page_representatives, to this
fork's allocator API (no free_segment)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 4
NUM_PAGES = 8
SIZE = NUM_PAGES * PAGE_SIZE


class FreeGroupOwnsDeferredViewsTest(CustomTestCase):
    def _allocator(self):
        pool = MHATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        return PagedTokenToKVPoolAllocator(
            SIZE, PAGE_SIZE, torch.bfloat16, "cpu", pool, False
        )

    def test_group_owns_deferred_views(self):
        alloc = self._allocator()
        row = alloc.alloc(2 * PAGE_SIZE)
        expected_pages = torch.unique(row // PAGE_SIZE)

        alloc.free_group_begin()
        alloc.free(row[:PAGE_SIZE])
        alloc.free(row[PAGE_SIZE:])
        # Mutate the storage the freed views read from, as a caller holding
        # the backing row might before the flush.
        row.zero_()
        alloc.free_group_end()

        freed_pages = torch.cat((alloc.free_pages, alloc.release_pages))[
            : expected_pages.numel()
        ]
        self.assertTrue(
            torch.equal(torch.sort(freed_pages)[0], torch.sort(expected_pages)[0])
        )
        self.assertEqual(alloc.available_size(), SIZE)


if __name__ == "__main__":
    unittest.main()
