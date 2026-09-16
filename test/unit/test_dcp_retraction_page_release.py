"""CPU regression for logical-page ownership during immediate PD retraction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache


class RequestPool:
    def __init__(self, indices):
        self.req_to_token = indices.unsqueeze(0)

    def free(self, req):
        req.req_pool_idx = None


class DCPRetractionPageReleaseTest(unittest.TestCase):
    def check_release(
        self, physical_page, dcp, committed, allocated, grouped, need_sort
    ):
        logical_page = physical_page * dcp
        allocator = PagedTokenToKVPoolAllocator(
            size=16 * logical_page,
            page_size=logical_page,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )
        # Non-contiguous backing pages ensure the test checks page identity,
        # rather than relying on a contiguous request-to-token mapping.
        allocator.free_pages = allocator.free_pages.flip(0)
        server_args = SimpleNamespace(
            page_size=physical_page,
            speculative_algorithm="DFLASH",
            strip_thinking_cache=False,
        )
        for _ in range(3):
            indices = allocator.alloc(allocated)
            if indices is None:
                allocator.merge_and_sort_free()
                indices = allocator.alloc(allocated)
            self.assertIsNotNone(indices)
            pool = RequestPool(indices)
            cache = ChunkCache(
                SimpleNamespace(
                    req_to_token_pool=pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=logical_page,
                )
            )
            req = SimpleNamespace(
                req_pool_idx=0,
                kv=SimpleNamespace(kv_allocated_len=allocated),
                kv_committed_len=committed,
                cache_protected_len=0,
                effective_kv_committed_len=lambda: committed,
            )
            if grouped:
                allocator.free_group_begin()
            with patch(
                "sglang.srt.mem_cache.common.get_server_args", return_value=server_args
            ):
                release_kv_cache(req, cache, is_insert=False)
            if grouped:
                allocator.free_group_end()
            self.assertEqual(allocator.available_size(), allocator.size)
            free_pages = torch.cat((allocator.free_pages, allocator.release_pages))
            self.assertEqual(free_pages.unique().numel(), free_pages.numel())
            self.assertIsNone(req.req_pool_idx)
            self.assertIsNone(req.kv)

    def test_immediate_dcp_release_does_not_free_boundary_page_twice(self):
        self.check_release(64, 8, 513, 1024, False, False)

    def test_release_boundaries_and_allocator_reuse(self):
        for physical_page, dcp in ((1, 1), (64, 1), (64, 2), (64, 8)):
            page = physical_page * dcp
            for committed in sorted({0, 1, physical_page, page - 1, page, page + 1}):
                for grouped in (False, True):
                    for need_sort in (False, True):
                        with self.subTest(
                            physical_page=physical_page,
                            dcp=dcp,
                            committed=committed,
                            grouped=grouped,
                            need_sort=need_sort,
                        ):
                            self.check_release(
                                physical_page,
                                dcp,
                                committed,
                                3 * page,
                                grouped,
                                need_sort,
                            )


if __name__ == "__main__":
    unittest.main()
