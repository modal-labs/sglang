"""``UnifiedRadixCache.cache_finished_req`` with a component-truncated effective
cache length issues two frees that meet at ``effective_cache_len``: the
truncated span ``[effective_cache_len:]`` and, after the insert, the unaligned
tail ``[page_aligned_len:effective_cache_len]``. The two are disjoint only if
``effective_cache_len`` is a page multiple; otherwise they share a page and
every prefill finish would double-free it. The free sites pin that alignment
with an assert."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

FULL = ComponentType.FULL
PAGE_SIZE = 4
KV_SIZE = 64 * PAGE_SIZE


def _build_cache(page_size=PAGE_SIZE):
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", page_size=page_size)
    )
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=KV_SIZE,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=16,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=KV_SIZE,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        disable=False,
        tree_components=(FULL,),
        eviction_policy="lru",
    )
    return UnifiedRadixCache(params=params), allocator, req_to_token_pool


def _key(tokens):
    return RadixKey(array("q", tokens))


class FinishedReqEffectiveCacheLenAlignmentTest(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator, self.req_pool = _build_cache()
        self.full = self.cache._components_tuple[0]

    def _start_req(self, tokens):
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(match.device_indices), 0)
        cache.inc_lock_ref(match.last_device_node)
        req = Req(
            rid="0",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        pool.alloc([req])
        req.output_ids = array("q")
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        num_pages = -(-len(tokens) // PAGE_SIZE)
        new_indices = allocator.alloc(num_pages * PAGE_SIZE)
        self.assertIsNotNone(new_indices)
        pool.write(
            (req.req_pool_idx, slice(0, len(tokens))), new_indices[: len(tokens)]
        )
        # The request's last page is partially filled; the surplus slots of
        # that page are, as in the scheduler, freed with the request.
        req.kv_committed_len = len(tokens)
        req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(tokens))
        return req, new_indices

    def test_aligned_truncation_frees_each_page_once(self):
        tokens = list(range(1, 11))  # 10 tokens over 3 pages
        req, _ = self._start_req(tokens)
        with mock.patch.object(
            self.full, "prepare_for_caching_req", return_value=2 * PAGE_SIZE
        ):
            self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
        cached = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(cached.device_indices), 2 * PAGE_SIZE)
        # Two pages cached, the third page (tokens 8..10) freed exactly once.
        self.assertEqual(self.allocator.available_size(), KV_SIZE - 2 * PAGE_SIZE)
        self.assertEqual(
            len(torch.unique(self.allocator.free_pages)),
            len(self.allocator.free_pages),
        )
        self.assertEqual(self.allocator.double_free_page_drops_total(refresh=True), 0)

    def test_unaligned_truncation_is_rejected_before_the_first_free(self):
        tokens = list(range(1, 11))
        req, _ = self._start_req(tokens)
        available_before = self.allocator.available_size()
        with (
            mock.patch.object(
                self.full, "prepare_for_caching_req", return_value=PAGE_SIZE + 2
            ),
            mock.patch.object(
                self.allocator, "free", wraps=self.allocator.free
            ) as free_spy,
        ):
            with self.assertRaisesRegex(AssertionError, "not a multiple of page_size"):
                self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
        free_spy.assert_not_called()
        self.assertEqual(self.allocator.available_size(), available_before)

    def test_no_truncation_keeps_the_unaligned_tail_path(self):
        tokens = list(range(1, 11))
        req, _ = self._start_req(tokens)
        self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
        cached = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(cached.device_indices), 2 * PAGE_SIZE)
        self.assertEqual(self.allocator.available_size(), KV_SIZE - 2 * PAGE_SIZE)
        self.assertEqual(self.allocator.double_free_page_drops_total(refresh=True), 0)


if __name__ == "__main__":
    unittest.main()
