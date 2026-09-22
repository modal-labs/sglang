"""A request flagged ``spec_target_degenerate`` (its verify target probs had a
row without positive finite mass) must not put its KV into the radix tree --
neither from ``cache_finished_req`` nor ``cache_unfinished_req`` -- while still
freeing its unprotected KV, releasing the tree lock it held and running every
component's ``cleanup_after_caching_req`` with ``insert_result=None``."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
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
KV_SIZE = 256


def _build_cache(page_size=1):
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
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE,
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


def _lock_ref(node):
    return node.component_data[FULL].lock_ref


class DegenerateReqSkipsInsertTest(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator, self.req_pool = _build_cache()
        self.full = self.cache._components_tuple[0]

    def _start_req(self, tokens):
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
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
        req.cache_protected_len = prefix_len
        req.swa_uuid_for_lock = None
        req.extra_key = None
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        new_indices = allocator.alloc(len(tokens) - prefix_len)
        self.assertIsNotNone(new_indices)
        pool.write((req.req_pool_idx, slice(prefix_len, len(tokens))), new_indices)
        req.kv_committed_len = len(tokens)
        req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(prefix_len, len(tokens))
        return req

    def _spy_cleanup(self, calls):
        real_cleanup = self.full.cleanup_after_caching_req

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real_cleanup(*args, **kwargs)

        return mock.patch.object(self.full, "cleanup_after_caching_req", spy)

    def _tree_len(self, tokens):
        return len(
            self.cache.match_prefix(MatchPrefixParams(key=_key(tokens))).device_indices
        )

    def test_clean_finished_req_is_inserted(self):
        tokens = list(range(1, 9))
        req = self._start_req(tokens)
        self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
        self.assertEqual(self._tree_len(tokens), len(tokens))
        self.assertEqual(self.allocator.available_size(), KV_SIZE - len(tokens))

    def test_degenerate_finished_req_frees_kv_without_insert(self):
        tokens = list(range(1, 9))
        root = self.cache.root_node
        root_lock_before = _lock_ref(root)
        req = self._start_req(tokens)
        req.mark_spec_target_degenerate()

        cleanup_calls = []
        with (
            self._spy_cleanup(cleanup_calls),
            mock.patch.object(
                self.cache, "insert", wraps=self.cache.insert
            ) as insert_spy,
            mock.patch.object(
                self.full,
                "prepare_for_caching_req",
                wraps=self.full.prepare_for_caching_req,
            ) as prepare_spy,
        ):
            self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))

        insert_spy.assert_not_called()
        prepare_spy.assert_not_called()
        self.assertEqual(len(root.children), 0)
        self.assertEqual(self._tree_len(tokens), 0)
        self.assertEqual(self.cache.evictable_size(), 0)
        self.assertEqual(self.cache.protected_size(), 0)
        self.assertEqual(_lock_ref(root), root_lock_before)
        self.assertEqual(self.allocator.available_size(), KV_SIZE)
        self.assertEqual(len(cleanup_calls), 1)
        self.assertTrue(cleanup_calls[0]["is_finished"])
        self.assertIsNone(cleanup_calls[0]["insert_result"])

    def test_degenerate_finished_req_keeps_protected_prefix(self):
        prefix = list(range(1, 5))
        first = self._start_req(prefix)
        self.cache.cache_finished_req(first, kv_len_to_handle=len(prefix))
        prefix_node = self.cache.match_prefix(
            MatchPrefixParams(key=_key(prefix))
        ).last_device_node

        tokens = prefix + list(range(101, 105))
        req = self._start_req(tokens)
        self.assertEqual(req.cache_protected_len, len(prefix))
        self.assertEqual(_lock_ref(prefix_node), 1)
        req.mark_spec_target_degenerate()

        self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))

        self.assertEqual(_lock_ref(prefix_node), 0)
        self.assertEqual(len(prefix_node.children), 0)
        self.assertEqual(self.allocator.available_size(), KV_SIZE - len(prefix))
        self.assertEqual(self._tree_len(tokens), len(prefix))

    def test_degenerate_unfinished_req_skips_insert_keeps_pages(self):
        tokens = list(range(1, 9))
        root = self.cache.root_node
        root_lock_before = _lock_ref(root)
        req = self._start_req(tokens)
        root_lock_held = _lock_ref(root)
        req.mark_spec_target_degenerate()
        own_indices = self.req_pool.req_to_token[
            req.req_pool_idx, : len(tokens)
        ].clone()

        cleanup_calls = []
        with (
            self._spy_cleanup(cleanup_calls),
            mock.patch.object(
                self.cache, "insert", wraps=self.cache.insert
            ) as insert_spy,
        ):
            self.cache.cache_unfinished_req(req)

        insert_spy.assert_not_called()
        self.assertEqual(len(root.children), 0)
        self.assertEqual(self._tree_len(tokens), 0)
        # Still decoding: its pages stay allocated and it keeps the lock it held.
        self.assertEqual(self.allocator.available_size(), KV_SIZE - len(tokens))
        self.assertEqual(_lock_ref(root), root_lock_held)
        self.assertEqual(req.prefix_indices.tolist(), own_indices.tolist())
        self.assertEqual(len(cleanup_calls), 1)
        self.assertFalse(cleanup_calls[0]["is_finished"])
        self.assertNotIn("insert_result", cleanup_calls[0])

        # The eventual finish releases everything, still without inserting.
        with mock.patch.object(
            self.cache, "insert", wraps=self.cache.insert
        ) as insert_spy:
            self.cache.cache_finished_req(req, kv_len_to_handle=len(tokens))
        insert_spy.assert_not_called()
        self.assertEqual(_lock_ref(root), root_lock_before)
        self.assertEqual(self.allocator.available_size(), KV_SIZE)


if __name__ == "__main__":
    unittest.main()
