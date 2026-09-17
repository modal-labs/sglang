"""Explicit eviction signal (``evict_on_finish``) on the unified radix cache.

A client that knows a request is the last turn of its trajectory sets
``evict_on_finish``. The request's KV is never inserted, and the tree frees
the request's private prefix chain on finish instead of leaving it as the
most recently used entry. Nodes that stay in the tree (backed up to host, or
pinned by a lock) are marked ``evict_first`` so device/host eviction takes
them before any live entry; a later prefix match clears the mark.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.session_controller import Session

FULL = ComponentType.FULL
KV_SIZE = 256


def _build_cache():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=KV_SIZE,
        page_size=1,
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
        page_size=1,
        disable=False,
        tree_components=(FULL,),
        eviction_policy="lru",
    )
    return UnifiedRadixCache(params=params), allocator, req_to_token_pool


def _key(tokens):
    return RadixKey(array("q", tokens))


class EvictOnFinishTest(unittest.TestCase):
    def setUp(self):
        self.cache, self.allocator, self.req_pool = _build_cache()
        self._rid = 0

    # -- helpers -------------------------------------------------------------

    def _finish(self, tokens, *, evict_on_finish=False, extra_lock=False):
        """Run one request over ``tokens`` the way the scheduler does: match
        the prefix, take the tree lock, allocate the remainder, then release
        through the finish path. Returns the request."""
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
        cache.inc_lock_ref(match.last_device_node)
        if extra_lock:
            # A second running request shares this node.
            cache.inc_lock_ref(match.last_device_node)

        req = Req(
            rid=str(self._rid),
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            evict_on_finish=evict_on_finish,
        )
        self._rid += 1
        pool.alloc([req])
        req.output_ids = array("q")
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.cache_protected_len = prefix_len
        req.swa_uuid_for_lock = None
        req.extra_key = None
        new_len = len(tokens) - prefix_len
        new_indices = allocator.alloc(new_len)
        self.assertIsNotNone(new_indices)
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        pool.write((req.req_pool_idx, slice(prefix_len, len(tokens))), new_indices)
        req.kv_committed_len = len(tokens)
        req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(prefix_len, len(tokens))

        release_kv_cache(req, cache)
        return req

    def _hit(self, tokens):
        return len(
            self.cache.match_prefix(MatchPrefixParams(key=_key(tokens))).device_indices
        )

    def _node(self, tokens):
        m = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        self.assertEqual(len(m.device_indices), len(tokens))
        return m.last_device_node

    def _fake_backup(self, node):
        node.component_data[FULL].host_value = torch.arange(
            len(node.key), dtype=torch.int64
        )

    # -- tests ---------------------------------------------------------------

    def test_final_turn_frees_private_chain_and_keeps_shared_prefix(self):
        P = list(range(1, 9))
        a1, a2 = list(range(101, 105)), list(range(105, 109))
        b1 = list(range(201, 205))
        self._finish(P + a1)  # trajectory A, turn 1
        self._finish(P + b1)  # trajectory B, turn 1 -> splits at P
        self.assertEqual(self._hit(P + a1), 12)
        cached_before = KV_SIZE - self.allocator.available_size()
        self.assertEqual(cached_before, 16)

        self._finish(P + a1 + a2, evict_on_finish=True)  # A, final turn

        # A's private node is gone, its final turn was never inserted.
        self.assertEqual(self._hit(P + a1 + a2), 8)
        # Shared prefix and the sibling trajectory are intact and unmarked.
        self.assertEqual(self._hit(P + b1), 12)
        self.assertFalse(self._node(P).evict_first)
        self.assertFalse(self._node(P + b1).evict_first)
        self.assertEqual(KV_SIZE - self.allocator.available_size(), 12)
        self.cache.sanity_check()

    def test_without_signal_final_turn_is_cached(self):
        P = list(range(1, 9))
        a1, a2 = list(range(101, 105)), list(range(105, 109))
        self._finish(P + a1)
        self._finish(P + a1 + a2)
        self.assertEqual(self._hit(P + a1 + a2), 16)
        self.assertEqual(KV_SIZE - self.allocator.available_size(), 16)
        self.cache.sanity_check()

    def test_backed_up_chain_is_demoted_marked_and_host_evicted_first(self):
        P = list(range(1, 9))
        a1, a2, a3 = (list(range(s, s + 4)) for s in (101, 105, 109))
        c1 = list(range(301, 305))
        self._finish(P + a1)
        self._finish(P + a1 + a2)
        n1, n2 = self._node(P + a1), self._node(P + a1 + a2)
        self.assertIs(n2.parent, n1)
        # Pretend HiCache wrote both to host (parent before child).
        self._fake_backup(n1)
        self._fake_backup(n2)
        # An older, unrelated host-only leaf that plain LRU would evict first.
        self._finish(c1)
        nc = self._node(c1)
        self._fake_backup(nc)
        self.cache._evict_to_host(nc, {FULL: 0})
        self.assertTrue(nc.evicted)
        # Age the unrelated leaf below A's nodes.
        nc.last_access_time = min(n1.last_access_time, n2.last_access_time) - 1

        self._finish(P + a1 + a2 + a3, evict_on_finish=True)

        # Backed-up nodes stay in the tree as marked host leaves, device freed.
        for n in (n1, n2):
            self.assertTrue(n.evicted)
            self.assertTrue(n.backuped)
            self.assertTrue(n.evict_first)
        self.assertFalse(nc.evict_first)
        self.assertEqual(self._hit(P + a1 + a2), 0)
        self.assertEqual(KV_SIZE - self.allocator.available_size(), 0)

        # Host eviction takes the marked chain (child, then parent) before the
        # older unmarked leaf.
        freed = self.cache.evict_host(len(P + a1 + a2))
        self.assertEqual(freed, len(P + a1 + a2))
        self.assertNotIn(n2.key.child_key(1), n1.children)
        self.assertNotIn(n1.key.child_key(1), self.cache.root_node.children)
        self.assertTrue(nc.backuped)
        self.assertIn(nc.key.child_key(1), self.cache.root_node.children)
        self.cache.sanity_check()

    def test_pinned_node_is_marked_not_freed_and_match_clears_mark(self):
        P = list(range(1, 9))
        a1, a2 = list(range(101, 105)), list(range(105, 109))
        self._finish(P + a1)
        node = self._node(P + a1)

        # Another running request still holds the node: the walk must stop.
        self._finish(P + a1 + a2, evict_on_finish=True, extra_lock=True)
        self.assertTrue(node.evict_first)
        self.assertEqual(self._hit(P + a1), 12)
        self.assertEqual(KV_SIZE - self.allocator.available_size(), 12)
        # The match above is a reuse: it clears the mark.
        self.assertFalse(node.evict_first)
        self.cache.dec_lock_ref(node)
        self.cache.sanity_check()

    def test_marked_node_is_first_in_device_eviction_heap(self):
        P = list(range(1, 9))
        b1 = list(range(201, 205))
        self._finish(P)
        self._finish(b1)
        np_, nb = self._node(P), self._node(b1)
        # Age the unrelated leaf below P; LRU alone would evict b1 first.
        nb.last_access_time = np_.last_access_time - 1
        self.assertLess(self.cache.eviction_key(nb), self.cache.eviction_key(np_))
        np_.evict_first = True
        self.assertLess(self.cache.eviction_key(np_), self.cache.eviction_key(nb))


class EvictOnFinishPlumbingTest(unittest.TestCase):
    def test_generate_req_input_carries_flag_per_item(self):
        obj = GenerateReqInput(
            text=["a", "b"], sampling_params=[{}, {}], evict_on_finish=True
        )
        obj.normalize_batch_and_arguments()
        self.assertTrue(obj[0].evict_on_finish)
        self.assertTrue(obj[1].evict_on_finish)
        self.assertFalse(GenerateReqInput(text="a").evict_on_finish)

    def test_release_kv_cache_never_inserts_flagged_request(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )
        seen = []
        tree_cache = SimpleNamespace(
            cache_finished_req=lambda req, is_insert, kv_len_to_handle: seen.append(
                is_insert
            ),
            supports_mamba=lambda: False,
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(free=lambda idx: None),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((1, 8), dtype=torch.int64),
                free=lambda req: None,
            ),
        )
        for flag in (False, True):
            req = Req(
                rid="r",
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(max_new_tokens=1),
                evict_on_finish=flag,
            )
            req.req_pool_idx = 0
            req.kv = ReqKvInfo(kv_allocated_len=3, swa_evicted_seqlen=0)
            req.kv_committed_len = 3
            release_kv_cache(req, tree_cache, is_insert=True)
        self.assertEqual(seen, [True, False])


class EvictOnFinishSessionTest(unittest.TestCase):
    """``Session.create_req`` must carry ``evict_on_finish`` onto the ``Req``
    it builds — session turns otherwise silently lose the signal and stay
    cached like any live prefix."""

    def _create_req(self, evict_on_finish):
        session = Session(capacity_of_str_len=8, streaming=False)
        req = TokenizedGenerateReqInput(
            rid="turn-0",
            input_text=None,
            input_ids=array("q", [1, 2, 3]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_params=SessionParams(id=session.session_id),
            evict_on_finish=evict_on_finish,
        )
        tokenizer = SimpleNamespace(bos_token_id=-1)
        return session.create_req(req, tokenizer, vocab_size=32000)

    def test_session_create_req_carries_flag(self):
        self.assertTrue(self._create_req(True).evict_on_finish)

    def test_session_create_req_flag_absent_defaults_false(self):
        self.assertFalse(self._create_req(False).evict_on_finish)


if __name__ == "__main__":
    unittest.main()
