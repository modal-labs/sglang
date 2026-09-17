"""Explicit eviction signal (``evict_on_finish``) across cache backends.

A client that knows a request is the last turn of its trajectory sets
``evict_on_finish``. The request's KV is never inserted, and the tree frees
the request's private prefix chain on finish instead of leaving it as the
most recently used entry. A request owns a node only if it extended strictly
past it (``kv_len > matched_len``); a request whose tokens end exactly at an
existing node has no claim on it, so a shared exact-hit leaf survives a
flagged finish.

Covered here: ``UnifiedRadixCache`` (marks ``evict_first`` and frees device
data inline), ``RadixCache`` and ``MambaRadixCache`` (walk-and-free at
finish), and ``StreamingSession`` (releases the session slot, then evicts
the first turn's tree node).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.hi_mamba_radix_cache import (
    HiMambaRadixCache,
    HostLRUList,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.mamba_radix_cache import (
    LRUList,
    MambaRadixCache,
    TreeNode,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase

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
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if new_len:
            new_indices = allocator.alloc(new_len)
            self.assertIsNotNone(new_indices)
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

    def test_exact_hit_leaf_survives_final_turn(self):
        """A flagged request whose tokens end exactly at an existing node has
        no claim on it: the strict-extension rule (``kv_len > matched_len``)
        must leave a shared exact-hit leaf untouched."""
        P = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._finish(P + a1)
        self._finish(P + b1)
        self._finish(P + a1, evict_on_finish=True)  # exact hit on P+a1

        self.assertEqual(self._hit(P + a1), 12)
        self.assertEqual(self._hit(P + b1), 12)
        self.assertFalse(self._node(P + a1).evict_first)
        self.assertEqual(KV_SIZE - self.allocator.available_size(), 16)
        self.cache.sanity_check()


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


def _build_radix_cache():
    """Plain ``RadixCache`` (no tree components) over small CPU pools."""
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
        eviction_policy="lru",
    )
    return RadixCache(params=params), allocator, req_to_token_pool


class EvictOnFinishRadixCacheTest(CustomTestCase):
    """``RadixCache.evict_finished_req_prefix``: the finishing request frees the
    leaf chain it privately extended, stopping at shared / pinned / host-backed
    nodes. Mid-chunking inserts stay visible so concurrent requests can still
    share them; only finish removes the private chain."""

    def setUp(self):
        self.cache, self.allocator, self.req_pool = _build_radix_cache()
        self._rid = 0

    def _finish(self, tokens, *, evict_on_finish=False):
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
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
        req.extra_key = None
        new_len = len(tokens) - prefix_len
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if new_len:
            pool.write(
                (req.req_pool_idx, slice(prefix_len, len(tokens))),
                allocator.alloc(new_len),
            )
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

    def _chunked_finish(self, tokens, chunk_end, *, evict_on_finish):
        """Drive a two-chunk prefill of ``tokens`` (first chunk covers
        ``[:chunk_end]``) and finish it."""
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
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
        req.extra_key = None
        # chunk 1: fill up to chunk_end
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if chunk_end > prefix_len:
            pool.write(
                (req.req_pool_idx, slice(prefix_len, chunk_end)),
                allocator.alloc(chunk_end - prefix_len),
            )
        req.kv = ReqKvInfo(kv_allocated_len=chunk_end, swa_evicted_seqlen=0)
        req.kv_committed_len = chunk_end
        req.full_untruncated_fill_ids = array("q", tokens[:chunk_end])
        req.set_extend_range(prefix_len, chunk_end)
        cache.cache_unfinished_req(req, chunked=True)
        # Mid-chunking sharing: the partial prefix is already in the tree.
        mid = cache.match_prefix(MatchPrefixParams(key=_key(tokens[:chunk_end])))
        assert len(mid.device_indices) == chunk_end
        # chunk 2: the rest, then finish
        if len(tokens) > chunk_end:
            pool.write(
                (req.req_pool_idx, slice(chunk_end, len(tokens))),
                allocator.alloc(len(tokens) - chunk_end),
            )
        req.kv.kv_allocated_len = len(tokens)
        req.kv_committed_len = len(tokens)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(chunk_end, len(tokens))
        release_kv_cache(req, cache)
        return req

    def test_chunked_prefill_flagged_finish_frees_private_chain(self):
        S = list(range(1, 9))
        b1 = list(range(201, 205))
        c1 = list(range(101, 105))
        c2 = list(range(105, 109))
        self._finish(S + b1)  # trajectory B turn 1

        req = self._chunked_finish(S + c1 + c2, len(S + c1), evict_on_finish=True)
        del req

        # The mid-chunking S+c1 insert was shareable while chunking; at finish
        # the walk removed it. S stays because b1 still shares it.
        self.assertEqual(self._hit(S + c1), len(S))
        self.assertEqual(self._hit(S + b1), len(S + b1))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), len(S + b1))
        self.assertEqual(self.cache.total_size(), len(S + b1))

    def test_chunked_prefill_unflagged_finish_keeps_chain(self):
        S = list(range(1, 9))
        b1 = list(range(201, 205))
        c1 = list(range(101, 105))
        c2 = list(range(105, 109))
        self._finish(S + b1)
        self._chunked_finish(S + c1 + c2, len(S + c1), evict_on_finish=False)
        self.assertEqual(self._hit(S + c1 + c2), len(S + c1 + c2))
        self.assertEqual(self._hit(S + b1), len(S + b1))

    def test_exact_hit_leaf_survives_flagged_finish(self):
        S = list(range(1, 9))
        self._finish(S)
        self._finish(S, evict_on_finish=True)
        self.assertEqual(self._hit(S), len(S))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), len(S))


class EvictOnFinishStreamingSessionTest(CustomTestCase):
    """Embedded streaming session on ``UnifiedRadixCache``: a flagged final
    turn releases the session slot (KV + req pool slot + first turn's tree
    lock) and then evicts the first turn's private tree chain."""

    def setUp(self):
        self.cache, self.allocator, self.req_pool = _build_cache()
        self._full_free_slots = len(self.req_pool.free_slots)
        self._rid = 0

    def _create_req(self, session, input_ids, evict_on_finish=False):
        obj = TokenizedGenerateReqInput(
            rid=f"turn-{self._rid}",
            input_text=None,
            input_ids=array("q", input_ids),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_params=SessionParams(id=session.session_id),
            evict_on_finish=evict_on_finish,
        )
        self._rid += 1
        return session.create_req(
            obj, SimpleNamespace(bos_token_id=-1), vocab_size=32000
        )

    def _turn(self, req):
        """Scheduler path for one streaming turn: match -> lock -> alloc ->
        write -> (first turn) insert+lock -> release."""
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(
            MatchPrefixParams(key=_key(req.origin_input_ids), req=req)
        )
        prefix_len = len(match.device_indices)
        inc = cache.inc_lock_ref(match.last_device_node)
        if req.req_pool_idx is None:
            pool.alloc([req])
        total = len(req.origin_input_ids)
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if total > prefix_len:
            pool.write(
                (req.req_pool_idx, slice(prefix_len, total)),
                allocator.alloc(total - prefix_len),
            )
        req.output_ids = array("q")
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.cache_protected_len = (
            match.cache_protected_len
            if match.cache_protected_len is not None
            else prefix_len
        )
        req.swa_uuid_for_lock = inc.swa_uuid_for_lock
        req.skip_lock_node_ids = inc.skip_lock_node_ids
        req.kv_committed_len = total
        if req.kv is None:
            req.kv = ReqKvInfo(kv_allocated_len=total, swa_evicted_seqlen=0)
        else:
            req.kv.kv_allocated_len = total
        req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        req.set_extend_range(prefix_len, total)
        cache.cache_unfinished_req(req)
        release_kv_cache(req, cache)

    def _drive_session(self, t1, t2, flag):
        session = Session(capacity_of_str_len=8, streaming=True)
        req1 = self._create_req(session, t1)
        self._turn(req1)
        self.assertTrue(self.cache.session.has_slot(session.session_id))
        req2 = self._create_req(session, t2, evict_on_finish=flag)
        self._turn(req2)
        return session, req2

    def test_flagged_final_turn_releases_slot_and_tree(self):
        t1 = list(range(1, 9))
        t2 = list(range(51, 55))
        session, _ = self._drive_session(t1, t2, flag=True)

        self.assertFalse(self.cache.session.has_slot(session.session_id))
        # Slot KV freed and the turn-1 tree chain evicted: nothing in use.
        self.assertEqual(KV_SIZE - self.allocator.available_size(), 0)
        self.assertEqual(len(self.req_pool.free_slots), self._full_free_slots)
        self.assertEqual(
            len(
                self.cache.match_prefix(MatchPrefixParams(key=_key(t1))).device_indices
            ),
            0,
        )
        self.cache.sanity_check()

    def test_unflagged_turn_keeps_slot_and_kv(self):
        t1 = list(range(1, 9))
        t2 = list(range(51, 55))
        session, _ = self._drive_session(t1, t2, flag=False)

        self.assertTrue(self.cache.session.has_slot(session.session_id))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), len(t1) + len(t2))


class EvictOnFinishMambaTest(CustomTestCase):
    """``MambaRadixCache.evict_finished_req_prefix``: frees the private leaf
    chain (full KV + mamba state) up to the first shared/pinned/host-backed
    node."""

    def setUp(self):
        server_args = ServerArgs(model_path="dummy", page_size=1)
        # MambaRadixCache reads mamba_cache_chunk_size, whose property would
        # load the HF config for the dummy model path; mirror the default.
        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)
        size = 64
        num_layers = 8
        global_interval = 4
        max_num_reqs = 8
        mamba_cache_size = 8
        max_context_len = 64
        device = "cpu"
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=64,
                n_groups=2,
                num_heads=4,
                head_dim=16,
                state_size=16,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
        self.req_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=None,
        )
        kv_pool = HybridLinearKVPool(
            size=size,
            dtype=torch.bfloat16,
            page_size=1,
            head_num=2,
            head_dim=16,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            enable_memory_saver=False,
            mamba_pool=self.req_pool.mamba_pool,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=torch.bfloat16,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=1,
            disable=False,
        )
        self.tree = MambaRadixCache(params=params)
        self.mamba_cache_size = mamba_cache_size
        self._rid = 0

    def _finish(self, tokens, *, evict_on_finish=False):
        tree, allocator, pool = self.tree, self.allocator, self.req_pool
        req = Req(
            rid=str(self._rid),
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            evict_on_finish=evict_on_finish,
        )
        self._rid += 1
        pool.alloc([req])
        match = tree.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
        tree.inc_lock_ref(match.last_device_node)
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if len(tokens) > prefix_len:
            pool.write(
                (req.req_pool_idx, slice(prefix_len, len(tokens))),
                allocator.alloc(len(tokens) - prefix_len),
            )
        req.output_ids = array("q")
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.cache_protected_len = prefix_len
        req.kv_committed_len = len(tokens)
        req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(prefix_len, len(tokens))
        release_kv_cache(req, tree)
        return req

    def _hit(self, tokens):
        return len(
            self.tree.match_prefix(MatchPrefixParams(key=_key(tokens))).device_indices
        )

    def test_flagged_final_turn_frees_private_leaf(self):
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        a2 = list(range(105, 109))
        b1 = list(range(201, 205))
        self._finish(S + b1)  # trajectory B turn 1
        self._finish(S + a1)  # trajectory A turn 1 (private leaf under S)
        mamba_used_before = self.mamba_cache_size - (
            self.req_pool.mamba_allocator.available_size()
        )

        self._finish(S + a1 + a2, evict_on_finish=True)

        # A's private leaf (S+a1) is gone with its mamba state. The shared S
        # node survives as a mamba tombstone (split nodes carry no mamba
        # state), so a match can no longer stop at S — the surviving chain
        # is observed through S+b1.
        self.assertEqual(self._hit(S + a1), 0)
        self.assertEqual(self._hit(S + b1), len(S + b1))
        self.assertEqual(64 - self.allocator.available_size(), len(S + b1))
        mamba_used_after = self.mamba_cache_size - (
            self.req_pool.mamba_allocator.available_size()
        )
        self.assertEqual(mamba_used_before, 2)  # S+b1 and S+a1 nodes
        self.assertEqual(mamba_used_after, 1)  # only S+b1's donated state
        self.tree.sanity_check()

    def test_exact_hit_leaf_survives_flagged_finish(self):
        S = list(range(1, 9))
        self._finish(S)
        self._finish(S, evict_on_finish=True)
        self.assertEqual(self._hit(S), len(S))
        self.assertEqual(64 - self.allocator.available_size(), len(S))
        self.tree.sanity_check()


class _RecordingAllocator:
    """Fake allocator/pool that records freed indices."""

    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.extend(torch.as_tensor(value).tolist())


class _FakeHiMambaController:
    """Minimal HiCache controller surface used by the hi-mamba free path."""

    write_policy = "write_back"
    ack_load_queue = []

    def __init__(self):
        self.device_freed = 0
        self.host_freed = 0

    def evict_device(self, value):
        self.device_freed += len(value)
        return len(value)

    def evict_host(self, value):
        self.host_freed += len(value)
        return len(value)


class _FakeHiRadixController:
    """Minimal HiCache controller surface used by the hiradix free paths.

    ``mem_pool_device_allocator`` is the same object the real
    ``HiCacheController`` stores (cache_controller.py), so device frees go
    through the real allocator while host/demote frees are counted here."""

    write_policy = "write_back"

    def __init__(self, allocator):
        self.mem_pool_device_allocator = allocator
        self.device_freed = 0
        self.host_freed = 0

    def evict_device(self, value):
        self.device_freed += len(value)
        return len(value)

    def evict_host(self, value):
        self.host_freed += len(value)
        return len(value)


class EvictOnFinishHiMambaTest(CustomTestCase):
    """``HiMambaRadixCache.evict_finished_req_prefix`` must drive the
    hierarchical free path: the inherited ``MambaRadixCache`` walk unpacks the
    3-tuple tombstone cascade of the hi-mamba override and dies mid-walk."""

    def _build(self):
        server_args = ServerArgs(model_path="dummy", page_size=1)
        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)
        cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
        cache.disable = False
        cache.page_size = 1
        cache.device = torch.device("cpu")
        cache.enable_storage = False
        cache.enable_kv_cache_events = False
        cache.mamba_max_states_per_path = -1
        cache.mamba_cache_chunk_size = 64
        cache.token_to_kv_pool_allocator = _RecordingAllocator()
        cache.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingAllocator())
        cache.cache_controller = _FakeHiMambaController()
        cache.tp_world_size = 1
        cache.metrics_collector = None
        cache.ongoing_write_through = {}
        cache.ongoing_load_back = {}
        cache.evictable_full_device_leaves = set()
        cache.evictable_full_host_leaves = set()
        cache.full_lru_list = LRUList(mamba=False)
        cache.mamba_lru_list = LRUList(mamba=True)
        cache.mamba_host_lru_list = HostLRUList()
        cache.mamba_pool_host = _RecordingAllocator()
        cache.full_evictable_size_ = 0
        cache.mamba_evictable_size_ = 0
        cache.full_protected_size_ = 0
        cache.mamba_protected_size_ = 0

        root = TreeNode()
        root.key = RadixKey(array("q"), None)
        root.value = []
        root.hash_value = []
        root.full_lock_ref = 1
        root.mamba_lock_ref = 1
        cache.root_node = root
        return cache

    def _insert(self, cache, tokens, mamba_slot, kv_start):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens), None),
                value=torch.arange(kv_start, kv_start + len(tokens), dtype=torch.int64),
                mamba_value=torch.tensor([mamba_slot], dtype=torch.int64),
            )
        )

    def _hit(self, cache, tokens):
        return len(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens), None))
            ).device_indices
        )

    def test_flagged_finish_frees_private_leaf_via_hierarchical_path(self):
        cache = self._build()
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        a2 = list(range(105, 109))
        b1 = list(range(201, 205))
        self._insert(cache, S + b1, 10, 1000)  # trajectory B turn 1
        # Splitting at S makes S a mamba tombstone (no state), so a match
        # cannot anchor there; it still holds the shared full KV.
        self.assertEqual(self._hit(cache, S + a1), 0)
        self._insert(cache, S + a1, 11, 2000)  # trajectory A turn 1
        a1_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S + a1), None))
        ).last_device_node

        # Flagged finish of S+a1+a2: the a1 leaf (device KV + mamba) is freed
        # via _evict_regular; the shared S tombstone and the S+b1 sibling
        # survive. A mamba host copy would make the leaf host-backed — that
        # survival case is covered by test_breaks_on_host_backed_*.
        cache.evict_finished_req_prefix(
            a1_node,
            matched_len=len(S + a1),
            kv_len=len(S + a1 + a2),
            rid="r-a2",
        )

        self.assertEqual(self._hit(cache, S + a1), 0)
        self.assertEqual(self._hit(cache, S + b1), len(S + b1))
        self.assertEqual(cache.cache_controller.device_freed, len(a1))
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [11])
        self.assertEqual(cache.mamba_pool_host.freed, [])
        cache.sanity_check()

    def test_host_backed_private_leaf_freed_on_both_tiers(self):
        """A private host-backed leaf is dropped on BOTH tiers: demoted off
        device via _evict_to_host, then host copy freed via _evict_host_leaf
        (host KV + mamba host copy), and detached from the tree."""
        cache = self._build()
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._insert(cache, S + b1, 10, 1000)
        self._insert(cache, S + a1, 11, 2000)
        a1_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S + a1), None))
        ).last_device_node
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)
        a1_node.mamba_host_value = torch.tensor([11], dtype=torch.int64)

        cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r"
        )
        self.assertEqual(cache.cache_controller.device_freed, len(a1))
        self.assertEqual(cache.cache_controller.host_freed, len(a1))
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [11])
        self.assertEqual(cache.mamba_pool_host.freed, [11])
        # a1 detached; the shared S tombstone and S+b1 survive.
        self.assertEqual(self._hit(cache, S + b1), len(S + b1))
        cache.sanity_check()

    def test_evicted_host_only_leaf_freed_and_detached(self):
        """An already-evicted (host-only) private leaf: host KV + mamba host
        copy freed and node detached."""
        cache = self._build()
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._insert(cache, S + b1, 10, 1000)
        self._insert(cache, S + a1, 11, 2000)
        a1_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S + a1), None))
        ).last_device_node
        # Demote through the real path: device freed, host copies remain.
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)
        a1_node.mamba_host_value = torch.tensor([11], dtype=torch.int64)
        cache._evict_to_host(a1_node)
        device_freed_after_demote = cache.cache_controller.device_freed

        cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r"
        )
        self.assertEqual(cache.cache_controller.device_freed, device_freed_after_demote)
        self.assertEqual(cache.cache_controller.host_freed, len(a1))
        self.assertEqual(cache.mamba_pool_host.freed, [11])
        self.assertIsNone(a1_node.host_value)
        self.assertIsNone(a1_node.mamba_host_value)
        self.assertEqual(self._hit(cache, S + b1), len(S + b1))
        cache.sanity_check()

    def test_breaks_on_locked_in_flight_and_host_in_use_nodes(self):
        """The walk must stop at locked / write-in-flight / host-in-use nodes."""
        S = list(range(1, 9))
        a1 = list(range(101, 105))

        # Locked leaf (another request shares it).
        cache = self._build()
        self._insert(cache, S + a1, 11, 2000)
        a1_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S + a1), None))
        ).last_device_node
        a1_node.full_lock_ref += 1
        cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r"
        )
        self.assertEqual(cache.cache_controller.device_freed, 0)
        self.assertIn(a1_node.key.child_key(1), a1_node.parent.children)

        # In-flight write-through: node marked in ongoing_write_through even
        # before host_value is published.
        cache = self._build()
        self._insert(cache, S + a1, 11, 2000)
        a1_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S + a1), None))
        ).last_device_node
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)
        cache.ongoing_write_through[a1_node.id] = a1_node
        cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r"
        )
        self.assertEqual(cache.cache_controller.device_freed, 0)
        self.assertEqual(cache.cache_controller.host_freed, 0)
        self.assertIsNotNone(a1_node.host_value)
        self.assertIn(a1_node.key.child_key(1), a1_node.parent.children)

        # Host KV in use (host_ref_counter > 0).
        cache = self._build()
        self._insert(cache, S + a1, 11, 2000)
        a1_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S + a1), None))
        ).last_device_node
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)
        a1_node.host_ref_counter = 1
        cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r"
        )
        self.assertEqual(cache.cache_controller.device_freed, 0)
        self.assertEqual(cache.cache_controller.host_freed, 0)
        self.assertIsNotNone(a1_node.host_value)
        self.assertIn(a1_node.key.child_key(1), a1_node.parent.children)

    def test_exact_hit_frees_nothing(self):
        """kv_len == matched_len: the request never extended past the node."""
        cache = self._build()
        S = list(range(1, 9))
        self._insert(cache, S, 10, 1000)
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", S), None))
        ).last_device_node
        cache.evict_finished_req_prefix(
            node, matched_len=len(S), kv_len=len(S), rid="r"
        )
        self.assertEqual(cache.cache_controller.device_freed, 0)
        self.assertEqual(self._hit(cache, S), len(S))
        cache.sanity_check()

    def test_exact_hit_on_loaded_back_prefix_survives(self):
        """Production load-back shape: the leaf is backuped (host copy) AND
        resident (device value present). A flagged exact hit has
        matched_len == finished_key_len(kv_len) and must leave it alone."""
        cache = self._build()
        P = list(range(1, 9))
        self._insert(cache, P, 10, 1000)
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", P), None))
        ).last_device_node
        node.host_value = torch.arange(len(node.key), dtype=torch.int64)
        node.mamba_host_value = torch.tensor([10], dtype=torch.int64)

        cache.evict_finished_req_prefix(
            node,
            matched_len=len(P),
            kv_len=cache.finished_key_len(len(P)),
            rid="r-lb",
        )
        self.assertEqual(cache.cache_controller.device_freed, 0)
        self.assertIsNotNone(node.host_value)
        self.assertIsNotNone(node.mamba_host_value)
        self.assertIsNotNone(node.value)
        self.assertIsNotNone(node.mamba_value)
        cache.sanity_check()


def _build_hiradix_cache():
    """``HiRadixCache`` over small CPU pools.

    ``__init__`` needs a live ``torch.distributed`` group and host pools, so
    the fixture sets only the attributes the inherited ``RadixCache`` walk
    touches; ``cache_controller.mem_pool_device_allocator`` is the same
    ``params.token_to_kv_pool_allocator`` object in the real
    ``HiCacheController`` (cache_controller.py), so freeing through the
    allocator here mirrors production.
    """
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
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.disable = False
    cache.disable_finished_insert = False
    cache.is_eagle = False
    cache.page_size = 1
    cache.device = torch.device("cpu")
    cache.enable_storage = False
    cache.enable_kv_cache_events = False
    cache.kv_event_queue = []
    cache.metrics_collector = None
    cache.req_to_token_pool = req_to_token_pool
    cache.token_to_kv_pool_allocator = allocator
    cache.cache_controller = _FakeHiRadixController(allocator)
    cache.write_through_threshold = 1 << 30
    cache.enable_session_radix_cache = False
    cache.ongoing_write_through = {}
    cache.ongoing_load_back = {}
    cache.ongoing_prefetch = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.evictable_leaves = set()
    cache.evictable_host_leaves = set()
    root = TreeNode()
    root.key = RadixKey(array("q"), None)
    root.value = []
    root.host_value = []  # backuped, so match_prefix's host walk stops at root
    root.lock_ref = 1
    root.hash_value = []
    cache.root_node = root
    cache._empty_match_result = MatchResult(
        device_indices=torch.empty((0,), dtype=torch.int64),
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
    )
    return cache, allocator, req_to_token_pool


class EvictOnFinishHiRadixCacheTest(CustomTestCase):
    """``HiRadixCache.evict_finished_req_prefix``: a private chain is dropped
    on both tiers — device via ``_evict_backuped``/``_evict_regular``, host
    via ``_evict_host_node`` — stopping only at shared, pinned, in-use, or
    in-flight nodes."""

    def setUp(self):
        self.cache, self.allocator, self.req_pool = _build_hiradix_cache()
        self._rid = 0

    def _finish(self, tokens, *, evict_on_finish=False):
        cache, allocator, pool = self.cache, self.allocator, self.req_pool
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
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
        req.extra_key = None
        new_len = len(tokens) - prefix_len
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if new_len:
            pool.write(
                (req.req_pool_idx, slice(prefix_len, len(tokens))),
                allocator.alloc(new_len),
            )
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

    def _leaf(self, tokens):
        return self.cache.match_prefix(
            MatchPrefixParams(key=_key(tokens))
        ).last_device_node

    def test_flagged_finish_frees_private_leaf(self):
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._finish(S + a1)
        self._finish(S + b1)  # splits at S; S+b1 is the shared sibling
        cached_before = KV_SIZE - self.allocator.available_size()
        self.assertEqual(cached_before, len(S) + len(a1) + len(b1))

        self._finish(S + a1, evict_on_finish=True)  # exact hit -> no claim
        self.assertEqual(self._hit(S + a1), len(S + a1))
        self.assertEqual(self._hit(S + b1), len(S + b1))

        # A flagged finish that extended past S+a1 frees the private a1 leaf.
        a1_node = self._leaf(S + a1)
        ext = list(range(301, 305))
        self.cache.evict_finished_req_prefix(
            a1_node,
            matched_len=len(S + a1),
            kv_len=len(S + a1) + len(ext),
            rid="r-ext",
        )
        self.assertEqual(self._hit(S + a1), len(S))
        self.assertEqual(self._hit(S + b1), len(S + b1))

    def test_flagged_finish_frees_private_extension_through_release(self):
        """(a) through release_kv_cache: private leaf freed, shared survives."""
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        a2 = list(range(105, 109))
        b1 = list(range(201, 205))
        self._finish(S + b1)
        self._finish(S + a1)
        self.assertEqual(self._hit(S + a1), len(S + a1))

        self._finish(S + a1 + a2, evict_on_finish=True)
        # S+a1's leaf was private to A's final turn -> freed; the S internal
        # node and S+b1 survive, so a match now stops at S.
        self.assertEqual(self._hit(S + a1 + a2), len(S))
        self.assertEqual(self._hit(S + b1), len(S + b1))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), len(S) + len(b1))

    def test_host_backed_private_chain_freed_on_both_tiers(self):
        """A private host-backed leaf is dropped on BOTH tiers: device via
        ``_evict_backuped``, host via ``_evict_host_node``; the node leaves
        the tree entirely."""
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._finish(S + b1)
        self._finish(S + a1)
        a1_node = self._leaf(S + a1)
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)

        self.cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r-host"
        )
        # a1 freed on both tiers and detached; the shared S internal node
        # survives via the S+b1 sibling.
        self.assertEqual(self._hit(S + a1), len(S))
        self.assertEqual(self._hit(S + b1), len(S + b1))
        self.assertEqual(self.cache.cache_controller.device_freed, len(a1))
        self.assertEqual(self.cache.cache_controller.host_freed, len(a1))
        self.assertIsNone(a1_node.host_value)

    def test_evicted_host_only_leaf_freed_and_detached(self):
        """An already-evicted (host-only) private leaf gets its host copy
        freed via ``_evict_host_node`` and is detached."""
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._finish(S + b1)
        self._finish(S + a1)
        a1_node = self._leaf(S + a1)
        used_before = KV_SIZE - self.allocator.available_size()
        # Simulate the demoted state: device gone, host copy remains.
        a1_node.value = None
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)

        self.cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r-host"
        )
        self.assertEqual(self.cache.cache_controller.device_freed, 0)
        self.assertEqual(self.cache.cache_controller.host_freed, len(a1))
        self.assertIsNone(a1_node.host_value)
        self.assertEqual(self._hit(S + a1), len(S))
        self.assertEqual(self._hit(S + b1), len(S + b1))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), used_before)

    def test_locked_in_flight_and_host_in_use_nodes_survive(self):
        S = list(range(1, 9))
        a1 = list(range(101, 105))

        # Locked leaf.
        self._finish(S + a1)
        a1_node = self._leaf(S + a1)
        a1_node.lock_ref += 1
        self.cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r-lock"
        )
        self.assertEqual(self._hit(S + a1), len(S + a1))

        # In-flight write-through: untouched on both tiers.
        cache2, allocator2, _ = _build_hiradix_cache()
        self.cache, self.allocator = cache2, allocator2
        self._finish(S + a1)
        a1_node = self._leaf(S + a1)
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)
        self.cache.ongoing_write_through[a1_node.id] = a1_node
        self.cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r-wt"
        )
        self.assertEqual(self._hit(S + a1), len(S + a1))
        self.assertEqual(self.cache.cache_controller.device_freed, 0)
        self.assertEqual(self.cache.cache_controller.host_freed, 0)
        self.assertIsNotNone(a1_node.host_value)

        # Host KV in use (host_ref_counter > 0).
        cache3, allocator3, _ = _build_hiradix_cache()
        self.cache, self.allocator = cache3, allocator3
        self._finish(S + a1)
        a1_node = self._leaf(S + a1)
        a1_node.host_value = torch.arange(len(a1), dtype=torch.int64)
        a1_node.host_ref_counter = 1
        self.cache.evict_finished_req_prefix(
            a1_node, matched_len=len(S), kv_len=len(S + a1), rid="r-href"
        )
        self.assertEqual(self._hit(S + a1), len(S + a1))
        self.assertEqual(self.cache.cache_controller.device_freed, 0)
        self.assertEqual(self.cache.cache_controller.host_freed, 0)
        self.assertIsNotNone(a1_node.host_value)

    def test_exact_hit_frees_nothing(self):
        S = list(range(1, 9))
        self._finish(S)
        node = self._leaf(S)
        self.cache.evict_finished_req_prefix(
            node, matched_len=len(S), kv_len=len(S), rid="r-exact"
        )
        self.assertEqual(self._hit(S), len(S))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), len(S))

    def test_exact_hit_on_loaded_back_prefix_survives(self):
        """Production load-back shape: the leaf is backuped (host copy) AND
        resident (device value present, evicted False). A flagged request that
        loaded P back and ended exactly on it releases through
        ``release_kv_cache`` and must leave the node, its host copy, and its
        device value intact."""
        P = list(range(1, 9))
        self._finish(P)
        leaf = self._leaf(P)
        leaf.host_value = torch.arange(len(P), dtype=torch.int64)
        used_before = KV_SIZE - self.allocator.available_size()

        self._finish(P, evict_on_finish=True)

        self.assertIs(self._leaf(P), leaf)
        self.assertIsNotNone(leaf.host_value)
        self.assertIsNotNone(leaf.value)
        self.assertEqual(self._hit(P), len(P))
        self.assertEqual(KV_SIZE - self.allocator.available_size(), used_before)

    def test_load_back_failure_state_drops_to_root(self):
        """schedule_policy failure branch sets ``req.last_node = root`` when
        host load-back fails. Verify the post-branch state is safe: locking
        and finishing at root are no-ops and never free a pre-existing node,
        whereas the pre-branch state (last_node=D, cache_protected_len=0)
        would have freed D."""
        P = list(range(1, 9))
        self._finish(P)
        d_node = self._leaf(P)
        used_before = KV_SIZE - self.allocator.available_size()

        # Locking root is a no-op for every tree (loops stop at root).
        self.cache.inc_lock_ref(self.cache.root_node)
        self.cache.dec_lock_ref(self.cache.root_node)
        self.assertEqual(self.cache.root_node.lock_ref, 1)

        # Post-branch finish state: last_node=root, matched_len=0.
        self.cache.evict_finished_req_prefix(
            self.cache.root_node,
            matched_len=0,
            kv_len=self.cache.finished_key_len(len(P)),
            rid="r-failed-lb",
        )
        self.assertIs(self._leaf(P), d_node)
        self.assertIsNotNone(d_node.value)
        self.assertEqual(KV_SIZE - self.allocator.available_size(), used_before)

        # Counterfactual: the pre-branch state would have freed D (matched 0,
        # childless, unlocked) — this is the bug the fix removes.
        self.cache.evict_finished_req_prefix(
            d_node,
            matched_len=0,
            kv_len=self.cache.finished_key_len(len(P)),
            rid="r-bug",
        )
        self.assertEqual(self._hit(P), 0)


class EvictOnFinishLogicalKeyLenTest(CustomTestCase):
    """``finished_key_len`` normalizes ``kv_len`` into logical radix-key units
    (bigram positions for EAGLE, page-aligned) — the unit ``matched_len`` lives
    in. Without it an EAGLE or page-unaligned exact hit looks like a strict
    extension and the shared leaf is evicted."""

    def test_finished_key_len_matches_radix_key(self):
        cache, _, _ = _build_radix_cache()
        for is_eagle in (False, True):
            for page_size in (1, 4):
                cache.is_eagle = is_eagle
                cache.page_size = page_size
                for kv_len in (0, 1, 7, 8, 9):
                    tokens = list(range(1, kv_len + 1))
                    expected = len(
                        RadixKey(
                            array("q", tokens), None, is_bigram=is_eagle
                        ).page_aligned(page_size)
                    )
                    self.assertEqual(
                        cache.finished_key_len(kv_len),
                        expected,
                        f"{is_eagle=}, {page_size=}, {kv_len=}",
                    )
                cache.is_eagle = False
                cache.page_size = 1

    def _eagle_finish(self, cache, allocator, pool, tokens, *, evict_on_finish):
        """Finish a request over raw ``tokens`` on an is_eagle cache."""
        match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        prefix_len = len(match.device_indices)
        cache.inc_lock_ref(match.last_device_node)
        req = Req(
            rid="eagle",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            evict_on_finish=evict_on_finish,
        )
        pool.alloc([req])
        req.output_ids = array("q")
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.cache_protected_len = prefix_len
        req.extra_key = None
        if prefix_len:
            pool.write((req.req_pool_idx, slice(0, prefix_len)), match.device_indices)
        if len(tokens) > prefix_len:
            pool.write(
                (req.req_pool_idx, slice(prefix_len, len(tokens))),
                allocator.alloc(len(tokens) - prefix_len),
            )
        req.kv_committed_len = len(tokens)
        req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(prefix_len, len(tokens))
        release_kv_cache(req, cache)

    def test_eagle_exact_hit_leaf_survives(self):
        """8 raw tokens -> 7 bigram key entries; an exact-hit flagged finish
        must not look like a strict extension."""
        cache, allocator, pool = _build_radix_cache()
        cache.is_eagle = True
        tokens = list(range(1, 9))
        self._eagle_finish(cache, allocator, pool, tokens, evict_on_finish=False)
        self.assertEqual(
            len(cache.match_prefix(MatchPrefixParams(key=_key(tokens))).device_indices),
            7,
        )
        used_before = KV_SIZE - allocator.available_size()

        self._eagle_finish(cache, allocator, pool, tokens, evict_on_finish=True)
        self.assertEqual(
            len(cache.match_prefix(MatchPrefixParams(key=_key(tokens))).device_indices),
            7,
        )
        self.assertEqual(KV_SIZE - allocator.available_size(), used_before)

    def test_eagle_strict_extension_still_frees_private_chain(self):
        cache, allocator, pool = _build_radix_cache()
        cache.is_eagle = True
        S = list(range(1, 9))
        a1 = list(range(101, 105))
        b1 = list(range(201, 205))
        self._eagle_finish(cache, allocator, pool, S + b1, evict_on_finish=False)
        self._eagle_finish(cache, allocator, pool, S + a1, evict_on_finish=False)

        a2 = list(range(105, 109))
        self._eagle_finish(cache, allocator, pool, S + a1 + a2, evict_on_finish=True)
        # The private a1 leaf is freed; the shared S prefix survives via b1.
        self.assertEqual(
            len(cache.match_prefix(MatchPrefixParams(key=_key(S + b1))).device_indices),
            len(S + b1) - 1,
        )
        self.assertEqual(
            len(
                cache.match_prefix(
                    MatchPrefixParams(key=_key(S + a1 + a2))
                ).device_indices
            ),
            len(S) - 1,
        )

    def test_page_unaligned_exact_hit_leaf_survives(self):
        """page_size=4: a 10-token request exact-hitting an 8-token leaf
        (matched_len=8, kv_len=10 -> logical 8) must not evict the leaf."""
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=4)
        )
        req_to_token_pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        kv_pool = MHATokenToKVPool(
            size=KV_SIZE,
            page_size=4,
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
            page_size=4,
            disable=False,
            eviction_policy="lru",
        )
        cache = RadixCache(params=params)
        tokens8 = list(range(1, 9))
        tokens10 = tokens8 + [50, 51]
        # Request A caches the 8-token leaf.
        self._eagle_finish(
            cache, allocator, req_to_token_pool, tokens8, evict_on_finish=False
        )
        leaf = cache.match_prefix(
            MatchPrefixParams(key=_key(tokens10))
        ).last_device_node
        self.assertIsNot(leaf, cache.root_node)
        # Request B exact-hits the leaf with a 2-token unaligned tail.
        self._eagle_finish(
            cache, allocator, req_to_token_pool, tokens10, evict_on_finish=True
        )
        self.assertEqual(
            len(
                cache.match_prefix(MatchPrefixParams(key=_key(tokens10))).device_indices
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
