from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ReqKvInfo,
    ScheduleBatch,
    release_req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 1
KV_SIZE = 512
MAMBA_SIZE = 8
VOCAB_SIZE = 32000


def _build():
    server_args = ServerArgs(model_path="dummy", page_size=PAGE)
    server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, PAGE)
    server_args.max_mamba_cache_size = MAMBA_SIZE
    server_args.mamba_max_states_per_path = 2
    set_global_server_args_for_scheduler(server_args)

    full_attention_layer_ids = [3, 7]
    mamba_layer_ids = [i for i in range(8) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layer_ids)
    with torch.device("cpu"):
        req_to_token_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=MAMBA_SIZE,
            mamba_spec_state_size=8,
            max_context_len=512,
            device="cpu",
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            speculative_num_draft_tokens=None,
        )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=2,
        head_dim=16,
        full_attention_layer_ids=full_attention_layer_ids,
        device="cpu",
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            disable=False,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            eviction_policy="lru",
        )
    )
    sessions = SimpleNamespace(sessions={})
    observer = SchedulerPoolStatsObserver(
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        session_controller=sessions,
        hisparse_coordinator=None,
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        enable_hisparse=False,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=KV_SIZE,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=PAGE,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=KV_SIZE,
        server_args=server_args,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        pool_stats_observer=observer,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    return server_args, cache, allocator, req_to_token_pool, observer, checker


def _recv(rid, input_ids, parent_rid=None):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        session_params=SessionParams(
            id="session-a",
            rid=parent_rid,
            offset=None,
            replace=False,
            drop_previous_output=False,
        ),
        lora_id=None,
        custom_logit_processor=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        evict_on_finish=False,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


def _prefill(req, cache, allocator, req_to_token_pool):
    key = RadixKey(array("q", req.origin_input_ids + req.output_ids))
    match = cache.match_prefix(MatchPrefixParams(key=key, req=req, cow_mamba=True))
    prefix_len = len(match.device_indices)
    lock_result = cache.inc_lock_ref(match.last_device_node)
    if req.req_pool_idx is None:
        req_to_token_pool.alloc([req])
    total = len(req.origin_input_ids) + len(req.output_ids)
    if prefix_len:
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, prefix_len)), match.device_indices
        )
    if total > prefix_len:
        req_to_token_pool.write(
            (req.req_pool_idx, slice(prefix_len, total)),
            allocator.alloc(total - prefix_len),
        )
    req.prefix_indices = match.device_indices
    req.last_node = match.last_device_node
    req.cache_protected_len = (
        match.cache_protected_len
        if match.cache_protected_len is not None
        else prefix_len
    )
    req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
    req.skip_lock_node_ids = lock_result.skip_lock_node_ids
    req.kv_committed_len = total
    req.kv = (
        ReqKvInfo(kv_allocated_len=total, swa_evicted_seqlen=0)
        if req.kv is None
        else req.kv
    )
    req.kv.kv_allocated_len = total
    req.full_untruncated_fill_ids = array("q", req.origin_input_ids + req.output_ids)
    req.set_extend_range(prefix_len, total)
    req.mamba_last_track_seqlen = 0
    if req.mamba_next_track_idx is None:
        req.mamba_next_track_idx = 0
    cache.cache_unfinished_req(req)


def _decode_step(req, allocator, req_to_token_pool, token):
    pos = req.kv.kv_allocated_len
    idx = allocator.alloc(1)
    req_to_token_pool.write((req.req_pool_idx, slice(pos, pos + 1)), idx)
    req.output_ids.append(token)
    req._refresh_fill_ids()
    req.kv_committed_len += 1
    req.kv.kv_allocated_len += 1


def _finish_turn(req, cache, finished_len):
    req.finished_reason = FINISH_LENGTH(length=finished_len)
    req.finished_len = finished_len
    from sglang.srt.mem_cache.common import release_kv_cache

    release_kv_cache(req, cache)


def _scheduler_stub(cache):
    output = Mock()
    scheduler = SimpleNamespace(
        chunked_req=None,
        _pending_chunked_abort_req=None,
        waiting_queue=[],
        enable_hicache_storage=False,
        tree_cache=cache,
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=output)
        ),
        disaggregation_mode=DisaggregationMode.NULL,
        dllm_config=None,
        enable_overlap=False,
        result_queue=deque(),
        grammar_manager=SimpleNamespace(abort_requests=Mock()),
        ps=SimpleNamespace(pp_size=1),
        running_batch=SimpleNamespace(reqs=[]),
        last_batch=SimpleNamespace(reqs=[]),
    )
    scheduler._release_dropped_waiting_req_mm_inputs = (
        Scheduler._release_dropped_waiting_req_mm_inputs.__get__(scheduler)
    )
    return scheduler


class TestSessionQueueAbort(CustomTestCase):
    def _setup_first_turn(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
        ) = _build()
        session = Session(capacity_of_str_len=0, session_id="session-a", streaming=True)
        req1 = session.create_req(
            _recv("turn-1", list(range(16))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        _prefill(req1, cache, allocator, req_to_token_pool)
        _finish_turn(req1, cache, finished_len=0)
        self.assertTrue(cache.session.has_slot(session.session_id))
        return (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        )

    def _assert_idle(self, observer, checker):
        pool_stats = observer.get_pool_stats()
        mamba_leak, mamba_msg = checker._check_mamba_pool(pool_stats)
        self.assertFalse(mamba_leak, mamba_msg)
        all_leak, all_messages = checker._check_all_pools(pool_stats)
        self.assertFalse(all_leak, all_messages)

    def test_queue_abort_of_restored_turn_finishes_and_releases_slot(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertIsNotNone(req.req_pool_idx)

        scheduler = _scheduler_stub(cache)
        scheduler.waiting_queue.append(req)
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertNotIn(session.session_id, cache.session.slots)
        self.assertEqual(cache.session.session_held_mamba_slots(), 0)
        self._assert_idle(observer, checker)

        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_queue_abort_of_retracted_turn_finishes(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertIsNotNone(req.req_pool_idx)
        release_req(
            req=req,
            remaing_req_count=1,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.req_pool_idx)

        scheduler = _scheduler_stub(cache)
        scheduler.waiting_queue.append(req)
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retracted_queue_abort_in_decode_mode_finishes(self):
        """PD decode: a retracted req (KV already nuked by release_req) sits in
        disagg_decode_prealloc_queue.retracted_queue with kv_cache_cpu. Aborting
        it must stamp FINISH_ABORT and clear the session's inflight turn."""
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        release_req(
            req=req,
            remaing_req_count=1,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.kv)

        req.kv_cache_cpu = torch.empty(0)
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[req], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertFalse(hasattr(req, "kv_cache_cpu"))
        send_output.assert_called_once()
        self.assertIsInstance(send_output.call_args[0][0], AbortReq)
        self.assertEqual(send_output.call_args[0][0].rid, req.rid)
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_oom_retract_of_last_streaming_turn_aborts_session_turn(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)

        batch = ScheduleBatch(reqs=[req])
        batch.req_to_token_pool = req_to_token_pool
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = cache
        batch.hisparse_coordinator = None
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)

        with patch.object(batch, "check_decode_mem", return_value=False):
            with patch.object(batch, "filter_batch"):
                with patch.object(
                    NewTokenRatioTracker,
                    "estimate_new_token_ratio_after_retract",
                    return_value=0.0,
                ):
                    retracted, _ratio, reqs_to_abort = batch.retract_decode(server_args)

        self.assertEqual(reqs_to_abort, [req])
        self.assertEqual(retracted, [])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.req_pool_idx)
        self.assertFalse(session.has_unfinished_request())
        self.assertFalse(cache.session.has_slot(session.session_id))
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_bootstrap_failure_before_allocation_aborts_session_turn(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.kv)
        self.assertIsNone(req.mamba_pool_idx)
        req.disagg_kv_sender = SimpleNamespace(failure_exception=lambda: None)
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0),
            tree_cache=cache,
            req_to_metadata_buffer_idx_allocator=None,
            output_streamer=SimpleNamespace(stream_output=lambda reqs, rl: None),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            enable_hicache_storage=False,
        )

        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retract_readmit_finish_matches_no_retract(self):
        worlds = []
        for retract in (False, True):
            (
                _server_args,
                cache,
                allocator,
                req_to_token_pool,
                observer,
                checker,
                session,
            ) = self._setup_first_turn()
            req = session.create_req(
                _recv("turn-2", list(range(32, 48))),
                tokenizer=None,
                vocab_size=VOCAB_SIZE,
            )
            req.init_next_round_input(cache)
            _prefill(req, cache, allocator, req_to_token_pool)
            _decode_step(req, allocator, req_to_token_pool, 100)
            _decode_step(req, allocator, req_to_token_pool, 101)
            if retract:
                release_req(
                    req=req,
                    remaing_req_count=1,
                    server_args=_server_args,
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=allocator,
                    tree_cache=cache,
                    hisparse_coordinator=None,
                    offload_kv=False,
                )
                self.assertFalse(cache.session.has_slot(session.session_id))
                self.assertEqual(set(session.req_nodes), {"turn-1"})
                self.assertTrue(session.has_unfinished_request())
                self.assertEqual(cache.session.session_held_mamba_slots(), 0)
                self._assert_idle(observer, checker)
                req.init_next_round_input(cache)
                _prefill(req, cache, allocator, req_to_token_pool)
                _decode_step(req, allocator, req_to_token_pool, 102)
                _decode_step(req, allocator, req_to_token_pool, 103)
            else:
                _decode_step(req, allocator, req_to_token_pool, 102)
                _decode_step(req, allocator, req_to_token_pool, 103)
            _finish_turn(req, cache, finished_len=4)
            worlds.append((cache, observer, checker, session, req))

        cache_a, observer_a, checker_a, session_a, req_a = worlds[0]
        cache_b, observer_b, checker_b, session_b, req_b = worlds[1]
        self.assertEqual(set(session_a.req_nodes), set(session_b.req_nodes))
        self.assertEqual(list(req_a.output_ids), list(req_b.output_ids))
        self.assertEqual(session_a.committed_origin_len, session_b.committed_origin_len)
        self.assertEqual(
            session_a.committed_unpadded_len, session_b.committed_unpadded_len
        )
        self.assertEqual(session_a.committed_fill_len, session_b.committed_fill_len)
        slot_a = cache_a.session.slots[session_a.session_id]
        slot_b = cache_b.session.slots[session_b.session_id]
        self.assertEqual(slot_a.kv_committed_len, slot_b.kv_committed_len)
        self.assertEqual(slot_a.kv.kv_allocated_len, slot_b.kv.kv_allocated_len)
        self.assertFalse(session_a.has_unfinished_request())
        self.assertFalse(session_b.has_unfinished_request())
        self.assertEqual(
            cache_a.session.session_held_mamba_slots(),
            cache_b.session.session_held_mamba_slots(),
        )
        self._assert_idle(observer_a, checker_a)
        self._assert_idle(observer_b, checker_b)
        next_a = session_a.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        next_b = session_b.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertEqual(list(next_a.origin_input_ids), list(next_b.origin_input_ids))

    def test_retract_then_queue_abort_keeps_previous_checkpoint(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        turn1 = session.req_nodes["turn-1"].req
        checkpoint_origin = list(turn1.origin_input_ids)
        checkpoint_output = list(turn1.output_ids)
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        _decode_step(req, allocator, req_to_token_pool, 100)
        _decode_step(req, allocator, req_to_token_pool, 101)
        release_req(
            req=req,
            remaing_req_count=1,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        scheduler = _scheduler_stub(cache)
        scheduler.waiting_queue.append(req)
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self.assertIs(session.req_nodes["turn-1"].req.session, session)
        self._assert_idle(observer, checker)
        self.assertEqual(cache.session.session_held_mamba_slots(), 0)
        follow_up = session.create_req(
            _recv("turn-3", [99], parent_rid="turn-1"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)
        self.assertEqual(
            list(follow_up.origin_input_ids),
            checkpoint_origin + checkpoint_output + [99],
        )

    def test_unflagged_release_of_unfinished_req_is_a_normal_finish(self):
        # Disaggregated prefill releases the KV of a successful transfer
        # *before* stamping FINISH_LENGTH; without explicit retract intent
        # that must still commit the turn (slot saved, finish_req ran).
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        self.assertIsNone(req.finished_reason)
        from sglang.srt.mem_cache.common import release_kv_cache

        release_kv_cache(req, cache)
        req.finished_reason = FINISH_LENGTH(length=0)
        req.finished_len = 0

        self.assertTrue(cache.session.has_slot(session.session_id))
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-2"})
        self.assertIs(session.req_nodes["turn-2"].req, req)
        self.assertIsNone(req.req_pool_idx)
        self.assertGreater(cache.session.session_held_mamba_slots(), 0)
        committed = list(req.origin_input_ids) + list(req.output_ids)
        follow_up = session.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)
        self.assertEqual(list(follow_up.origin_input_ids), committed + [99])
        cache.session.release_session(session.session_id)
        self._assert_idle(observer, checker)

    def test_prealloc_queue_abort_in_decode_mode_finishes_session_turn(self):
        """PD decode: a prealloc-queue req has no KV yet, so release_kv_cache
        never runs and nothing clears the session inflight turn. The abort
        path must stamp FINISH_ABORT and clear the turn itself."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.kv)

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[decode_req], retracted_queue=[], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        decode_req.kv_receiver.abort.assert_called_once()
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.message, "Aborted")
        # The session stays inflight until the queue actually finishes the
        # request — a follow-up must not be admitted before that.
        self.assertTrue(session.has_unfinished_request())
        # Removal + output streaming are pop_preallocated's job, not abort's.
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.queue, [decode_req])
        send_output.assert_not_called()

        # Drive the prealloc queue's abort-scan: it finishes the req, clears
        # the receiver, and unblocks the session.
        prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc_queue.queue = [decode_req]
        prealloc_queue.pending_reqs = []
        prealloc_queue.retracted_queue = []
        prealloc_queue._resolve_pending_reqs = MagicMock()
        prealloc_queue._update_handshake_waiters = MagicMock()
        prealloc_queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        prealloc_queue._allocatable_token_budgets = MagicMock(return_value=0)
        prealloc_queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        prealloc_queue.scheduler = scheduler

        preallocated, failed = prealloc_queue.pop_preallocated()

        self.assertEqual(failed, [decode_req])
        self.assertEqual(prealloc_queue.queue, [])
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_transfer_queue_abort_in_decode_mode_finishes_session_turn(self):
        """PD decode: a transfer-queue req holds real KV; abort must stamp
        FINISH_ABORT and clear the session turn while leaving the KV for
        pop_transferred's Failed branch to release."""
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)

        decode_req = SimpleNamespace(req=req, kv_receiver=Mock())
        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[decode_req])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        decode_req.kv_receiver.abort.assert_called_once()
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.message, "Aborted")
        # The session stays inflight until pop_transferred finishes the req
        # and releases its KV.
        self.assertTrue(session.has_unfinished_request())
        self.assertEqual(scheduler.disagg_decode_transfer_queue.queue, [decode_req])
        send_output.assert_not_called()
        self.assertIsNotNone(req.req_pool_idx)

        # Drive the transfer queue's Failed branch (the aborted receiver
        # polls Failed): releases KV, unblocks the session, and must NOT
        # count a user abort as a transfer failure.
        decode_req.metadata_buffer_index = 3
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = [decode_req]
        transfer_queue.enable_staging = False
        transfer_queue.gloo_group = MagicMock()
        transfer_queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        transfer_queue.tp_rank = 0
        transfer_queue.tree_cache = cache
        transfer_queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        transfer_queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        transfer_queue._clean_hicache_prefetch_resources = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.server_args = MagicMock()
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        transfer_queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            transferred = transfer_queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(transfer_queue.queue, [])
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_not_called()
        self.assertFalse(session.has_unfinished_request())
        self.assertIsNone(req.req_pool_idx)
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_abort_all_over_multiple_retracted_and_queued_decode_reqs(self):
        """abort_all must hit retracted, prealloc, and transfer reqs in one
        pass, tolerating plain Reqs with no session attached."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            _session,
        ) = self._setup_first_turn()

        def _plain_req(rid):
            return Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
                vocab_size=VOCAB_SIZE,
            )

        retracted_req = _plain_req("retracted-1")
        retracted_req.kv_cache_cpu = torch.empty(0)
        prealloc_req = _plain_req("prealloc-1")
        transfer_req = _plain_req("transfer-1")
        prealloc_decode_req = SimpleNamespace(req=prealloc_req, kv_receiver=Mock())
        transfer_decode_req = SimpleNamespace(req=transfer_req, kv_receiver=Mock())

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[prealloc_decode_req],
            retracted_queue=[retracted_req],
            held_rebootstrap_reqs=[],
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[transfer_decode_req]
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))

        for req in (retracted_req, prealloc_req, transfer_req):
            self.assertTrue(req.finished())
            self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertFalse(hasattr(retracted_req, "kv_cache_cpu"))
        send_output.assert_called_once()
        self.assertIsInstance(send_output.call_args[0][0], AbortReq)
        self.assertEqual(send_output.call_args[0][0].rid, "retracted-1")
        prealloc_decode_req.kv_receiver.abort.assert_called_once()
        transfer_decode_req.kv_receiver.abort.assert_called_once()

    def test_retracted_queue_abort_retry_after_send_failure(self):
        """If send_output raises, the req must stay fully intact in
        retracted_queue so a retry re-runs idempotent steps: the
        kv_cache_cpu del happens only after a successful send."""
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        release_req(
            req=req,
            remaing_req_count=1,
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        req.kv_cache_cpu = torch.empty(0)

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[req], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        send_output.side_effect = [RuntimeError("ipc"), None]

        with self.assertRaises(RuntimeError):
            Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
        self.assertTrue(hasattr(req, "kv_cache_cpu"))
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [req])

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertFalse(hasattr(req, "kv_cache_cpu"))
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_abort_all_partial_retracted_failure_keeps_failed_entry_retryable(self):
        """A multi-entry retracted abort must commit each removal in place:
        if send_output raises for entry B after A succeeded, the queue must
        already hold only [B] so a retry does not trip A's kv_cache_cpu
        assert."""
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            _session,
        ) = self._setup_first_turn()

        def _plain_req(rid):
            req = Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
                vocab_size=VOCAB_SIZE,
            )
            req.kv_cache_cpu = torch.empty(0)
            return req

        req_a = _plain_req("retracted-a")
        req_b = _plain_req("retracted-b")

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[req_a, req_b], held_rebootstrap_reqs=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        send_output.side_effect = [None, RuntimeError("ipc down")]

        with self.assertRaises(RuntimeError):
            Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))
        self.assertEqual(
            scheduler.disagg_decode_prealloc_queue.retracted_queue, [req_b]
        )
        self.assertFalse(hasattr(req_a, "kv_cache_cpu"))
        self.assertTrue(hasattr(req_b, "kv_cache_cpu"))

        send_output.side_effect = None
        Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.retracted_queue, [])
        self.assertFalse(hasattr(req_b, "kv_cache_cpu"))
        self.assertTrue(req_a.finished())
        self.assertTrue(req_b.finished())
        self.assertEqual(send_output.call_count, 3)

    def test_prefill_bootstrap_queue_abort_finishes_session_turn(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.req_pool_idx)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[req])
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()
        scheduler.req_to_metadata_buffer_idx_allocator = None

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIsNone(req.finished_reason.status_code)
        self.assertFalse(session.has_unfinished_request())
        req.disagg_kv_sender.abort.assert_called_once()
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [req])

        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertIsNone(req.finished_reason.status_code)
        scheduler.metrics_collector.increment_bootstrap_failed_reqs.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_prefill_inflight_queue_abort_stamps_user_abort(self):
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_inflight_queue = [req]
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertIsNone(req.finished_reason.status_code)
        req.disagg_kv_sender.abort.assert_called_once()
        # The sender's failure_exception() is also its cleanup; it must still
        # run for a user abort, with its exception swallowed.
        req.disagg_kv_sender.failure_exception.side_effect = RuntimeError("aborted")

        exc = SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )

        req.disagg_kv_sender.failure_exception.assert_called_once_with()
        self.assertIsNone(exc)
        self.assertIsNone(req.finished_reason.status_code)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_not_called()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_natural_prefill_transfer_failure_still_reports_500(self):
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        req.disagg_kv_sender = Mock()
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )

        scheduler = _scheduler_stub(cache)
        scheduler.ps.tp_rank = 0
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_inflight_queue = [req]
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=True)
        scheduler.metrics_collector = Mock()

        exc = SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )

        self.assertIsNone(exc)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 500)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_called_once()
        self.assertFalse(session.has_unfinished_request())
        self._assert_idle(observer, checker)

    def test_held_rebootstrap_abort_targeted_and_abort_all(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req_a = SimpleNamespace(
            rid="held-a",
            session=None,
            multimodal_inputs=None,
            finished_reason=None,
            return_logprob=False,
        )
        req_b = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertTrue(session.has_unfinished_request())

        scheduler = _scheduler_stub(cache)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[],
            retracted_queue=[],
            held_rebootstrap_reqs=[req_a, req_b],
            add=Mock(),
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req_a.rid))
        held = scheduler.disagg_decode_prealloc_queue.held_rebootstrap_reqs
        self.assertEqual(held, [req_b])
        self.assertIsInstance(req_a.finished_reason, FINISH_ABORT)
        send_output.assert_called_once()
        self.assertEqual(send_output.call_args[0][0].rid, req_a.rid)
        self.assertTrue(session.has_unfinished_request())

        Scheduler.abort_request(scheduler, AbortReq(rid="", abort_all=True))
        self.assertEqual(held, [])
        self.assertIsInstance(req_b.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(send_output.call_count, 2)

        DecodePreallocQueue.enqueue_held_rebootstrap(
            scheduler.disagg_decode_prealloc_queue
        )
        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()
        self._assert_idle(observer, checker)

    def test_dllm_queue_abort_stamps_finish_abort_and_clears_session(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertTrue(session.has_unfinished_request())

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = object()
        scheduler.dllm_manager = SimpleNamespace(
            pop_aborted_reqs=lambda abort_all, rid: [req]
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        send_output.assert_called_once()
        self.assertIsInstance(send_output.call_args[0][0], AbortReq)
        self.assertEqual(send_output.call_args[0][0].rid, req.rid)
        self._assert_idle(observer, checker)

    def test_dllm_queue_abort_skips_already_finished_req(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        # Finished in the last forward but not yet dropped by
        # filter_finished_reqs(); the abort must not restamp it.
        req.finished_reason = FINISH_LENGTH(length=1)
        req.multimodal_inputs = Mock()

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = object()
        scheduler.dllm_manager = SimpleNamespace(
            pop_aborted_reqs=lambda abort_all, rid: [req]
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertIsNotNone(req.multimodal_inputs)
        send_output.assert_not_called()

    def test_dllm_abort_defers_req_with_pending_overlap_result(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            _observer,
            _checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        self.assertTrue(session.has_unfinished_request())

        from sglang.srt.dllm.mixin.scheduler import DllmManager

        scheduler = _scheduler_stub(cache)
        scheduler.enable_overlap = True
        scheduler.result_queue = deque([(SimpleNamespace(reqs=[req]), None)])
        scheduler.dllm_config = object()
        scheduler.dllm_manager = DllmManager()
        scheduler.dllm_manager.staging_queue = [req]
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertFalse(req.finished())
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(scheduler.dllm_manager.staging_queue, [req])
        self.assertTrue(session.has_unfinished_request())
        send_output.assert_not_called()
        self.assertIsNotNone(req.req_pool_idx)

        # The same req with no pending result takes the direct abort path.
        req.to_finish = None
        scheduler.enable_overlap = False
        scheduler.result_queue = deque()

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        send_output.assert_called_once()

    def _dllm_result_stub(self, fdfo, empty=True):
        return SimpleNamespace(
            copy_done=None,
            next_token_ids=(
                [torch.zeros(4, dtype=torch.long)]
                if fdfo
                else [torch.tensor([], dtype=torch.long)]
            ),
            accept_length_per_req_cpu=[0] if fdfo else None,
            dllm_algo_state=None,
            can_run_cuda_graph=False,
        )

    def _run_process_batch_result_dllm(self, session, cache, req, fdfo):
        from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin

        scheduler = _scheduler_stub(cache)
        scheduler.dllm_config = SimpleNamespace(
            first_done_first_out_mode=fdfo, block_size=4
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=Mock(), free_group_end=Mock()
        )
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        scheduler.metrics_reporter = SimpleNamespace(
            num_generated_tokens=0, report_prefill_stats=Mock()
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release_mock:
            SchedulerDllmMixin.process_batch_result_dllm(
                scheduler, batch, self._dllm_result_stub(fdfo)
            )
        return scheduler, release_mock

    def test_dllm_process_result_finalizes_to_finish_on_empty_result(self):
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                (
                    _server_args,
                    cache,
                    _allocator,
                    _req_to_token_pool,
                    _observer,
                    _checker,
                    session,
                ) = self._setup_first_turn()
                req = session.create_req(
                    _recv("turn-2", list(range(32, 48))),
                    tokenizer=None,
                    vocab_size=VOCAB_SIZE,
                )
                req.to_finish = FINISH_ABORT()
                req.time_stats = SimpleNamespace(
                    set_completion_time=Mock(), set_first_token_time=Mock()
                )

                scheduler, release_mock = self._run_process_batch_result_dllm(
                    session, cache, req, fdfo
                )

                self.assertTrue(req.finished())
                self.assertIsInstance(req.finished_reason, FINISH_ABORT)
                self.assertIsNone(req.to_finish)
                release_mock.assert_called_once()
                scheduler.output_streamer.stream_output.assert_called_once_with(
                    [req], False
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
