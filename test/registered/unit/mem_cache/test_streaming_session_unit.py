import time
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.io_struct import SessionReapPlan
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.session.session_controller import Session, SessionController
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.match_results = list(match_results or [])
        self.dec_lock_ref_calls = []
        self.dec_lock_ref_params = []

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def match_prefix(self, *args, **kwargs):
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)
        self.dec_lock_ref_params.append(args[0] if args else kwargs.get("params"))

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeSessionTreeCache:
    def __init__(self):
        self.released = []

    def release_session(self, session_id):
        self.released.append(session_id)


class _FinishedReq:
    def finished(self):
        return True


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.rid = session_id
        self.session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            _inflight=False,
            _inflight_rid=None,
        )

        def abort_req(rid):
            if self.session._inflight_rid == rid:
                self.session._inflight = False
                self.session._inflight_rid = None

        self.session.abort_req = abort_req
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(
            kv_allocated_len=allocated,
            swa_evicted_seqlen=0,
        )
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.skip_lock_node_ids = {}
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None
        self.spec_target_degenerate = False


def test_preabort_detaches_session_and_preserves_slot():
    """Pre-aborted req (to_finish set before match_prefix) is detached from
    the session: session=None, abort_req(rid) called. Slot stays intact."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_indices=torch.tensor([], dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            )
        ],
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv=SimpleNamespace(kv_allocated_len=48, swa_evicted_seqlen=0),
        cache_protected_len=16,
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    # Req detached from session.
    assert req.session is None
    # Slot untouched.
    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_preabort_detaches_without_slot():
    """Pre-aborted req detaches even when its session has no active slot."""
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    raw_result = MatchResult(
        device_indices=torch.tensor([], dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
    )
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[raw_result],
    )
    tree_cache = StreamingSession(inner)
    req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)
    req.session._inflight_rid = req.rid
    aborted = []
    original_abort_req = req.session.abort_req

    def record_abort_req(rid):
        aborted.append(rid)
        original_abort_req(rid)

    req.session.abort_req = record_abort_req
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    assert req.session is None
    assert aborted == ["session-a"]
    assert result is raw_result
    assert len(result.device_indices) == 0
    assert tree_cache.slots == {}


def test_preabort_of_non_inflight_req_keeps_inflight():
    """Only the matching in-flight request may clear session state."""

    def match_preaborted_req(rid):
        req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
        req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
        raw_result = MatchResult(
            device_indices=torch.tensor([], dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )
        inner = _FakeInnerCache(
            req_to_token_pool,
            _FakeAllocator(),
            page_size=16,
            match_results=[raw_result],
        )
        tree_cache = StreamingSession(inner)
        req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)
        req.rid = rid
        req.session._inflight = True
        req.session._inflight_rid = "turn-1"
        abort_req = Mock()
        real_abort_req = req.session.abort_req

        def record_abort_req(request_rid):
            if req.session._inflight_rid == request_rid:
                abort_req(request_rid)
            real_abort_req(request_rid)

        req.session.abort_req = record_abort_req
        req.to_finish = FINISH_ABORT("too long")

        tree_cache.match_prefix(
            SimpleNamespace(
                req=req,
                key=SimpleNamespace(token_ids=list(range(64))),
            )
        )
        return req, abort_req

    non_inflight_req, abort_req = match_preaborted_req("stub")
    assert non_inflight_req.session is None
    abort_req.assert_not_called()

    inflight_req, abort_req = match_preaborted_req("turn-1")
    assert inflight_req.session is None
    abort_req.assert_called_once_with("turn-1")


def test_first_mid_abort_nukes_ephemeral_slot():
    """First-request mid-processing abort: no slot exists yet, ephemeral
    slot is created from req state and nuked via release_session."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # No slot exists yet (first request).
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=20)
    req.finished_reason = FINISH_ABORT("input too long")

    tree_cache.cache_finished_req(req)

    # Slot must NOT be created.
    assert "session-a" not in tree_cache.slots
    # Transient pool slot freed.
    assert req.req_pool_idx is None
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(20))


def test_nth_mid_abort_nukes_session_slot():
    """Nth-request mid-processing abort: slot exists, restore_to_req ran.
    ALL KV is wiped (release_session). Slot is deleted. Token IDs stay
    in req_nodes for next turn's re-prefill."""
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # Session already has a slot from a previous turn.
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=None,
        cache_protected_len=0,
    )

    # Mid-processing abort: req has the SESSION slot's pool_idx (restore_to_req ran).
    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    # Slot wiped — deleted from slots dict.
    assert "session-a" not in tree_cache.slots
    # All KV freed: [0, 65) from release_session (slot extended to req's allocated).
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(65))
    # Pool slot returned.
    assert req_to_token_pool.free_slots == [0]
    assert req.req_pool_idx is None


def test_release_session_threads_mamba_skip_ids():
    """release_session must forward the slot's skip_lock_node_ids to
    dec_lock_ref. The first req's last_node may be full-only-locked (mamba
    skipped at inc), so without the skip set the release would drop a mamba
    lock the session never took -- another request's, on a shared node."""
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    tree_cache = StreamingSession(inner)

    lock_node = SimpleNamespace(id=42)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=lock_node,
        cache_protected_len=0,
        skip_lock_node_ids={ComponentType.MAMBA: {42}},
    )

    tree_cache.release_session("session-a")

    assert inner.dec_lock_ref_calls == [lock_node]
    params = inner.dec_lock_ref_params[0]
    assert params is not None
    assert params.skip_lock_node_ids.get(ComponentType.MAMBA) == {42}


def test_release_session_skips_lazy_ping_pong_sentinels():
    allocator = MambaSlotAllocator(size=8, device="cpu")
    allocator.alloc(8)
    req_to_token_pool = SimpleNamespace(mamba_allocator=allocator)
    inner = _FakeInnerCache(
        req_to_token_pool,
        _FakeAllocator(),
        page_size=1,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        mamba_pool_idx=torch.tensor(3),
        mamba_ping_pong_track_buffer=torch.tensor([5, -1]),
    )

    tree_cache.release_session("session-a")

    assert set(allocator.free_slots.tolist()) == {3, 5}
    assert -1 not in allocator.free_slots.tolist()
    assert allocator.available_size() == 2


def test_session_held_mamba_slots_ignores_sentinels():
    req_to_token_pool = SimpleNamespace(mamba_allocator=_FakeAllocator())
    inner = _FakeInnerCache(
        req_to_token_pool,
        _FakeAllocator(),
        page_size=1,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        mamba_pool_idx=torch.tensor(3),
        mamba_ping_pong_track_buffer=torch.tensor([5, -1]),
    )
    assert tree_cache.session_held_mamba_slots() == 2

    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-b"] = SessionSlot(
        mamba_pool_idx=torch.tensor(4),
        mamba_ping_pong_track_buffer=torch.tensor([-1, -1]),
    )
    assert tree_cache.session_held_mamba_slots() == 1


def test_session_controller_plan_reap_defers_application():
    tree_cache = _FakeSessionTreeCache()
    controller = SessionController(tree_cache)
    ready = Session(16, "ready")
    ready.close_on_finish = True
    ready.req_nodes["req"] = SimpleNamespace(req=_FinishedReq())
    timed_out = Session(16, "timed-out", timeout=1)
    timed_out.last_active_time = time.monotonic() - 2
    controller.sessions.update({"ready": ready, "timed-out": timed_out})

    controller._last_reap_time = 10
    assert controller.plan_reap(10.5) is None
    plan = controller.plan_reap(12)

    assert plan == SessionReapPlan(
        deferred=["ready"],
        timed_out=["timed-out"],
    )
    assert set(controller.sessions) == {"ready", "timed-out"}
    assert tree_cache.released == []


def test_session_controller_apply_reap_filters_stale_sessions():
    tree_cache = _FakeSessionTreeCache()
    controller = SessionController(tree_cache)
    deferred = Session(16, "deferred")
    deferred.close_on_finish = True
    not_deferred = Session(16, "not-deferred")
    timed_out = Session(16, "timed-out")
    controller.sessions.update(
        {
            "deferred": deferred,
            "not-deferred": not_deferred,
            "timed-out": timed_out,
        }
    )

    controller.apply_reap(
        SessionReapPlan(
            deferred=["deferred", "missing", "not-deferred"],
            timed_out=["timed-out", "missing"],
        )
    )

    assert tree_cache.released == ["deferred", "timed-out"]
    assert set(controller.sessions) == {"not-deferred"}


def test_session_controller_apply_reap_is_rank_symmetric():
    controllers = []
    for _ in range(2):
        tree_cache = _FakeSessionTreeCache()
        controller = SessionController(tree_cache)
        deferred = Session(16, "deferred")
        deferred.close_on_finish = True
        controller.sessions.update(
            {"deferred": deferred, "untouched": Session(16, "untouched")}
        )
        controllers.append(controller)

    plan = SessionReapPlan(deferred=["deferred"], timed_out=[])
    for controller in controllers:
        controller.apply_reap(plan)

    assert set(controllers[0].sessions) == set(controllers[1].sessions) == {"untouched"}


def test_request_receiver_classifies_session_reap_plan_as_work():
    receiver = object.__new__(SchedulerRequestReceiver)
    plan = SessionReapPlan(deferred=[], timed_out=[])

    work, control = receiver._split_work_and_control_reqs([plan])

    assert work == [plan]
    assert control == []


# Shrink tests removed: streaming sessions are append-only after the
# rollback fix in session_controller (rollback_aborted_req).  The shrink
# code path in cache_finished_req no longer exists.


def test_trim_overshoot_postcondition():
    """`_trim_overshoot` postcondition: every per-req KV field is capped at
    target = origin+finished_len, output_ids is truncated, and the tail
    KV slots are freed. Covers both non-SWA fields (kv_committed_len,
    kv_allocated_len, output_ids) and SWA bookkeeping (swa_evicted_seqlen)
    in one shot — same invariant `_free_tail` enforces on the match_prefix
    path.
    """
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )

    # Overshoot scenario: origin=26, finished_len=12 -> target=38.
    # committed=40 (overshoot 2), allocated=44, swa_evicted=42 (> target),
    # output_ids extended to 14 by the overshoot round.
    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 42

    tree_cache._trim_overshoot(req, finished_len=12)

    target = 38
    assert req.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.swa_evicted_seqlen == target
    assert len(req.output_ids) == 12
    # Tail [38, 44) freed by _free_kv_aligned.
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(38, 44))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
