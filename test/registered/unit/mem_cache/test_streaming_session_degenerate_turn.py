"""A streaming-session turn flagged ``spec_target_degenerate`` must not be saved
into the session slot for the next turn: its KV / mamba state is released like
a mid-processing abort, the slot is dropped, and the session's token history is
still advanced (``finish_req``) so the next turn re-prefills from scratch. As on
the healthy path, output_ids are trimmed to ``finished_len`` first so tokens the
verify batch committed past the stop (here the sentinel) never enter the next
turn's prompt."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.session.streaming_session import SessionSlot, StreamingSession


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = 1
        self.dec_lock_ref_calls = []

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("streaming requests must not reach the raw path")

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)

    def supports_mamba(self):
        return False


class _FakeMambaAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


SENTINEL = 163839


def _req(session_id, req_pool_idx, committed, allocated, finished):
    # Verify batch committed [11, SENTINEL, 22, 23]; the EOS match stops at the
    # sentinel, so finished_len=2 and the client only ever saw [11, SENTINEL].
    req = SimpleNamespace(
        rid=f"{session_id}-turn",
        req_pool_idx=req_pool_idx,
        kv_committed_len=committed,
        kv=SimpleNamespace(kv_allocated_len=allocated, swa_evicted_seqlen=0),
        origin_input_ids=list(range(committed)),
        output_ids=[11, SENTINEL, 22, 23],
        finished_len=2,
        finished_reason=None,
        evict_on_finish=False,
        last_node=None,
        cache_protected_len=0,
        swa_uuid_for_lock=None,
        skip_lock_node_ids={},
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
        spec_target_degenerate=True,
    )
    req.session = SimpleNamespace(
        session_id=session_id,
        streaming=True,
        finish_req=lambda r: finished.append(r.rid),
        abort_req=lambda rid: None,
    )
    req.evict_matched_len = lambda: 0
    return req


def _build(rows):
    req_to_token = torch.arange(rows * 128, dtype=torch.int32).reshape(rows, 128)
    pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    return pool, allocator, StreamingSession(_FakeInnerCache(pool, allocator))


def test_first_degenerate_turn_creates_no_slot_and_frees_kv():
    pool, allocator, cache = _build(rows=1)
    finished = []
    req = _req("s", req_pool_idx=0, committed=30, allocated=40, finished=finished)

    cache.cache_finished_req(req)

    assert "s" not in cache.slots
    assert req.req_pool_idx is None and req.kv is None
    assert pool.free_slots == [0]
    assert [t.tolist() for t in allocator.freed] == [list(range(40))]
    assert finished == ["s-turn"]
    assert req.output_ids == [11, SENTINEL]


def test_nth_degenerate_turn_drops_existing_slot_and_mamba_state():
    pool, allocator, cache = _build(rows=2)
    mamba = _FakeMambaAllocator()
    pool.mamba_allocator = mamba
    # Previous turn's slot; restore_to_req handed its pool idx / mamba state to the req.
    cache.slots["s"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=None,
        cache_protected_len=0,
        mamba_pool_idx=torch.tensor(7),
        mamba_ping_pong_track_buffer=torch.tensor([3, -1]),
    )
    finished = []
    req = _req("s", req_pool_idx=0, committed=60, allocated=65, finished=finished)
    req.mamba_pool_idx = torch.tensor(7)
    req.mamba_ping_pong_track_buffer = torch.tensor([3, -1])

    cache.cache_finished_req(req)

    assert "s" not in cache.slots
    assert pool.free_slots == [0]
    assert [t.tolist() for t in allocator.freed] == [list(range(65))]
    assert [t.tolist() for t in mamba.freed] == [[7], [3]]
    assert req.mamba_pool_idx is None and req.mamba_ping_pong_track_buffer is None
    assert finished == ["s-turn"]
    assert req.output_ids == [11, SENTINEL]


def test_degenerate_turn_without_finished_len_keeps_all_output():
    pool, allocator, cache = _build(rows=1)
    finished = []
    req = _req("s", req_pool_idx=0, committed=30, allocated=40, finished=finished)
    req.finished_len = None

    cache.cache_finished_req(req)

    assert "s" not in cache.slots
    assert req.output_ids == [11, SENTINEL, 22, 23]
    assert finished == ["s-turn"]


def test_healthy_turn_still_saves_slot():
    pool, allocator, cache = _build(rows=1)
    finished = []
    req = _req("s", req_pool_idx=0, committed=30, allocated=40, finished=finished)
    req.spec_target_degenerate = False

    cache.cache_finished_req(req)

    assert "s" in cache.slots and cache.slots["s"].req_pool_idx == 0
    assert pool.free_slots == []
    assert finished == ["s-turn"]
    assert req.output_ids == [11, SENTINEL]


@pytest.mark.parametrize(
    "existing_slot,first_matched_len", [(False, None), (True, 5), (True, None)]
)
def test_degenerate_turn_honors_requested_prefix_eviction(
    existing_slot, first_matched_len
):
    pool, allocator, cache = _build(rows=1)
    finished = []
    req = _req("s", req_pool_idx=0, committed=30, allocated=40, finished=finished)
    req.evict_on_finish = True
    lock_node = object()
    req.last_node = lock_node
    req.cache_protected_len = 8
    req.evict_matched_len = lambda: 6
    expected_matched_len = 6
    if existing_slot:
        cache.slots["s"] = SessionSlot(
            req_pool_idx=0,
            kv_committed_len=20,
            kv=SimpleNamespace(kv_allocated_len=20, swa_evicted_seqlen=0),
            last_node=lock_node,
            cache_protected_len=8,
            first_matched_len=first_matched_len,
        )
        req.last_node = cache.slots["s"].virtual_node
        expected_matched_len = first_matched_len if first_matched_len is not None else 8

    evictions = []

    def evict(node, **kwargs):
        # Cached prefix must be unlocked before eviction can reclaim it.
        assert cache.inner.dec_lock_ref_calls == [lock_node]
        assert "s" not in cache.slots
        evictions.append((node, kwargs))

    cache.inner.evict_finished_req_prefix = evict
    cache.inner.finished_key_len = lambda n: n - 1
    cache.cache_finished_req(req)

    assert evictions == [
        (lock_node, dict(matched_len=expected_matched_len, kv_len=31, rid=req.rid))
    ]
    assert [t.tolist() for t in allocator.freed] == [list(range(8, 40))]
    assert pool.free_slots == [0]
    assert req.output_ids == [11, SENTINEL]
    assert finished == ["s-turn"]
