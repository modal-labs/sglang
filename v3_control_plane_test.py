"""Dependency-light checks for the HiCache V3 control plane (phase 1).

Covers docs/HICACHE_V3.md phase 1: rank-0 authority worker + record
publication riding the request broadcast, peer record application, CRC
fail-stop, the flush rule, and the mamba lazy-alloc fail-stop that replaced
the V2 consensus.

The production package is not imported (an SGLang import pulls GPU deps).
hicache_authority.py is import-light and loaded directly via importlib;
schedule_batch method bodies are extracted with ``ast`` and executed against
small fakes, in the style of mamba_repair_test.py / r1_reconcile_test.py.

Run: ``python3 v3_control_plane_test.py`` (torch required).
"""

from __future__ import annotations

import ast
import importlib.util
import textwrap
import time
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parent
AUTHORITY_PATH = (
    REPO / "python/sglang/srt/mem_cache/hybrid_cache/hicache_authority.py"
)
CACHE_PATH = REPO / "python/sglang/srt/mem_cache/unified_radix_cache.py"
SCHEDULE_BATCH_PATH = REPO / "python/sglang/srt/managers/schedule_batch.py"
RECEIVER_PATH = (
    REPO / "python/sglang/srt/managers/scheduler_components/request_receiver.py"
)

import sys

_spec = importlib.util.spec_from_file_location("hicache_authority", AUTHORITY_PATH)
authority_mod = importlib.util.module_from_spec(_spec)
sys.modules["hicache_authority"] = authority_mod
_spec.loader.exec_module(authority_mod)

HiCacheAuthority = authority_mod.HiCacheAuthority
HiCacheRecordBatch = authority_mod.HiCacheRecordBatch
CacheRecord = authority_mod.CacheRecord
WriteIntent = authority_mod.WriteIntent
fold_crc = authority_mod.fold_crc
indices_crc = authority_mod.indices_crc
RECORD_PLACE = authority_mod.RECORD_PLACE
RECORD_EVICT = authority_mod.RECORD_EVICT
RECORD_COMPLETE = authority_mod.RECORD_COMPLETE
OP_WRITE = authority_mod.OP_WRITE
OP_LOAD = authority_mod.OP_LOAD

PASSED = 0


def check(condition, message):
    global PASSED
    if not condition:
        raise AssertionError(message)
    PASSED += 1


# ---- fakes -------------------------------------------------------------------


class FakeEvent:
    def __init__(self, fired=True):
        self.fired = fired

    def query(self):
        return self.fired

    def synchronize(self):
        check(self.fired or True, "")  # synchronize is always legal


class FakeAck:
    def __init__(self, node_ids, num_tokens=0):
        self.node_ids = list(node_ids)
        self.num_tokens = num_tokens
        self.finish_event = FakeEvent()
        self.timing_enabled = False


class FakeController:
    def __init__(self, fail_reservations=0):
        self.ack_write_queue = []
        self.ack_load_queue = []
        self.committed = []
        self.fail_reservations = fail_reservations
        self._next_host = 0

    def reserve_write(self, device_indices, node_id=-1, extra_pools=None, *,
                      allow_evict=False, priority=None):
        if self.fail_reservations > 0:
            self.fail_reservations -= 1
            return None
        n = len(device_indices)
        host = torch.arange(self._next_host, self._next_host + n, dtype=torch.int64)
        self._next_host += n
        for pool in extra_pools or []:
            if pool.host_indices is None and pool.device_indices is not None:
                pool.host_indices = torch.tensor([7] * len(pool.device_indices))
        return SimpleNamespace(
            host_indices=host, device_indices=device_indices, node_id=node_id,
            extra_pools=list(extra_pools or []),
        )

    def commit_write(self, reservation):
        self.committed.append(reservation)
        self.ack_write_queue.append(FakeAck([reservation.node_id]))
        return reservation.host_indices


class FakeApplierCache:
    """Records every record-apply; returns deterministic observed tuples so
    both sides of the CRC fold can be reproduced in tests."""

    def __init__(self):
        self.applied = []
        self.evict_results = {}

    def _hicache_apply_place(self, record, stash):
        self.applied.append(("place", record.seq, stash is not None))
        return (record.kv_len, record.kv_crc, record.pool_counts)

    def _hicache_apply_write_complete(self, record, pop_local_ack):
        self.applied.append(("write_complete", record.seq, pop_local_ack))
        return (tuple(record.node_ids), record.ok)

    def _hicache_apply_load_complete(self, record, local_ack=None):
        self.applied.append(("load_complete", record.seq, local_ack is not None))
        return (tuple(record.node_ids), int(record.num_tokens))

    def _hicache_apply_evict(self, record):
        self.applied.append(("evict", record.seq))
        freed = self.evict_results.get(record.seq, record.freed)
        if freed != record.freed:
            raise RuntimeError("divergence")
        return (freed,)

    def _hicache_run_watermark_evictions(self, force=False):
        return []


def make_peer_authority():
    cache = FakeApplierCache()
    authority = HiCacheAuthority(
        cache=cache, controller=FakeController(), is_rank0=False, device=None
    )
    return authority, cache


def make_batch(records, base_crc=0):
    crc = base_crc
    cache = FakeApplierCache()
    for record in records:
        # Reproduce the observed tuples the fake applier returns.
        if record.kind == RECORD_PLACE:
            observed = (record.kv_len, record.kv_crc, record.pool_counts)
        elif record.kind == RECORD_EVICT:
            observed = (record.freed,)
        elif record.op == OP_LOAD:
            observed = (tuple(record.node_ids), int(record.num_tokens))
        else:
            observed = (tuple(record.node_ids), record.ok)
        crc = fold_crc(crc, record, observed)
    return HiCacheRecordBatch(records=list(records), crc_after=crc)


# ---- (1) record apply: ordering + idempotence --------------------------------


def test_apply_ordering_and_idempotence():
    authority, cache = make_peer_authority()
    records = [
        CacheRecord(kind=RECORD_PLACE, seq=1, op=OP_WRITE, node_id=10, kv_len=4,
                    kv_crc=indices_crc([0, 1, 2, 3])),
        CacheRecord(kind=RECORD_COMPLETE, seq=2, op=OP_WRITE, node_ids=(10,)),
    ]
    batch = make_batch(records)
    authority.apply_batch(batch)
    check(cache.applied == [("place", 1, False), ("write_complete", 2, True)],
          "records must apply in seq order with peer semantics")
    check(authority._applied_seq == 2, "applied seq must advance")

    # Idempotence: redelivery of the same batch is a no-op and does not
    # corrupt the CRC chain.
    authority.apply_batch(batch)
    check(len(cache.applied) == 2, "redelivered records must not re-apply")
    check(authority._applied_seq == 2, "seq unchanged on redelivery")

    # Ordering: a gap in the stream is fail-stop.
    gap = make_batch(
        [CacheRecord(kind=RECORD_COMPLETE, seq=9, op=OP_WRITE, node_ids=(1,))]
    )
    try:
        authority.apply_batch(gap)
        check(False, "seq gap must raise")
    except RuntimeError as e:
        check("gap" in str(e), "gap error must name the gap")


# ---- (2) CRC fail-stop --------------------------------------------------------


def test_crc_mismatch_raises():
    authority, cache = make_peer_authority()
    records = [
        CacheRecord(kind=RECORD_PLACE, seq=1, op=OP_WRITE, node_id=3, kv_len=2,
                    kv_crc=indices_crc([5, 6])),
    ]
    batch = make_batch(records)
    batch.crc_after ^= 0xDEADBEEF  # tamper
    try:
        authority.apply_batch(batch)
        check(False, "CRC mismatch must raise")
    except RuntimeError as e:
        check("CRC mismatch" in str(e), "CRC error message must be explicit")

    # Divergent local observation (peer evicts a different amount than
    # published) must also fail the batch even with an untampered CRC.
    authority2, cache2 = make_peer_authority()
    evict = CacheRecord(kind=RECORD_EVICT, seq=1, component=0, need=8, freed=8)
    batch2 = make_batch([evict])
    cache2.evict_results[1] = 5  # local replay frees a different amount
    try:
        authority2.apply_batch(batch2)
        check(False, "divergent eviction replay must raise")
    except RuntimeError:
        check(True, "")


# ---- (3) flush rule -----------------------------------------------------------


def test_flush_rule_bounds_staleness():
    authority, _ = make_peer_authority()
    # 15 skips allowed, the 16th step forces the broadcast.
    decisions = [authority.allow_skip() for _ in range(16)]
    check(all(decisions[:15]), "first 15 steps may skip")
    check(decisions[15] is False, "step 16 must force the broadcast (flush rule)")
    # After a broadcast the streak restarts.
    authority.note_broadcast()
    check(authority.allow_skip() is True, "streak must reset after a broadcast")


# ---- (4) rank-0 worker drains intents ----------------------------------------


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_worker_drains_intents_and_publishes():
    controller = FakeController()
    cache = FakeApplierCache()
    authority = HiCacheAuthority(
        cache=cache, controller=controller, is_rank0=True, device=None
    )
    try:
        authority.submit_write_intent(
            WriteIntent(node_id=42, op=OP_WRITE,
                        kv_device=torch.arange(4), extra_pools=None, kv_len=4)
        )
        check(wait_until(lambda: len(controller.committed) == 1),
              "worker must reserve+commit the intent")
        # PLACE first, then the COMPLETE from the (instantly fired) ack.
        check(wait_until(lambda: len(authority._outbox) >= 2),
              "worker must emit PLACE and COMPLETE records")
        batch = authority.build_publish_batch()
        check(batch is not None, "publish batch must carry the records")
        kinds = [r.kind for r in batch.records]
        check(kinds[0] == RECORD_PLACE and RECORD_COMPLETE in kinds,
              "PLACE must precede its COMPLETE in the stream")
        place = batch.records[0]
        check(place.node_id == 42 and place.kv_len == 4,
              "PLACE must name the node and placement size")
        check(place.kv_crc == indices_crc(controller.committed[0].host_indices),
              "PLACE must carry the exact host-index CRC")
        # Rank 0 self-applied at build (stash consumed, tree updated).
        check(("place", place.seq, True) in cache.applied,
              "rank 0 must self-apply the PLACE with its stashed reservation")
        check(place.seq not in authority._stash, "stash must be consumed")
        # A peer applying the same batch converges to the same CRC.
        peer, peer_cache = make_peer_authority()
        peer.apply_batch(batch)
        check(peer._crc == authority._crc,
              "peer CRC must converge with rank 0 after applying the batch")
    finally:
        authority.shutdown()


def test_worker_cancels_exhausted_intent():
    # Reservation fails MAX_INTENT_TRIES times with nothing evictable: the
    # intent must be cancelled with a COMPLETE ok=False (fail-safe), and the
    # cancellation must reach the published stream.
    controller = FakeController(fail_reservations=100)
    cache = FakeApplierCache()
    authority = HiCacheAuthority(
        cache=cache, controller=controller, is_rank0=True, device=None
    )
    try:
        authority.submit_write_intent(
            WriteIntent(node_id=7, op=OP_WRITE,
                        kv_device=torch.arange(2), extra_pools=None, kv_len=2)
        )
        # Drive the retry loop: each build re-submits stalled intents.
        for _ in range(20):
            authority.build_publish_batch()
            if any(
                r.kind == RECORD_COMPLETE and not r.ok
                for r in [a for a in []]
            ):
                break
            done = any(
                item[0] == "write_complete" for item in cache.applied
            )
            if done:
                break
            time.sleep(0.05)
        check(any(item[0] == "write_complete" for item in cache.applied),
              "exhausted intent must be cancelled via COMPLETE ok=False")
        check(7 in authority._failed_nodes,
              "failed node must be remembered to cancel dependent children")
    finally:
        authority.shutdown()


# ---- (5) mamba fail-stop (schedule_batch) -------------------------------------


def _method_node(path: Path, class_name: str, method_name: str):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return source, method


def _extract_method(path, class_name, method_name, namespace):
    source, node = _method_node(path, class_name, method_name)
    segment = textwrap.dedent(ast.get_source_segment(source, node))
    local_ns = dict(namespace)
    exec(
        compile("from __future__ import annotations\n" + segment, str(path), "exec"),
        local_ns,
    )
    return local_ns[method_name]


class MambaLifetimeError(RuntimeError):
    pass


class FakeMambaAllocator:
    def __init__(self, free_slots):
        self.free_slots = free_slots

    def alloc(self, n):
        if len(self.free_slots) < n:
            return None
        out = torch.tensor(self.free_slots[:n], dtype=torch.int64)
        del self.free_slots[:n]
        return out

    def available_size(self):
        return len(self.free_slots)


def make_alloc_batch(*, free_slots, evictable, evict_replenishes):
    batch = SimpleNamespace()
    batch.req_to_token_pool = SimpleNamespace(
        mamba_allocator=FakeMambaAllocator(list(free_slots))
    )
    batch.reqs = [SimpleNamespace(rid="r0")]
    evict_calls = []

    def evict(params):
        evict_calls.append(params)
        if evict_replenishes:
            batch.req_to_token_pool.mamba_allocator.free_slots.append(99)

    batch.tree_cache = SimpleNamespace(
        mamba_evictable_size=lambda: evictable, evict=evict
    )
    batch._evict_calls = evict_calls
    namespace = {
        "MambaLifetimeError": MambaLifetimeError,
        "EvictParams": lambda **kw: SimpleNamespace(**kw),
        "envs": SimpleNamespace(
            SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=SimpleNamespace(get=lambda: False)
        ),
        "torch": torch,
    }
    fn = _extract_method(
        SCHEDULE_BATCH_PATH, "ScheduleBatch", "_mamba_alloc_or_fail_stop", namespace
    )
    batch._mamba_alloc_or_fail_stop = types.MethodType(fn, batch)
    return batch


def test_mamba_fail_stop():
    # Plain success: no eviction, slot returned.
    batch = make_alloc_batch(free_slots=[3], evictable=0, evict_replenishes=False)
    slot = batch._mamba_alloc_or_fail_stop(1)
    check(int(slot[0]) == 3, "healthy pool must allocate directly")
    check(not batch._evict_calls, "no eviction when the pool has room")

    # Exhausted with evictable-idle: evict first, then retry succeeds.
    batch = make_alloc_batch(free_slots=[], evictable=5, evict_replenishes=True)
    slot = batch._mamba_alloc_or_fail_stop(1)
    check(int(slot[0]) == 99, "evictable-idle must be reclaimed before failing")
    check(len(batch._evict_calls) == 1 and batch._evict_calls[0].mamba_num == 1,
          "eviction must target the mamba component with the needed count")

    # Genuinely exhausted (no evictable): fail-stop raise, loud message.
    batch = make_alloc_batch(free_slots=[], evictable=0, evict_replenishes=False)
    try:
        batch._mamba_alloc_or_fail_stop(1)
        check(False, "exhausted-with-no-evictable must raise")
    except MambaLifetimeError as e:
        check("divergence" in str(e), "fail-stop message must name divergence")
    check(not batch._evict_calls, "no eviction attempt when nothing is evictable")

    # Evictable but eviction does not actually free a slot: still fail-stop.
    batch = make_alloc_batch(free_slots=[], evictable=5, evict_replenishes=False)
    try:
        batch._mamba_alloc_or_fail_stop(1)
        check(False, "failed eviction retry must raise")
    except MambaLifetimeError:
        check(True, "")


# ---- (6) peer paths never enter a collective ----------------------------------

COLLECTIVE_TOKENS = {
    "all_reduce",
    "all_gather",
    "all_gather_object",
    "broadcast",
    "broadcast_object",
    "broadcast_object_list",
    "barrier",
    "reduce",
    "isend",
    "irecv",
    "send",
    "recv",
    "broadcast_pyobj",
    "point_to_point_pyobj",
}


def _collective_uses(node):
    uses = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            name = None
            if isinstance(fn, ast.Attribute):
                name = fn.attr
            elif isinstance(fn, ast.Name):
                name = fn.id
            if name in COLLECTIVE_TOKENS:
                uses.append(name)
    return uses


def test_peer_paths_have_no_collectives():
    # Every method on the record-driven path (peer apply, triggers, per-step
    # checks, load path) must be collective-free — the grep-able invariant.
    peer_methods = [
        "_hicache_enqueue_write",
        "_hicache_apply_place",
        "_hicache_apply_write_complete",
        "_hicache_apply_load_complete",
        "_hicache_apply_evict",
        "_hicache_run_watermark_evictions",
        "_hicache_watermark_targets",
        "write_backup",
        "writing_check",
        "loading_check",
        "check_hicache_events",
        "flush_write_through_acks",
        "load_back",
        "_load_back_transfers",
    ]
    for name in peer_methods:
        _, node = _method_node(CACHE_PATH, "UnifiedRadixCache", name)
        uses = _collective_uses(node)
        check(not uses,
              f"UnifiedRadixCache.{name} must not call collectives, found {uses}")

    # The whole authority module is collective-free by construction.
    tree = ast.parse(AUTHORITY_PATH.read_text())
    uses = _collective_uses(tree)
    check(not uses, f"hicache_authority.py must be collective-free, found {uses}")

    # The V2 mamba consensus is gone from schedule_batch, and the lazy-alloc
    # paths are collective-free.
    sb_source = SCHEDULE_BATCH_PATH.read_text()
    check("_mamba_lazy_alloc_consensus" not in sb_source,
          "_mamba_lazy_alloc_consensus must be deleted")
    for name in (
        "_mamba_alloc_or_fail_stop",
        "mamba_lazy_prealloc_at_boundary",
        "mamba_lazy_spec_prepare",
    ):
        _, node = _method_node(SCHEDULE_BATCH_PATH, "ScheduleBatch", name)
        uses = _collective_uses(node)
        check(not uses, f"ScheduleBatch.{name} must be collective-free, got {uses}")

    # The DFLASH identity probe no longer owns a collective.
    dflash_source = (REPO / "python/sglang/srt/speculative/dflash_worker_v2.py").read_text()
    tree = ast.parse(dflash_source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Attribute)
                    and n.attr == "all_reduce"
                    and "_batch_identity_probe_tick" in ast.dump(node)
                ):
                    check(False, "DFLASH identity probe must not all_reduce")
    check("note_batch_identity" in dflash_source,
          "DFLASH probe must publish identity via the authority mailbox")

    # The publication carrier: request_receiver appends/extracts the record
    # batch around the EXISTING broadcast_pyobj (allowed), and peers apply
    # via bus.apply_batch only.
    receiver_source = RECEIVER_PATH.read_text()
    check("HiCacheRecordBatch" in receiver_source
          and "build_publish_batch" in receiver_source
          and "apply_batch" in receiver_source,
          "request_receiver must carry and apply record batches")


# ---- run ----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests, {PASSED} checks passed")
