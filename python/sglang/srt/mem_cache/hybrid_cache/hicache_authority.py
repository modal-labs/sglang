"""HiCache V3 control plane: rank-0-authoritative dedup-pool decisions.

Design doc: docs/HICACHE_V3.md. Invariant: steady-state decode executes ZERO
scheduler-thread collectives. Rank 0 owns every physical dedup-pool decision
(host KV placements, evictions) on a background worker thread and PUBLISHES
``CacheRecord``s to peers by piggybacking the existing per-step rank0->peers
request broadcast (bytes on an existing collective, never a new collective).
Peers apply records at batch prep; cross-rank agreement is enforced by
fail-stop asserts + a running bookkeeping CRC, never by negotiation.

Thread model (constraints, not narration):
- Worker thread exists on rank 0 only. It touches ONLY the cache controller
  and pool allocators (never the radix tree) and only under ``self._lock``.
- Every allocator-mutating operation on rank 0 (worker reserve/commit,
  scheduler-side watermark eviction) runs under ``self._lock`` and emits its
  record inside the same critical section: physical mutation order must equal
  record-sequence order, because peers replay allocator state in seq order.
- ``build_publish_batch`` (rank 0) and ``apply_batch`` (peers) run on the
  scheduler thread inside the request-broadcast window; that is the only
  place tree state mutates from records, so trees stay in lockstep at loop
  boundaries on every rank.
- Peers never enter a collective in any method of this module.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import weakref
import zlib
from collections import deque
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )

logger = logging.getLogger(__name__)

# Record kinds.
RECORD_PLACE = 1
RECORD_EVICT = 2
RECORD_COMPLETE = 3

# Op codes carried by PLACE/COMPLETE records. Values intentionally mirror the
# V2 fingerprint opcodes in unified_radix_cache.py so log forensics line up.
OP_WRITE = 1
OP_LOAD = 2
OP_WRITE_MAMBA_REPAIR = 7

# Publication bounds (docs/HICACHE_V3.md: flush rule N=16, payload bounded).
MAX_RECORDS_PER_STEP = 64
FLUSH_INTERVAL_STEPS = 16
# A write intent that cannot reserve host slots is retried after a
# rank-0-published eviction round at most this many times before it is
# cancelled fail-safe (COMPLETE ok=False on every rank).
MAX_INTENT_TRIES = 3


def indices_crc(indices) -> int:
    """CRC of an exact host-slot index list. Peers recompute over their
    mirror allocation; a mismatch is state divergence (fail-stop)."""
    if indices is None:
        return 0
    if isinstance(indices, torch.Tensor):
        indices = indices.flatten().tolist()
    return zlib.crc32(",".join(str(int(i)) for i in indices).encode())


@dataclasses.dataclass
class CacheRecord:
    """One published control-plane decision. Must stay cheaply picklable:
    ints/tuples only, no tensors (rank-local tensors ride the rank-0 stash,
    never the wire)."""

    kind: int
    seq: int
    op: int = 0
    node_id: int = -1
    kv_len: int = 0
    kv_crc: int = 0
    # ((pool_name:str, count:int), ...) for rank-local pools (mamba/draft):
    # peers allocate their OWN indices; only counts must agree.
    pool_counts: tuple = ()
    # COMPLETE: ack ids finished by this record.
    node_ids: tuple = ()
    # EVICT: deterministic replay inputs + rank-0 observed result.
    component: int = -1
    need: int = 0
    freed: int = 0
    ok: bool = True
    num_tokens: int = 0

    def fold_key(self) -> bytes:
        return repr(
            (
                self.kind,
                self.seq,
                self.op,
                self.node_id,
                self.kv_len,
                self.kv_crc,
                self.pool_counts,
                self.node_ids,
                self.component,
                self.need,
                self.ok,
            )
        ).encode()


@dataclasses.dataclass
class HiCacheRecordBatch:
    """Wire object appended to the per-step request broadcast payload."""

    records: list
    crc_after: int
    # (probe_tick, bs, rid_crc) sampled batch identity, or None.
    identity: Optional[tuple] = None


@dataclasses.dataclass
class WriteIntent:
    """Rank-0 write-through intent. Carries scheduler-thread snapshots so the
    worker never reads tree state."""

    node_id: int
    op: int
    kv_device: torch.Tensor
    extra_pools: Optional[list]
    kv_len: int
    # Ancestor node ids whose backups were still pending at enqueue: if any
    # of them failed on the worker, this intent must fail too (the tree
    # invariant "parent backuped before child" would otherwise break).
    parent_ids: tuple = ()
    # Forward-done event captured at enqueue: the D2H reads device pool state
    # and must order behind the in-flight forward's writes (fence_state_read
    # semantics, hoisted to the worker thread).
    fence_event: Any = None
    tries: int = 0


def fold_crc(crc: int, record: CacheRecord, observed: tuple) -> int:
    """Chain the running bookkeeping CRC over a record and the applying
    rank's OBSERVED outcome. Rank 0 folds at emission/self-apply, peers fold
    at apply; equality of the chained value is the cross-rank agreement
    check (fail-stop on mismatch, no negotiation)."""
    crc = zlib.crc32(record.fold_key(), crc)
    return zlib.crc32(repr(observed).encode(), crc)


_ACTIVE_AUTHORITY: Optional["weakref.ref"] = None


def note_batch_identity(tick: int, bs: int, rid_crc: int) -> None:
    """Mailbox for the sampled DFLASH batch-identity probe: replaces the
    probe's own gloo collective; identity now rides the published record
    batches and is compared rank-locally on receipt."""
    global _ACTIVE_AUTHORITY
    if _ACTIVE_AUTHORITY is None:
        return
    authority = _ACTIVE_AUTHORITY()
    if authority is not None:
        authority.record_batch_identity(tick, bs, rid_crc)


class HiCacheAuthority:
    """Rank-0: decision worker + publisher. Non-zero ranks: thin applier."""

    def __init__(
        self,
        *,
        cache: Any,
        controller: "HybridCacheController",
        is_rank0: bool,
        device: Any = None,
        max_records_per_step: int = MAX_RECORDS_PER_STEP,
        flush_interval: int = FLUSH_INTERVAL_STEPS,
    ):
        self.cache = cache
        self.controller = controller
        self.is_rank0 = is_rank0
        self.device = device
        self.max_records_per_step = max_records_per_step
        self.flush_interval = flush_interval

        self._lock = threading.RLock()
        self._seq = 0  # last emitted seq (rank 0)
        self._applied_seq = 0  # last applied seq (every rank)
        self._crc = 0  # running bookkeeping CRC (every rank)
        self._outbox: deque = deque()  # worker/scheduler emitted records
        self._stash: dict = {}  # rank 0: seq -> HybridWriteReservation
        self._stalled: deque = deque()  # intents awaiting an eviction round
        self._failed_nodes: set = set()
        self._skip_streak = 0
        # seqs whose load bookkeeping already ran on rank 0 (with the ack).
        self._load_applied_seqs: set = set()
        # Sampled batch-identity probe mailbox: tick -> (bs, rid_crc).
        self._identity_notes: deque = deque(maxlen=8)
        self._published_identity_tick = -1

        self._intents: Optional[Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        if self.is_rank0:
            self._intents = Queue()
            self._worker = threading.Thread(
                target=self._worker_loop, name="hicache-authority", daemon=True
            )
            self._worker.start()

        global _ACTIVE_AUTHORITY
        _ACTIVE_AUTHORITY = weakref.ref(self)

    # ---- flush rule -------------------------------------------------------

    def allow_skip(self) -> bool:
        """Bound record staleness when the scheduler recv-skipper is active:
        the request broadcast (the publication carrier) must run at least
        every ``flush_interval`` loop iterations. The decision is a pure
        function of a local counter that advances identically on every rank
        (the skipper's own decision is rank-symmetric), so no rank ever
        waits in a broadcast the others skipped."""
        self._skip_streak += 1
        if self._skip_streak >= self.flush_interval:
            self._skip_streak = 0
            return False
        return True

    def note_broadcast(self) -> None:
        self._skip_streak = 0

    # ---- identity probe ----------------------------------------------------

    def record_batch_identity(self, tick: int, bs: int, rid_crc: int) -> None:
        with self._lock:
            self._identity_notes.append((int(tick), int(bs), int(rid_crc)))

    def _pop_identity_for_publish(self) -> Optional[tuple]:
        for tick, bs, rid_crc in reversed(self._identity_notes):
            if tick > self._published_identity_tick:
                self._published_identity_tick = tick
                return (tick, bs, rid_crc)
        return None

    def _check_identity(self, identity: Optional[tuple]) -> None:
        if identity is None:
            return
        tick, bs, rid_crc = identity
        for local_tick, local_bs, local_crc in self._identity_notes:
            if local_tick != tick:
                continue
            if local_bs != bs or local_crc != rid_crc:
                # Same failure semantics as the deleted DFLASH gloo probe:
                # alarm loudly; rank-local outcomes stay self-consistent.
                logger.error(
                    "HiCache V3 batch identity diverged across TP ranks "
                    "(tick=%d rank0=(bs=%d,crc=%d) local=(bs=%d,crc=%d)).",
                    tick,
                    bs,
                    rid_crc,
                    local_bs,
                    local_crc,
                )
            return

    # ---- rank-0 intent API (scheduler thread) ------------------------------

    def submit_write_intent(self, intent: WriteIntent) -> None:
        assert self.is_rank0 and self._intents is not None
        self._intents.put(intent)

    # ---- worker (rank 0 only) ----------------------------------------------

    def _worker_loop(self) -> None:
        if self.device is not None and getattr(self.device, "type", "") == "cuda":
            torch.cuda.set_device(self.device)
        while not self._stop.is_set():
            try:
                intent = self._intents.get(timeout=0.02)
            except Empty:
                intent = None
            try:
                if intent is not None:
                    self._process_intent(intent)
                self._poll_write_acks()
            except Exception:
                # A worker death silently stalls every pending write on all
                # ranks (locks never release); crash loud instead.
                logger.exception("HiCache authority worker failed; fail-stop.")
                raise

    def _emit(self, record: CacheRecord) -> None:
        # Callers hold self._lock: seq order == emission order == physical
        # allocator mutation order, which peers replay.
        self._outbox.append(record)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _cancel_intent(self, intent: WriteIntent) -> None:
        self._failed_nodes.add(intent.node_id)
        self._emit(
            CacheRecord(
                kind=RECORD_COMPLETE,
                seq=self._next_seq(),
                op=intent.op,
                node_id=intent.node_id,
                node_ids=(intent.node_id,),
                ok=False,
            )
        )

    def _process_intent(self, intent: WriteIntent) -> None:
        with self._lock:
            if any(p in self._failed_nodes for p in intent.parent_ids):
                self._cancel_intent(intent)
                return
            reservation = self.controller.reserve_write(
                intent.kv_device,
                node_id=intent.node_id,
                extra_pools=intent.extra_pools or None,
                allow_evict=False,
            )
            if reservation is None:
                # Host pools need an eviction round; eviction walks the tree,
                # so it must run on the scheduler thread (build hook). Park
                # the intent; the hook re-submits it after publishing EVICTs.
                intent.tries += 1
                if intent.tries >= MAX_INTENT_TRIES:
                    logger.error(
                        "HiCache V3: write intent for node %d exhausted %d "
                        "reservation attempts; cancelling on every rank.",
                        intent.node_id,
                        intent.tries,
                    )
                    self._cancel_intent(intent)
                else:
                    self._stalled.append(intent)
                return
            if intent.fence_event is not None:
                torch.get_device_module().current_stream().wait_event(
                    intent.fence_event
                )
            self.controller.commit_write(reservation)
            pool_counts = tuple(
                (str(t.name), int(len(t.host_indices)))
                for t in (intent.extra_pools or ())
                if t.host_indices is not None
            )
            seq = self._next_seq()
            self._stash[seq] = reservation
            self._emit(
                CacheRecord(
                    kind=RECORD_PLACE,
                    seq=seq,
                    op=intent.op,
                    node_id=intent.node_id,
                    kv_len=int(len(reservation.host_indices)),
                    kv_crc=indices_crc(reservation.host_indices),
                    pool_counts=pool_counts,
                )
            )

    def _poll_write_acks(self) -> None:
        with self._lock:
            queue = self.controller.ack_write_queue
            while queue and queue[0].finish_event.query():
                ack = queue.pop(0)
                self._emit(
                    CacheRecord(
                        kind=RECORD_COMPLETE,
                        seq=self._next_seq(),
                        op=OP_WRITE,
                        node_ids=tuple(ack.node_ids),
                    )
                )

    # ---- rank-0 publish hook (scheduler thread, pre-broadcast) --------------

    def build_publish_batch(self) -> Optional[HiCacheRecordBatch]:
        """Drain emitted records (bounded), add load COMPLETEs and watermark
        EVICTs, self-apply everything in seq order, and return the wire
        batch. Holding the lock across the whole build keeps worker emissions
        from interleaving out of seq order with build-side emissions."""
        assert self.is_rank0
        self.note_broadcast()
        with self._lock:
            records: list = []
            while self._outbox and len(records) < self.max_records_per_step:
                records.append(self._outbox.popleft())
            backlog = bool(self._outbox)
            if not backlog:
                # New emissions ride this batch only when no lower-seq
                # records were deferred (seq order on the wire is strict).
                records.extend(self._poll_load_acks())
                records.extend(self._run_evictions_and_retries())
            for record in records:
                observed = self._apply_record(record, rank0=True)
                self._crc = fold_crc(self._crc, record, observed)
            if not records:
                return None
            return HiCacheRecordBatch(
                records=records,
                crc_after=self._crc,
                identity=self._pop_identity_for_publish(),
            )

    def _poll_load_acks(self) -> list:
        records = []
        queue = self.controller.ack_load_queue
        while queue and queue[0].finish_event.query():
            ack = queue.pop(0)
            record = CacheRecord(
                kind=RECORD_COMPLETE,
                seq=self._next_seq(),
                op=OP_LOAD,
                node_ids=tuple(ack.node_ids),
                num_tokens=int(getattr(ack, "num_tokens", 0)),
            )
            # Local bookkeeping needs the ack's events (metrics); do it here
            # with the concrete ack, then skip tree work at self-apply.
            self.cache._hicache_apply_load_complete(record, local_ack=ack)
            self._load_applied_seqs.add(record.seq)
            records.append(record)
        return records

    def _run_evictions_and_retries(self) -> list:
        records = []
        force = bool(self._stalled)
        for component, need, freed in self.cache._hicache_run_watermark_evictions(
            force=force
        ):
            records.append(
                CacheRecord(
                    kind=RECORD_EVICT,
                    seq=self._next_seq(),
                    component=int(component),
                    need=int(need),
                    freed=int(freed),
                )
            )
        while self._stalled:
            self._intents.put(self._stalled.popleft())
        return records

    # ---- apply (every rank; rank 0 self-applies at build) -------------------

    def apply_batch(self, batch: HiCacheRecordBatch) -> None:
        self.note_broadcast()
        with self._lock:
            for record in batch.records:
                if record.seq <= self._applied_seq:
                    continue  # idempotent under redelivery
                if record.seq != self._applied_seq + 1:
                    raise RuntimeError(
                        "HiCache V3 record gap: expected seq "
                        f"{self._applied_seq + 1}, got {record.seq}; "
                        "control-plane stream corrupt (fail-stop)."
                    )
                observed = self._apply_record(record, rank0=False)
                self._crc = fold_crc(self._crc, record, observed)
                self._applied_seq = record.seq
            if self._crc != batch.crc_after:
                raise RuntimeError(
                    "HiCache V3 bookkeeping CRC mismatch after applying seq "
                    f"{self._applied_seq}: local={self._crc} "
                    f"rank0={batch.crc_after}. Mirrored cache state diverged "
                    "(fail-stop)."
                )
            self._check_identity(batch.identity)

    def _apply_record(self, record: CacheRecord, *, rank0: bool) -> tuple:
        if rank0:
            self._applied_seq = record.seq
        if record.kind == RECORD_PLACE:
            stash = self._stash.pop(record.seq, None) if rank0 else None
            return self.cache._hicache_apply_place(record, stash)
        if record.kind == RECORD_EVICT:
            if rank0:
                # Rank 0 evicted at decision time (under this lock, in seq
                # order); fold its observed result.
                return (record.freed,)
            return self.cache._hicache_apply_evict(record)
        if record.kind == RECORD_COMPLETE:
            if record.op == OP_LOAD:
                if rank0:
                    # Bookkeeping already ran in _poll_load_acks with the ack;
                    # observed shape must match the peer fold exactly.
                    self._load_applied_seqs.discard(record.seq)
                    return (tuple(record.node_ids), int(record.num_tokens))
                return self.cache._hicache_apply_load_complete(record, local_ack=None)
            return self.cache._hicache_apply_write_complete(
                record, pop_local_ack=not rank0
            )
        raise RuntimeError(f"HiCache V3 unknown record kind {record.kind}")

    # ---- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            if self._intents is not None:
                while True:
                    try:
                        self._intents.get_nowait()
                    except Empty:
                        break
            self._outbox.clear()
            self._stash.clear()
            self._stalled.clear()
            self._failed_nodes.clear()
            self._identity_notes.clear()
            # seq/crc continue monotonically: reset is a symmetric control
            # request applied on every rank, but in-flight worker output may
            # straddle it; keeping the stream contiguous avoids false gaps.

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
