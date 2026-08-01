# HiCache V3 — rank-0-authoritative control plane

Date: 2026-08-01. Owners: James / Claude session. Status: phase 1 in build.

## The invariant

Steady-state decode executes ZERO scheduler-thread collectives. Every
cross-rank agreement is (a) structural (cannot diverge), (b) shipped with
the existing per-step batch broadcast (bytes, not ops), or (c) asynchronous
on a dedicated worker + process group. Any perf cost that remains must be
inherent to hierarchical caching (data-plane bandwidth), never control.

## Why (measured, 2026-08-01)

- gloo all_reduce, 8 ranks, IDLE gVisor container: 6.9 ms (5xint64),
  21 ms (514xint64). Native Linux equivalent: 1.6/4.0 ms on a starved
  4-core box; ~0.3-0.8 ms on real hosts. gVisor netstack multiplies small
  CPU collectives 10-50x.
- K3 decode step budget: ~11 ms (bs1, block 8). Overlap hides CPU work
  only below that. V2 spent 60-110 ms/step on collectives -> the prod
  regression (rolled back 2026-08-01 ~09:50Z).
- V1 paid 2 polls/step (~14 ms on gVisor) and was already quietly
  CPU-bound; nobody had step-time telemetry. `step-ms` is now in the
  decode log line and is a permanent battery gate.

## Design

### Storage (unchanged from V2)
- MLA KV: device-replicated; host DEDUPLICATED (rank-0-owned physical
  pool, dummy allocator mirrors elsewhere). Loadback: rank-0 H2D +
  layerwise NCCL broadcast (device-side).
- Mamba/KDA + draft KV: head-sharded => per-rank host pools, rank-local
  DMA.

### Control plane (new)
1. FAIL-STOP replaces alloc consensus. Mirrored allocators are
   deterministic given the rank-identical request order (arrival_stamp).
   A rank whose alloc outcome differs from what determinism requires
   raises immediately (pod restart, loud) instead of negotiating a
   demote. Evidence basis: post-ordering-fix divergence rate measured 0
   by the V2 consensus instrument; pool sizing gives 2x headroom over
   worst-case boundary demand (analysis below). Same-step device allocs
   MUST be local: publication cannot make this-step decisions.
2. RANK-0 PUBLICATION replaces write/evict/ack consensus. Rank 0 owns
   the physical dedup pool, so it decides placements/evictions alone on
   a background worker and PUBLISHES records (node_id -> host slots,
   evictions, completions, bookkeeping CRC) by appending them to the
   existing per-step rank0->peers request broadcast. Peers apply records
   at batch prep (dict ops). Staleness degrades to a conservative cache
   miss, never disagreement. Flush rule: records ride the next broadcast,
   forced at most N=16 steps after creation.
3. PEERS NEVER POLL. Rank-0's worker observes its own DMA completions
   locally and publishes them; peers finish ack bookkeeping (lock-ref
   decs, backup flags, R2 complete_op) on receipt. The per-step
   loading/writing_check collectives are DELETED on peers; rank 0 polls
   its own CUDA events locally (no collective).
4. DETECTION WITHOUT NEGOTIATION. Each record batch carries rank-0's
   bookkeeping CRC; peers compare against their mirror after apply and
   raise on mismatch (fail-stop). The DFLASH batch-identity probe stays,
   sampled, riding published CRCs rather than its own collective.
5. Partial hits (KV hosted, mamba evicted): match clamps, request
   re-prefills (fail-safe, unchanged), and the mamba-only repair op is
   queued to the worker (free) so the miss is transient.
6. R2 slot-lifetime FSM stays: RESERVED/PENDING_FREE across steps is
   what makes worker-async commits safe.

### What is DELETED from V2
- _mamba_lazy_alloc_consensus (both sites) -> fail-stop assert.
- reserve_write consensus / _all_ranks_succeeded chains -> worker.
- watermark trigger MIN-reduce -> rank-0 unilateral eviction, published.
- peer-side loading_check/writing_check collectives -> publication.
- identity probe collective -> published CRC comparison.
- R1 batch load-back consensus branch -> superseded by placements.

### Hot-path cost accounting (target)
| path | V2 | V3 |
|---|---|---|
| decode step steady state | ~8 gloo (60-110 ms) | 0 collectives; ~us of record apply |
| conversation finish | 30-50 serialized gloo | 1 intent-queue append |
| loadback | consensus + NCCL bcast | placement lookup + same NCCL bcast |
| alloc failure | negotiated demote | raise (never-event by sizing) |

### Known residual risks (watched by gates, all second-order)
1. Worker GIL contention (Python thread): worker CPU work is us-scale;
   if step-ms shows stalls, escalate worker to a subprocess.
2. Loadback NCCL/PCIe contention during bursts: inherent, bounded,
   amortized; shows as p99 proportional to loadback traffic.
3. Slow-host classes (fault <1 GiB/s) multiply the remaining python
   section 3-5x; validated explicitly on an austin/AWS lane.
4. Publication payload spikes: bounded by flush rule + record size.

## Validation ladder
1. CPU rig (tests/… + repo-root harness): 8-proc gloo, measured gVisor
   constants; gates: collectives/step budget, symmetry under randomized
   schedules, zero device reads in prepare paths.
2. Dummy-weights MIN1 lane: step-ms p50 within noise of no-hicache
   baseline (13.3 ms measured 2026-08-01), p99 bounded.
3. Real-weights MIN1 lane: two-phase clean, bench pass, multi-hour hang
   soak, step-ms gates.
4. Austin/AWS lane: host-variance floor.
5. Prod: rolling, step-ms on dashboard, unpinned region.

## Pool-sizing analysis (fail-stop justification)
Device mamba: 130 slots; worst-case simultaneous demand = max_running
(16) x 2 ping-pong + in-flight loadback reservations (bounded by
mamba_max_states_per_path x concurrent loadbacks <= 16) + cached idle
(evictable on demand, device evict is local+mirrored). Alloc failure
requires demand > 130 with zero evictable — structurally impossible
while cached-idle eviction precedes hard failure; the fail-stop assert
therefore only fires on true state divergence, which is the event we
WANT loud.
