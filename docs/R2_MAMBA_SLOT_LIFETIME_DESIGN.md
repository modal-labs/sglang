# R2: Device Mamba-Slot Lifetime Management — Design Specification

**Status:** DRAFT for review — implementation-ready except items marked OPEN
**Grounding:** branch `codex/k3-rc5-situ-hicache-combined`, HEAD `58937d8ec`
("perf(hicache): targeted forward-write fences + graph-compatible load fence").
All file:line references below are against this checkout.
**Deployment context:** TP8 Kimi-K3 (KDA/GDN linear-attention layers), HiCache
host offload + MLA host dedup + DFLASH spec + overlap scheduler. The active
device pool is sized at 130 slots in the production deployment
(`--max-mamba-cache-size`); the design is size-independent.

---

## 0. Problem statement

A device mamba/KDA slot is **mutable, aliased, per-request recurrent state**:

- *Mutable*: every decode step rewrites the slot in place (`temporal`, `conv`,
  and — under ReplaySSM — per-slot ring cursors `replayssm_write_pos/cache_base/
  is_flush`, `memory_pool.py:819`, `932-940`).
- *Aliased*: the same physical slot id is named concurrently by a `Req`
  (`req.mamba_pool_idx`, `schedule_batch.py:842`), a radix-tree node
  (`cd.value`, `mamba_component.py:221`), queued DMA descriptors
  (`PoolTransfer.device_indices`, `hicache_storage.py:93-107`), deferred-CoW
  staging (`req.mamba_cow_src_index`, `schedule_batch.py:852`), ack queues
  (`HiCacheAck`, `managers/cache_controller.py:167-172`), and session slots
  (`session/streaming_session.py:60-119`).
- *Managed by machinery designed for immutable KV pages*: the radix tree, the
  HiCache controller, and the allocator all assume "once written, a page's
  content is stable and a free is always safe."

The mismatch produced seven production corruption/UAF classes, each fixed this
week by a **scattered point guard** (§0.1). R2 replaces the guards with a
single slot-lifetime state machine whose transition rules make each bug
inexpressible, with fence tokens carried on transitions so cross-stream
visibility is part of the ownership contract rather than an ad-hoc `getattr`
call.

### 0.1 Inventory of current guards (all become transition rules in §1)

| # | Bug (verified in prod) | Current point guard | Code |
|---|---|---|---|
| G1 | Slot freed while a committed load-back DMA still targeted it → reallocated and clobbered mid-flight | Allocator pin/unpin + deferred free (park freed-but-pinned slots) | `allocator/mamba.py:88-134`; pinned at `unified_radix_cache.py:2212-2231`; unpinned at `3028-3031` |
| G2 | CoW from a node whose H2D load hadn't landed (slot still holds previous tenant) | `ongoing_load_back` probe + drop the prefix match while the op is commit→ack | `mamba_component.py:127-155` (`finalize_match_result`), op record `unified_radix_cache.py:297-306` |
| G3 | Failed load-back leaving a deep KV prefix paired with a zeroed mamba slot | `init_load_back` returns `None` → prefix drop + CoW-staging cleanup | `unified_radix_cache.py:3094-3099`; `schedule_policy.py:1229-1242` |
| G4 | Ring cursors inherited from the slot's previous tenant on load-back | `reset_replayssm_cursors` at prepare + commit | `mamba_component.py:617-626`, `743-748`; fresh-alloc reset `memory_pool.py:1304-1308` |
| G5 | Torn checkpoints: schedule/write-stream reads of state the in-flight overlapped forward still writes (donate `copy_from`, `write_pos` RMW, write-through D2H) | `note_forward_launch` → `fence_state_read` event waits at each reader | producer `scheduler.py:3599-3610`; fence `unified_radix_cache.py:1764-1782`; consumers `unified_radix_cache.py:1850-1853`, `mamba_component.py:479-486`, `531-545` |
| G6 | Path-cap pruning / eviction of nodes with in-flight DMA | skip nodes in `ongoing_write_through`/`ongoing_load_back` | `mamba_component.py:268-287`; host watermark reasoning `unified_radix_cache.py:1784-1796` |
| G7 | CUDA-graph replay skips Python-level per-layer `wait_until` load gates | one wait on the load op's **final** event on the forward stream, pre-dispatch | `model_runner.py:1590-1601`; graph-eligibility comment `runner/prefill_cuda_graph_runner.py:725-730` (supersedes the eager gate of `a3fa931a0`) |

### 0.2 Known-latent holes (R2 must close by construction)

| Hole | Evidence |
|---|---|
| int8 checkpoint pool bypasses deferred-free/pin protection: `MambaCheckpointPool` embeds its **own** `MambaSlotAllocator` (`mamba_checkpoint_pool.py:225`, `232-233`) and `_free_mamba_value` frees into it (`mamba_component.py:452-456`), but load-back pins are taken only on `req_to_token_pool.mamba_allocator` (`unified_radix_cache.py:2224`) | a ckpt-pool `cd.value` freed while any future DMA names it has no pin |
| Virtual/physical id translation is inconsistent: cursor reset in `prepare_load_back` translates (`mamba_component.py:624-625`), the LOAD_BACK-commit reset does **not** (`mamba_component.py:745-748`), the `write_pos` RMW does **not** (`mamba_component.py:485`), the fresh-alloc reset does **not** (`memory_pool.py:1308`) | benign today only because ReplaySSM is gated off under the unified pool (`unified_memory_pool.py:946-949`) — a footgun, not an invariant |
| Pin bookkeeping does D2H syncs: `pin_slots`/`unpin_slots`/`free` call `.tolist()` on CUDA tensors (`allocator/mamba.py:93, 99, 124`) | scheduler-thread stalls proportional to pin traffic |
| `UnifiedMambaSlotAllocator` has **no** `pin_slots`/`unpin_slots` at all (`unified_memory_pool.py:794-887`) — HiCache + unified pool would `AttributeError` at `unified_radix_cache.py:2224` | the unified-memory HiCache wiring is inert (§6) |
| Rejected/aborted-request cleanup is scattered across five sites that each re-derive "who owns the slot now": `scheduler.py:3235-3252` (admission reject), `mem_cache/common.py:132-176` (`release_kv_cache`), `memory_pool.py:1441-1492` (`free_mamba_cache`), `mamba_component.py:549-591` (`cleanup_after_caching_req`) + `629-635` (`finalize_load_back`), `schedule_batch.py:1745-1772` (`release_req`/retract) | each site must independently remember pins, sessions, ping-pong buffers, staging |

---

## 1. State model

### 1.1 Shape of the state: ownership × op-set, not a flat enum

The prompt's proposed enum — FREE, ACTIVE, DONATED, LOADING, BACKING_UP,
PENDING_FREE — cannot be flat, because today's code *already* realizes
combinations of them simultaneously:

- **DONATED + BACKING_UP**: write-through reads `cd.value` (a donated slot) on
  the write stream while the node stays live (`unified_radix_cache.py:1866-
  1876`, `mamba_component.py:649-658`).
- **ACTIVE + LOADING**: the per-request CoW H2D targets `req.mamba_pool_idx`,
  which the request already owns (`mamba_component.py:611-627`, `679-687`).
- **DONATED + LOADING**: the tree-restore H2D targets the slot that becomes
  `cd.value` at commit (`mamba_component.py:667-675`, `749-759`).
- **PENDING_FREE + LOADING**: the exact G1 window — freed by admission-reject
  or request-finish while the committed DMA is queued
  (`unified_radix_cache.py:2212-2216`).

So R2 factors the state as:

```
SlotRecord = (owner_state, owner_ref, ops, tokens)

owner_state ∈ { FREE, RESERVED, ACTIVE, DONATED, PENDING_FREE }
owner_ref   : req handle | node id | reservation id | None
ops         : dict[op_id -> OpKind]        # refcounted overlay, any owner_state
OpKind      ∈ { H2D_WRITE,   # load-back writes INTO the slot
                D2H_READ,    # write-through reads FROM the slot
                DEV_READ }   # deferred CoW / donate-copy reads FROM the slot
tokens      : (write_token, ring_state)    # §4
```

`RESERVED` is required by the existing two-phase HiCache protocol: between
`reserve_load` (device indices allocated, `hybrid_cache_controller.py:590-638`)
and `commit_load`/`abort_load` (`640-657`), a slot is allocated but has no req
or node owner; `abort_*` rolls it straight back to FREE
(`hybrid_cache_controller.py:652-657`, `1062+`). Today this window is invisible
to everything but the reservation object.

`ring_state ∈ {DIRTY, CLEAN}` tracks whether the slot's ReplaySSM cursors are
guaranteed zero. This absorbs G4: every acquisition path either inherits a
CLEAN slot or must reset (§1.3, T1/T6).

### 1.2 Transition graph

```
                       acquire(req)                    donate(node, tok)
              ┌────────────────────────────┐   ┌──────────────────────────────┐
              │                            v   │                              v
   ┌──────┐   │    reserve(op)   ┌─────────────────┐    commit(node)   ┌──────────┐
   │ FREE │───┤                  │     ACTIVE      │                   │ DONATED  │
   └──────┘   └────> RESERVED ───│  owner = req    │                   │owner=node│
      ^  ^            │   │      └─────────────────┘                   └──────────┘
      │  │      abort │   │ commit(node|req)  │                          │      │
      │  └────────────┘   └──> DONATED|ACTIVE │ release(req)     evict / │      │ attach_op /
      │                        (+H2D op)      v                  release │      │ complete_op
      │                              ┌──────────────┐                    │      │ (self-loops)
      │        last complete_op      │ PENDING_FREE │<───────────────────┘      │
      └──────────────────────────────│ (ops ≠ ∅ at  │                           │
              (ops == ∅)             │  release)    │<──────(release with ops)──┘
                                     └──────────────┘
   attach_op / complete_op are self-loops legal in RESERVED / ACTIVE / DONATED /
   PENDING_FREE; illegal in FREE.
```

**Transition table.** "Trigger" = the only code path allowed to drive it;
"Token" = the fence/event the transition must carry (§4).

| # | Transition | Who may trigger | Token carried | Replaces guard |
|---|---|---|---|---|
| T1 | FREE → ACTIVE(req) | Scheduler thread: `HybridReqToTokenPool.alloc` (`memory_pool.py:1287-1328`), match-time CoW dst alloc (`mamba_component.py:156-173`), `prepare_load_back` (`mamba_component.py:595-627`), ping-pong alloc (`memory_pool.py:1378-1402`), incl. the `alloc_group` batch window (`allocator/mamba.py:58-79`, `scheduler.py:3182-3255`) | none inbound; slot content undefined → acquisition MUST set `ring_state=CLEAN` (cursor reset) and schedule content init (clear, CoW, or H2D op). Enforces G4 at the type level: there is no way to get an ACTIVE slot without declaring how its content becomes defined | G4 |
| T2 | FREE → RESERVED(op) | `reserve_load`/`reserve_write` pool-transfer reservation (`hybrid_cache_controller.py:1083+` via `590-638`) | none | (new; today invisible) |
| T3 | RESERVED → FREE | `abort_load`/`abort_write` (`hybrid_cache_controller.py:484-488`, `652-657`) — group-reject path `unified_radix_cache.py:2180-2183` | none | G3's slot half |
| T4 | RESERVED → DONATED(node) + attach H2D op | `commit_load` → `commit_hicache_transfer(LOAD_BACK)` (`unified_radix_cache.py:2186-2231`, `mamba_component.py:749-759`) — **post-consensus only** (`unified_radix_cache.py:2167-2186`) | op token = the load op's ack `finish_event` (`hybrid_cache_controller.py:718-733`); consumers of `cd.value` must wait it | G1 (pin), G2, G7 |
| T5 | ACTIVE → ACTIVE + attach H2D op | per-request CoW load (`mamba_component.py:679-687`) at the same commit point | same op token as T4 | G1 (the "request rejected after commit" case, `unified_radix_cache.py:2212-2216`) |
| T6 | ACTIVE(req) → DONATED(node) | `prepare_for_caching_req` donate/finish insert (`mamba_component.py:458-547`) + `commit_insert_component_data` (`212-243`); ping-pong variant `donate_mamba_ping_pong_slot` (`memory_pool.py:1416-1439`) | write token = latest `forward_done` (§4). The donate **copy** (no_buffer path, `mamba_component.py:536-545`) and the `write_pos` RMW (`479-486`) become *effects gated on the token*, not open-coded fences. For the finished no_buffer path (`cache_finished_req`), the slot itself changes owner and the ring must be flushed — captured by `ring_state` | G5 |
| T7 | DONATED → DONATED + attach D2H op | `write_backup` transfer build/commit (`unified_radix_cache.py:1830-1951`, `mamba_component.py:649-658`) — post-consensus (`1908-1931`) | op requires the slot's write token before the D2H may start (today: `fence_state_read()` at `unified_radix_cache.py:1853`); op token = write ack `finish_event` | G5 (write-through leg), G6 |
| T8 | DONATED → DONATED + attach DEV_READ op | CoW staging at match (`mamba_component.py:156-173` sets `req.mamba_cow_src_index`; consumed `schedule_batch.py:2547-2566`; executed `model_runner.py:1495-1541`) | op token = the *consuming batch's* `forward_done`; the source slot may not leave DONATED until it fires | replaces the CoW-source admission lock (`schedule_policy.py:1173-1187`) and the G2 match-drop **query** (§3) |
| T9 | ACTIVE → PENDING_FREE or FREE | the **single** release choke point (§3), called from: request finish (`free_mamba_cache`, `memory_pool.py:1441-1492`), admission reject (`scheduler.py:3235-3252`), retract (`schedule_batch.py:1745-1772` → `common.py:132-176`), called-off load-back (`finalize_load_back`, `mamba_component.py:629-635`), session teardown (`streaming_session.py:305-315`) | goes to PENDING_FREE iff `ops ≠ ∅`; else FREE | G1's deferred-free (`allocator/mamba.py:120-134`) |
| T10 | DONATED → PENDING_FREE or FREE | eviction: `evict_component` (`mamba_component.py:300-315`), path-cap prune (`245-288`), host-leaf cascade. **Rule: illegal while `ops ≠ ∅` → the eviction driver must query, not skip-by-node-dict** | G6 |
| T11 | PENDING_FREE → FREE | last `complete_op` (today: `unpin_slots` flush, `allocator/mamba.py:97-118`, driven by `loading_check` `unified_radix_cache.py:3020-3031`) | none (op tokens already synchronized by the ack protocol, `3018`) | G1 |
| T12 | attach/complete op (self-loop) | attach: only at DMA **commit** (never at reserve/build — matches "queueing after consensus", `unified_radix_cache.py:2186`) or CoW staging. complete: only from the rank-symmetric ack drain (`writing_check`/`loading_check`, `2941-3042`) or, for DEV_READ, the batch's forward completion | per-op token | G1, G2, G6 |

**Enforced-by-construction restatement of each guard:**

- **G1**: `free()` no longer exists on the allocator path; T9/T10 route through
  the lifetime table, which *cannot* emit FREE while `ops ≠ ∅`. The pin/park
  dance is the PENDING_FREE state.
- **G2**: `finalize_match_result` asks `lifetime.cow_source_ok(slot)` ≡
  (state == DONATED ∧ no H2D op in `ops`). Today's probe reconstructs exactly
  this from `ongoing_load_back` + `pinned_mamba_slots is not None`
  (`mamba_component.py:136-142`) — fragile because it keys on *node id* and
  the op record's optional field. In R2 it is a slot-level O(1) lookup, and it
  also automatically covers the case the node-keyed probe misses: an op that
  landed on the slot via a *different* node (aliasing after a split — see
  `_replace_pending_write_through_node`, `unified_radix_cache.py:1963-1993`).
- **G3**: the failure path is T3 (RESERVED→FREE rollback) plus explicit
  invalidation of the staged DEV_READ (T8 op abort). The prefix/CoW cleanup at
  `schedule_policy.py:1237-1242` stops being five manual field resets: it is
  `lifetime.abort_op(cow_op)` + the req-level match-state reset.
- **G4**: `ring_state` is part of the record; T1 requires CLEAN-or-reset, T4/T5
  commit handlers require CLEAN dst (the resets at `mamba_component.py:625` and
  `745-748` become one rule; the translate inconsistency disappears because the
  reset is performed by the lifetime module which owns the id space, §7).
- **G5**: T6/T7/T8 carry the forward-write token; a reader that doesn't go
  through the token API has no way to obtain the slot's payload location
  (§3: `checkout_for_read` returns (physical ids, token)).
- **G6**: T10 is illegal with in-flight ops. The eviction drivers stop
  consulting `ongoing_write_through`/`ongoing_load_back` dicts (node-keyed)
  and ask the slot (`mamba_component.py:269-282` deleted in M5, §5).
- **G7**: the load op's token is the op's final `finish_event` — exactly what
  `model_runner.py:1590-1601` waits on today. R2 makes the wait a consequence
  of "forward consumes slots with in-flight H2D ops → forward stream must wait
  their tokens" rather than a special-cased counter probe.

### 1.3 Combinations that are impossible to express today

| Combination | Reality | R2 representation |
|---|---|---|
| DONATED ∧ BACKING_UP | exists; representable only as node-id membership in `ongoing_write_through` — lost across node splits unless `_replace_pending_write_through_node` fires correctly (`unified_radix_cache.py:1963-1993`) | DONATED + D2H op ref; node splits don't touch slot records |
| ACTIVE ∧ LOADING | exists; representable only via `_OngoingLoadBack.pinned_mamba_slots` (`297-306`) | ACTIVE + H2D op ref |
| PENDING_FREE ∧ LOADING | exists; the allocator's `_deferred_free` list | PENDING_FREE + H2D op ref |
| DONATED ∧ LOADING ∧ BACKING_UP | *possible in principle* (load-back commit for node A while write-through for ancestor B was split onto the same slot? No — slots are not shared across nodes; but a slot CAN be both H2D-target (tree restore) and D2H-source if a backup of the same node is queued in the same step) | legal in R2: `ops` is a set; ordering is by op tokens (the D2H attach requires the H2D op's token as its start precondition). **Decision: one owner state with an op refcount set, not a state×op product enum** — the product would need 5×2³ states and adds nothing |
| RESERVED at all | invisible today outside the reservation object | first-class |

### 1.4 OPEN questions on the state model

- **OPEN-1**: slot 0 is a reserved dummy write target for padded tokens
  (`allocator/mamba.py:136-139`, mirrored in the ckpt pool
  `mamba_checkpoint_pool.py:204`). Proposal: a permanent pseudo-state
  `RESERVED_DUMMY` outside the table (never allocatable, writes allowed,
  reads undefined). Needs confirmation that no path ever reads slot 0 back.
- **OPEN-2**: lazy ping-pong `-1` sentinel (`memory_pool.py:1394-1401`,
  `1480-1485`): is an unallocated second track-slot a lifetime concern or a
  req-local concern? Proposal: req-local (the sentinel never enters the
  table), but the donate path must then assert the donated id ≠ -1 without the
  `.item()` sync (`memory_pool.py:1430-1437`) — free once ids are
  CPU-shadowed (§7).
- **OPEN-3**: DFLASH draft-worker clones (`clone_with_new_mamba`,
  `memory_pool.py:1247-1280`) create a *second* allocator over a *different*
  pool keyed by draft layer. Proposal: one `SlotLifetime` instance per
  (pool, allocator) pair; the draft pool gets its own. Needs an audit that no
  draft slot id ever crosses into target-pool machinery.

---

## 2. Ownership + refcounting

One row per reference kind. "Invalidated by" = the unique transition after
which holding/using the reference is a bug.

| State | Reference holder | Where | Invalidated by |
|---|---|---|---|
| ACTIVE | `req.mamba_pool_idx` (device scalar tensor) | `schedule_batch.py:842`; set at `memory_pool.py:1302`, `mamba_component.py:171`, `626` | T9 (release) — all five release sites must null it; today each does its own `= None` (`memory_pool.py:1447`, `scheduler.py:3251`, `common.py:145`, `mamba_component.py:635`) |
| ACTIVE | `req.mamba_ping_pong_track_buffer` (+ device mirror `req_index_to_mamba_ping_pong_track_buffer_mapping`) | `memory_pool.py:1394-1414`, `schedule_batch.py:843` | T9 with keep-idx carve-out (`memory_pool.py:1449-1492`); the kept slot transitions T6 instead |
| ACTIVE | `req_index_to_mamba_index_mapping[req_pool_idx]` (device-side, read by kernels) | `memory_pool.py:1322` | `req_to_token_pool.free(req)`; stale rows are benign only because kernels index via live `req_pool_idx` — **must remain true in R2** |
| ACTIVE (session-parked) | `SessionSlot.mamba_pool_idx` (+ ping-pong fields) | `streaming_session.py:60-117`; counted by `session_held_mamba_slots` (`unified_radix_cache.py:3181-3182`) | session release / abort-carryover (`streaming_session.py:305-315`); **the session is an owner in the table (`owner_ref = session`), not an untracked alias** — the admission-reject guard already special-cases it (`scheduler.py:3245-3247`) |
| DONATED | tree `cd.value` (device tensor, shape [1]) | set `mamba_component.py:221/229` (insert), `752` (load-back commit) | T10 (evict, `cd.value = None` at `mamba_component.py:314`) — note `commit_hicache_transfer` stores a `.clone()` of the transfer's index tensor (`752`): two tensors alias one slot; in R2 the table entry is the truth and tensors are payload |
| DONATED (int8 variant) | tree `cd.value` holds **ckpt-pool** slot ids | `_commit_int8_checkpoint` (`mamba_component.py:443-450`), freed via ckpt allocator (`452-456`) | same T10, but in the ckpt pool's own table instance (closes latent hole §0.2) |
| RESERVED | `HybridWriteReservation`/`HybridLoadReservation.pool_reservation.transfers[*].device_indices` | `hybrid_cache_controller.py:67-88`, `590-638` | T3 (abort rollback `1062+`) or T4/T5 (commit) |
| any + op | queued `CacheOperation.pool_transfers[*].device_indices` in `write_queue`/`load_queue`, then merged ops (`merge_pool_transfers` concatenates — the merged tensor aliases every constituent slot) | `hybrid_cache_controller.py:471-482`, `640-650`, `102-127` | op completion (T12): ack drain pops `HiCacheAck` and the op record; the DMA thread's `_record_transfer_indices_on_stream` (`819+`, used at `582-587`, `719-724`) keeps the tensors alive until the stream passes them — that keep-alive stays |
| any + op | `_OngoingLoadBack.pinned_mamba_slots` / `ongoing_write_through` node records | `unified_radix_cache.py:289-306`, `555-556` | **deleted in R2** — replaced by per-slot op refs; the node-keyed dicts remain only for lock bookkeeping (`lock_params`) until M6 |
| DONATED + DEV_READ | `req.mamba_cow_src_index` → batched into `ForwardBatch.mamba_cow_src_indices` | staged `mamba_component.py:172`; batched `schedule_batch.py:2547-2566`; executed + nulled `model_runner.py:1495-1541` | consuming batch's `forward_done` (op complete), or G3 cleanup (`schedule_policy.py:1241`) / admission reject (`scheduler.py:3243`) (op abort) |
| ACTIVE | int8 dequant target = `mamba_cow_dst_indices` (active-pool ids) while src are ckpt-pool ids (`model_runner.py:1526-1532`) | same staging | same as above; **note the two id spaces in one op** — the op record must carry (pool, id) pairs, not bare ints |

**Refcount rule:** a slot's `ops` set is the only refcount. Owner is single
(exactly one of req/node/session/reservation). Anything else holding a tensor
that names the slot (batch tensors, ack queues, keep-alive rings
`scheduler.py:3592-3598`) is *payload plumbing* and must be reachable from an
op record or a live owner — the invariant checker (§8) enforces this by
sweeping all of the above columns against the table.

---

## 3. API: the `SlotLifetime` module

### 3.1 Placement

New class `MambaSlotLifetime` in
`python/sglang/srt/mem_cache/allocator/mamba.py` (same module as the
allocator), **owning** a `MambaSlotAllocator` instance rather than extending
it:

- The allocator keeps exactly one job: the free-list (`alloc`/`free`/
  `alloc_group_*`, `allocator/mamba.py:58-86`). `pin_slots`/`unpin_slots`/
  `_deferred_free` (`88-134`) move up and are deleted from the allocator.
- Both back-ends satisfy the same free-list protocol —
  `MambaSlotAllocator` and `UnifiedMambaSlotAllocator`
  (`unified_memory_pool.py:794-887`) — so one lifetime class covers static,
  unified, **and** the int8 ckpt pool (`mamba_checkpoint_pool.py:225`) by
  instantiation, which is what closes the two latent holes in §0.2.
- `HybridReqToTokenPool._init_mamba_pool` constructs it
  (`memory_pool.py:1213-1216`) and exposes it as
  `self.mamba_lifetime`; the tree cache and controller reach it via
  `req_to_token_pool` exactly as they reach the allocator today
  (`unified_radix_cache.py:2224`).

Why not extend `MambaSlotAllocator`: the ckpt pool embeds an allocator but
must NOT inherit pins-in-the-allocator semantics (that's the current bug), and
the unified allocator is a differently-shaped delegate; composition keeps one
lifetime implementation over three free-list back-ends.

### 3.2 Surface

```python
class OpKind(IntEnum): H2D_WRITE = 1; D2H_READ = 2; DEV_READ = 3

@dataclass(frozen=True)
class FenceToken:
    event: torch.cuda.Event | None      # None = already visible
    def wait_on(self, stream) -> None: ...

class MambaSlotLifetime:
    # -- acquisition / ownership (scheduler thread only) --
    def acquire(self, n: int, owner: OwnerRef) -> torch.Tensor | None
        # FREE->ACTIVE; returns device tensor of ids AND records CPU ints;
        # marks ring DIRTY->caller must init (clear/CoW/H2D) before kernel use.
    def acquire_group_begin(self, n)/acquire_group_end(self)   # wraps alloc_group_*
    def reserve(self, n: int, op_id: int) -> torch.Tensor | None   # FREE->RESERVED
    def commit_reservation(self, op_id, new_owner: OwnerRef)       # T4/T5
    def abort_reservation(self, op_id)                             # T3
    def donate(self, slot: int, node_id: int, write_token: FenceToken)  # T6
    def release(self, slots, owner) -> None                        # T9/T10
        # -> FREE or PENDING_FREE; idempotence is an error (double-free assert)

    # -- ops --
    def attach_op(self, slots, op_id: int, kind: OpKind,
                  completion: FenceToken) -> None                  # T12 attach
    def complete_op(self, op_id: int) -> None                      # T12 complete
        # MUST be called identically on every rank (see 3.3); flushes
        # PENDING_FREE slots whose op-set emptied.
    def abort_op(self, op_id: int) -> None                         # G3 path

    # -- queries (all O(1), no device access) --
    def state(self, slot: int) -> SlotState
    def cow_source_ok(self, slot: int) -> bool      # DONATED and no H2D op   (G2)
    def evictable(self, slot: int) -> bool          # DONATED, ops empty      (G6)
    def quiescent(self, slot: int) -> bool          # ops empty
    def read_token(self, slot: int) -> FenceToken   # what a reader stream waits (G5/G7)
    def note_forward_launch(self, forward_done: Event, written_slots: CpuIds) -> None
        # per-batch write-token update (see 4.1)

    # -- introspection for invariant checker / metrics --
    def counts(self) -> dict[SlotState, int]
    def dump(self) -> ...
```

Call-site mapping (what each existing path calls):

| Path | Calls |
|---|---|
| `HybridReqToTokenPool.alloc` (`memory_pool.py:1287-1328`) | `acquire(1, req)` |
| match CoW dst (`mamba_component.py:156-173`) | `acquire(1, req)`; `attach_op(src, batch_op, DEV_READ, next_forward_token)` — replaces `req.mamba_cow_src_index` bookkeeping *ownership* role (the field stays as plumbing) |
| `finalize_match_result` in-flight probe (`127-155`) | `cow_source_ok(cd.value)` |
| `prepare_load_back` (`595-627`) | `acquire(1, req)` (ring reset inside) |
| `reserve_load`/`reserve_write` extra-pool alloc (`hybrid_cache_controller.py:1083+`) | `reserve(n, op_id)` |
| load/write commit (`unified_radix_cache.py:2186-2231`, `1933-1951`) | `commit_reservation` + `attach_op(all named slots, op_id, H2D_WRITE / D2H_READ, ack_token)` — subsumes `pin_slots` |
| `loading_check`/`writing_check` ack drain (`2993-3042`, `2941-2991`) | `complete_op(op_id)` — subsumes `unpin_slots` |
| `prepare_for_caching_req` donate (`458-547`) | `donate(slot, node, forward_token)` |
| eviction drivers (`245-288`, `300-315`, `337-360`) | `evictable(slot)` gate + `release(slot, node)` |
| the five cleanup sites (§0.2 last row) | `release(slot, req)` — one choke point |

### 3.3 TP-rank symmetry

The table is **per-rank** (mamba state is rank-sharded — see the rank-local
MAMBA pool-transfer carve-out in `start_writing`,
`hybrid_cache_controller.py:515-531` and `566-573`); physical slot ids may
differ across ranks. What must be symmetric is the **transition sequence**,
because divergent tables ⇒ divergent admission/eviction ⇒ divergent trees ⇒
the mirrored-tree assumption underlying the whole coordinated protocol
collapses (stated explicitly at `unified_radix_cache.py:1789-1796` and
`1843-1848`).

Current transition triggers, classified:

| Trigger | Rank-local or symmetric today | R2 requirement |
|---|---|---|
| admission / alloc / donate / evict decisions | symmetric by mirrored trees + deterministic scheduler | unchanged; the invariant checker cross-checks `counts()` fingerprints (cheap CRC in the existing `_fingerprinted_check_reduce` piggyback) |
| DMA op **commit** | already post-consensus: `_all_ranks_succeeded` fingerprinted MIN/MAX reduce (`unified_radix_cache.py:474-504`, used at `1908-1931`, `2167-2186`) | `commit_reservation`/`attach_op` callable only after group success — assert via an `in_consensus` flag the cache sets around the commit block |
| DMA op **completion** | rank-local `finish_event.query()` polls, made symmetric by MIN-reducing the drainable ack count (`2961-2970`, `3000-3008`) | `complete_op` consumes only the reduced count — unchanged mechanism, relocated |
| one-rank exception in build/reserve | fail-closed into the consensus (`1901-1907`, `2047-2054`, `2115-2121`) | maps to `abort_reservation` on every rank (symmetric because group_succeeded=False everywhere) |
| **R1 batch-level reconciliation** | R1 (the companion redesign of the per-op collectives) replaces the per-op `_all_ranks_succeeded` calls and the two per-step check reduces with **one reconciliation reduce per scheduler step** carrying a vector of `(op_id, outcome)` pairs | `SlotLifetime`'s contract is deliberately narrower than R1: it requires only that `complete_op`/`abort_op` be invoked with an identical `(op_id, outcome)` sequence on all ranks. R1's reconciled outcome vector is exactly that sequence; until R1 lands, the existing fingerprinted reduces provide it. **OPEN-4**: op_id allocation must be symmetric — proposal: `(consensus_seq, node_id)` tuples, since `_hicache_consensus_seq` already increments in lockstep (`464`, `484`) |

---

## 4. Fencing integration

### 4.1 Tokens instead of ad-hoc fences

Two event families exist today; both become `FenceToken`s stored in the table:

1. **Forward-write token** — `forward_done`, recorded on the forward stream
   right after each launch (`scheduler.py:3606-3610`), consumed via
   `fence_state_read` (`unified_radix_cache.py:1769-1782`). R2: a single
   module-level `latest_forward_token` (exactly the current
   `_latest_forward_done_event`) is the conservative write token for **all**
   ACTIVE slots; `note_forward_launch(event, written_slots)` also stamps it
   per-slot so `read_token(slot)` can later be narrowed to per-slot precision
   without API change. `read_token()` for an ACTIVE slot returns it; T6/T7
   capture it at transition time.
   *Why per-slot stamping is deferred:* the batch's written-slot id list is
   already on CPU at launch (`req.mamba_pool_idx` per scheduled req), so the
   stamp is O(bs) dict writes — cheap — but the conservative global token is
   what today's code semantics are, and step M4 must be behavior-preserving.
2. **Op token** — the DMA ack `finish_event`
   (`hybrid_cache_controller.py:513-514`, `691`, recorded at `581`/`718`;
   surfaced as `HiCacheAck`, `managers/cache_controller.py:167-172`). Attached
   at T4/T5/T7. `read_token(slot)` for a slot with an in-flight H2D op returns
   the op token (a reader must see the loaded bytes); a D2H op does not gate
   readers (it is itself a reader) but gates T10 (free) and content mutation.

**The rule that replaces every scattered fence:** *a transition's effects
become visible to a consumer stream only through the token.* Concretely:

- schedule-stream donate copy (`mamba_component.py:536-545`): wait
  `read_token(src)` = forward token — deletes the open-coded
  `getattr(cache, "fence_state_read")` at `531-538`.
- schedule-stream `write_pos` RMW (`479-486`): same — deletes the second
  open-coded fence.
- write stream D2H (`unified_radix_cache.py:1850-1853`): the op's start waits
  the source slots' write tokens — moves the fence from "once per
  `write_backup` call on the current stream" to "recorded as the op's start
  dependency on the write stream" (strictly more correct: today the fence is
  on the *scheduler's* stream, and ordering transfers to the write stream only
  via `start_event.record()` on it, `hybrid_cache_controller.py:553-555`).
- forward stream consuming loaded slots (`model_runner.py:1590-1601`): wait
  `read_token` of every batch slot with an H2D op — identical effect to the
  current single final-event wait (all ops merge into one final event per
  `start_loading`, `hybrid_cache_controller.py:681-733`), but derived from the
  table instead of `layer_transfer_counter.consumer_index`.

### 4.2 Flow diagrams

Streams: `S`=schedule, `F`=forward, `L`=load (H2D), `W`=write (D2H),
`C`=copy/D2H-results (irrelevant to slots). `tok(x)` = FenceToken.

**(a) Donate (no_buffer, unfinished chunk)** — today
`mamba_component.py:509-546`:

```
S: acquire(dst)               T1  [dst: FREE->ACTIVE-ish staging]
S: wait tok(forward_done_N)       <- read_token(src=req.slot)     (was G5 fence, :536)
S: copy_from(src_phys, dst_phys)  device op on S, ordered after F's writes
S: donate(dst, node, tok(forward_done_N))                          T6
F: (batch N+1 launch) ......      no ordering needed vs the copy: F waits S
                                  via forward_stream.wait_stream(schedule_stream)
                                  scheduler.py:3559
```

**(b) Write-through of a donated checkpoint** — today
`unified_radix_cache.py:1830-1951` + `hybrid_cache_controller.py:508-588`:

```
S: reserve_write -> consensus -> commit_write                     T2..T4 (host side)
S: attach_op(cd.value slots, op, D2H_READ, tok(ack.finish))       T7
W: wait tok(forward_done_N)        op start precondition (was :1853 on S)
W: backup_extra_from_device_all_layer(...)   D2H reads slot
W: record ack.finish_event
   (cd.host_value was already published at commit,  mamba_component.py:733-737)
S: (later step) writing_check MIN-reduce -> complete_op(op)       T12
   -> slot evictable again; store events + write-lock release     (:1995-2007)
```

**(c) Load-back (tree restore + per-request CoW dst)** — today
`unified_radix_cache.py:2009-2233`, `hybrid_cache_controller.py:677-734`,
`model_runner.py:1590-1605`:

```
S: prepare_load_back: acquire(dst_req)  ring reset                T1  (:611-627)
S: reserve_load (tree slot)                                        T2
S: consensus (_all_ranks_succeeded)                               (:2167)
S: commit_load; cd.value=tree_slot                                 T4
S: attach_op({tree_slot, dst_req}, op, H2D_WRITE, tok(ack.finish)) (was pin_slots :2224)
S: start_loading (next sched step, scheduler.py:3295-3299)
L: per-layer H2D ... record ack.finish_event                       (:693-718)
F: (batch that consumes the prefix) wait tok(ack.finish)           G7 (:1598-1601)
F: deferred CoW copy_from(tree_slot, dst2) if staged               (:1603-1605)
S: loading_check MIN-reduce -> complete_op(op)                     T11/T12 (:3020-3031)
   -> PENDING_FREE slots flushed (was unpin_slots)
```

**(d) CoW from a matched node** — today `mamba_component.py:156-173`,
`schedule_batch.py:2547-2566`, `model_runner.py:1495-1541`:

```
S: match: cow_source_ok(src)?  else drop match                     G2 query
S: acquire(dst_req)                                                T1
S: attach_op(src, cow_op, DEV_READ, tok(forward_done_{N+1}))       T8
S: batch build: mamba_cow_src/dst_indices tensors                  plumbing
F: wait tok(read_token(src))  -- src may have an H2D op (case c)   G7
F: copy_from(src_phys, dst_phys); reset dst cursors                (:1535-1538, 982)
F: forward_done_{N+1}.record  -> completes cow_op                  T12
   [reject path instead: abort_op(cow_op); release(dst_req)        G3 (:1229-1242,
    prefix drop stays req-level]                                    3235-3252)
```

**(e) Spec-commit (ReplaySSM ring) + finish-donate RMW** — today
`hybrid_linear_attn_backend.py:90-152`, `commit_gdn_replayssm_spec`
(`:100`, `:342`), `mamba_component.py:471-486`:

```
F: decode/verify batch N: kernel writes ring; write_pos scatter    (:129-152)
F: forward_done_N.record
S: cache_finished_req for req R:
S:   wait tok(forward_done_N)      <- read_token(R.slot)           (was :482-484)
S:   cache_len -= write_pos[slot]; write_pos[slot]=0   RMW on S    (:485-486)
S:   donate/insert ...                                             T6
```

### 4.3 OPEN fencing questions

- **OPEN-5**: the RMW in (e) reads a device scalar (`write_pos_buf[idx].item()`
  at `mamba_component.py:485`) — a hard sync per finished request even after
  tokens. Fix belongs to §7 (shadow `write_pos` on CPU? it's device-authored,
  so shadowing requires the kernel to also publish a CPU copy, or moving the
  donate-cap computation onto F). Flagged, not solved here.
- **OPEN-6**: `copy_from` on S in flow (a) vs the *next* forward's reads of
  `dst`: ordered because F waits S wholesale (`scheduler.py:3559`). If that
  coarse wait is ever narrowed, T6's token must also gate F's first read of
  dst. The token API can express it; the current scheduler cannot violate it.
- **OPEN-7**: WAR direction (forward READS slot X; S then frees and
  reallocates X to a new req whose clear/CoW runs on F) — safe today because
  clear/CoW run on F *after* the reading forward (F is a single stream), and
  the WAR barrier covers shared-buffer writes on S (`scheduler.py:1624-1639`).
  If clears ever move to S or a second forward stream appears, T1 must carry a
  read-drain token. Record as an assert (acquire on a slot whose previous
  owner's last batch hasn't launched → error) rather than machinery.

---

## 5. Migration plan

Each step compiles, passes the existing suites (§8), and is independently
shippable/revertable. Estimates are engineering time including tests.

| Step | Content | Deletes | Est. |
|---|---|---|---|
| **M1** | Introduce `MambaSlotLifetime` in **shadow mode**: constructed by `HybridReqToTokenPool._init_mamba_pool` (`memory_pool.py:1213`), all call sites in §3.2 wired to *record* transitions; CPU int shadowing of ids (§7) lands here. Strict-assert env flag (`SGLANG_MAMBA_LIFETIME_CHECK`) makes any illegal transition raise; default = count + log. No behavioral dependency. | nothing | 3d |
| **M2** | Move pin/deferred-free into the table: `attach_op`/`complete_op` become authoritative; `MambaSlotAllocator.free` consults the table via a callback for PENDING_FREE parking. | `pin_slots`/`unpin_slots`/`_pinned`/`_deferred_free` (`allocator/mamba.py:88-134`, calls at `unified_radix_cache.py:2224`, `3028-3031`); the `.tolist()` syncs go with them | 2d |
| **M3** | Single release choke point: the five cleanup sites call `lifetime.release`; `free_mamba_cache` (`memory_pool.py:1441-1492`) becomes a thin wrapper; the admission-reject block (`scheduler.py:3240-3251`) and `finalize_load_back` (`mamba_component.py:629-635`) stop free-ing the allocator directly. Double-free asserts on. | the session-vs-fresh special-case comment logic at `scheduler.py:3236-3247` (subsumed by owner check); the `release_kv_cache` early-free branch (`common.py:136-146`) collapses to `release` | 3d |
| **M4** | Fence tokens: `note_forward_launch` feeds the table; `read_token` replaces `fence_state_read` at the three readers (`mamba_component.py:479-486`, `531-545`; `unified_radix_cache.py:1850-1853`); load-token wait in `model_runner.py:1590-1601` derives from the table. Behavior-identical (same events, same waits). | `fence_state_read`/`_latest_forward_done_event` on the cache (`unified_radix_cache.py:1764-1782`) as public API (kept internally for FULL-KV consumers until they migrate); both `getattr(..., "fence_state_read")` probes | 2d |
| **M5** | State queries replace dict probes: `finalize_match_result` uses `cow_source_ok` (delete the `ongoing_load_back` probe `mamba_component.py:136-142`); `_evict_excess_path_states` and eviction drivers use `evictable` (delete the node-dict skip `269-282`); admission's CoW-source node lock (`schedule_policy.py:1173-1187`) demoted to DEV_READ op attach. | G2's probe, G6's skip, the `lock_best_match` context (after a soak with both paths asserting agreement) | 3d |
| **M6** | Ops first-class: `reserve`/`commit_reservation`/`abort_reservation` wired through `HybridCacheController` reservations; `_OngoingLoadBack.pinned_mamba_slots` field removed; op_ids = `(consensus_seq, node_id)` ready for R1's reconciliation vector. | `pinned_mamba_slots` (`unified_radix_cache.py:297-306`, `2212-2231`); the mamba branch of the reservation rollback special-casing | 3d |
| **M7** | Second/third instances: int8 ckpt pool (`mamba_checkpoint_pool.py`) and `UnifiedMambaSlotAllocator` get lifetime tables; `_free_mamba_value` (`mamba_component.py:452-456`) routes through the right instance. Closes both §0.2 latent holes; unified+HiCache stays inert but now fails with a designed error, not `AttributeError`. | nothing (adds coverage) | 2d |
| **M8** | Delete shadow mode; strict asserts default-on in debug builds; upstream extraction (§6). | shadow plumbing | 1d |

Total ≈ 19 engineering-days. Guards G1–G7 are individually deletable at:
G1→M2, G4→M1 (ring_state assert) with the redundant resets deletable at M5,
G5→M4, G2/G3/G6→M5, G7→M4, G3's slot-half→M6. The req-level halves of G3
(prefix drop) stay forever — they are match-state policy, not slot lifetime.

---

## 6. Upstreaming strategy

### 6.1 Classification

| Piece | Upstream-generic? | Notes |
|---|---|---|
| `MambaSlotLifetime` core (states, ops, tokens, CPU shadowing) | **Yes** | depends only on `MambaSlotAllocator`, `HybridReqToTokenPool`, `torch.cuda.Event`; upstream has the same allocator (`allocator/mamba.py` minus our pin code) and the same donate/CoW paths in `MambaRadixCache` |
| G4 ring hygiene (T1 CLEAN rule) | **Yes** | upstream ReplaySSM has the identical previous-tenant hazard; the resets we ship (`mamba_component.py:617-626`, `743-748`, `memory_pool.py:1304-1308`) are already upstream-shaped |
| forward-write token (G5) | **Yes** | upstream runs the same overlap scheduler; their WAR fast path has the same reads-only scope (`scheduler.py:1624-1639`) |
| graph-compatible load fence (G7) | **Yes** | upstream `mamba2_layer_cache` per-layer `wait_until` (`memory_pool.py:1341-1345`) has the same replay hole |
| HiCache mamba transfers, pins, consensus integration (T2-T5, T7, T12) | **Partially** | the unified `MambaComponent` HiCache hooks exist upstream but the TP-coordination layer (`_all_ranks_succeeded` fingerprints, watermark eviction) is fork-hardened; upstream's **unified-memory HiCache wiring is inert** (no `pin_slots` on `UnifiedMambaSlotAllocator`, `unified_memory_pool.py:794-887`; `mamba_ckpt_pool=None` under unified, `:970`) — R2's M7 is the honest fix to propose |
| MLA host dedup interplay (`_mla_skip_host_io` + rank-local MAMBA pools, `hybrid_cache_controller.py:515-531`, `566-573`) | **Fork-specific** (dedup) | keep behind the existing rank-local pool-name set |
| rc5/SiTU, DFLASH lazy ping-pong specifics | **Fork-specific** | lifetime table treats them as owner/op flavors; no upstream coupling |
| int8 ckpt pool instance | **Fork feature** (`--enable-int8-mamba-checkpoint`) | PR-able as a feature, independent of R2 |

Relation to upstream's known open items:
- their **mamba-pressure TODO** (admission/retraction under mamba-slot
  pressure — cf. the fork's shared-pool budget gates,
  `schedule_policy.py:560-590`, `752-777`): R2's `counts()` gives the planner
  an exact FREE/PENDING_FREE split, which is the number their TODO needs
  (today `available_size()` silently under-counts by the deferred-free
  population).
- their **inert unified-memory HiCache wiring**: M7 makes the failure mode
  explicit and provides the missing lifetime layer; propose as the enabling
  PR rather than a bug report.
- their **DCP index-translation notes**: the v↔p translation ownership moves
  into the lifetime module (which stores CPU (virtual, physical) pairs at
  acquire time, §7), eliminating the caller-side `translate_mamba_indices`
  discipline that their DCP notes flag as error-prone
  (`memory_pool.py:1333-1339` contract; the two missed call sites in §0.2).

### 6.2 Proposed PR series (upstream `sgl-project/sglang`)

| # | Content | Size | Depends on |
|---|---|---|---|
| U1 | ReplaySSM ring hygiene on slot reuse (G4 resets + T1-CLEAN unit tests) | S (~150 loc) | — |
| U2 | forward-done write fence for schedule-stream state readers (G5, scheduler event + `fence_state_read`) | S (~120 loc) | — |
| U3 | graph-compatible HiCache load fence (G7) | S (~60 loc) | — |
| U4 | `MambaSlotLifetime` shadow mode + CPU id shadowing + invariant checker hooks | M (~600 loc) | U1 |
| U5 | pins/deferred-free → lifetime (G1) + load-back lifecycle tests | M (~400 loc, net-negative in allocator) | U4 |
| U6 | match/evict/admission queries (G2/G3/G6) + single release choke point | M | U5 |
| U7 | unified-memory + ckpt-pool instances (their inert-wiring fix) | M | U6 |

U1–U3 are extractions of already-soaked fork commits (`d248e0dc3`,
`58937d8ec`, `02a0257d9` lineage) and can go out immediately; U4+ track the
fork's M-steps after each soaks on K3.

---

## 7. Performance notes

Budget: every transition executes on the scheduler thread's critical path
(admission loop `scheduler.py:3186-3255` runs per waiting request per step).

- **All table state is CPU-side**: `dict[int, SlotRecord]` keyed by physical
  slot int + a `dict[int, set[int]]` op index. Every operation in §3.2 is O(1)
  dict work (attach_op over k slots is O(k), k ≤ a few per op).
- **No device syncs — the id-shadowing fix.** Root cause of the current
  syncs: slot ids are *born on device* (`free_slots` is a CUDA tensor,
  `allocator/mamba.py:138`; `alloc` returns slices of it) so any CPU decision
  about them forces `.tolist()` (`:93, 99, 124`) or `.item()`
  (`mamba_component.py:485`, `memory_pool.py:1432`, debug asserts
  `memory_pool.py:953-960`). Fix, landing in M1:
  1. the free-list's **authoritative copy moves to CPU** (`list[int]` /
     `collections.deque`);
  2. `acquire(n)` pops n ints, records them in the table, and materializes the
     device tensor with one async H2D
     (`torch.tensor(ids, dtype=int64, device=..., non_blocking pin)`); callers
     keep receiving the same-shaped tensor (`req.mamba_pool_idx` unchanged);
  3. `release`/`attach_op`/`complete_op` take the CPU ints from the table —
     never read a device tensor. Call sites that today only have a tensor
     (e.g. `cd.value`) get the ints from the table entry created when that
     tensor was born (the tensor is payload; the record is truth);
  4. for the unified pool, the (virtual, physical) pair is captured at acquire
     (one gather at alloc time, already device-side in
     `UnifiedMambaSlotAllocator.translate`, `unified_memory_pool.py:810-812`)
     — **OPEN-8**: unified-pool compaction remaps v→p asynchronously
     (`_compact_pending`, `unified_memory_pool.py:790`), so the CPU shadow of
     p can go stale; either compaction publishes remaps to the table
     (preferred: it is a scheduler-driven event) or the table stores virtual
     ids only and translation stays device-side for unified. Must be resolved
     before M7's unified instance.
- **Token bookkeeping**: `FenceToken` wraps an existing event — no new event
  creation beyond the one `forward_done` per batch that HEAD already records
  (`scheduler.py:3606`). Per-slot write-token stamping is O(bs) dict writes
  per launch; the conservative global token (M4) is O(1).
- **What gets cheaper**: `pin_slots`/`unpin_slots` D2H syncs deleted (M2); the
  `free()` pinned-scan `.tolist()` deleted; `alloc_group_begin`'s device
  `split(1)` iterator (`allocator/mamba.py:58-72`) becomes a CPU slice; the
  lazy-donate `.item()` debug assert becomes a free CPU compare.
- **What stays device-side**: the actual state copies/clears/DMA — unchanged.
- **Shadow-mode overhead** (M1, both systems live): one extra dict update per
  existing operation; measured target < 1 µs per transition, i.e. noise
  against the admission loop's existing per-req work. Verify with the
  scheduler-loop profile before/after on the K3 bench (§8).

---

## 8. Test plan

### 8.1 Unit — state-transition property tests (new, `test/registered/unit/mem_cache/test_mamba_slot_lifetime.py`)

- **Legal-graph property test**: hypothesis-style random walks over the API;
  assert (a) every reachable record matches §1.2, (b) FREE is emitted only
  with `ops == ∅`, (c) double-release and attach-on-FREE raise, (d) owner is
  always unique, (e) `counts()` conserves pool size.
- **Guard-regression vectors**, one per G1–G7, expressed as transition
  sequences that must be rejected or parked (e.g. G1: acquire → attach H2D →
  release ⇒ PENDING_FREE; acquire must not return that id until complete_op).
- **Mocked-stream fence-ordering tests**: fake `Event` objects recording
  (record, wait) order per fake stream; replay flows (a)–(e) of §4.2 and
  assert every read of a slot's payload is preceded (in that stream's op list)
  by a wait on the slot's current token. This machine-checks the "effects
  visible only through the token" rule without a GPU.
- **Symmetry harness**: two table instances driven by the same
  op-outcome sequence with different physical ids must produce identical
  state-count fingerprints; inject a divergent `complete_op` and assert the
  checker trips. (Extends `test_hi_mamba_hicache_tp_transactions.py`,
  which already unit-tests the consensus fail-closed paths.)

### 8.2 Existing suites that must stay green at every M-step

- `test/registered/unit/mem_cache/test_mamba_unittest.py`,
  `test_mamba_donated_alloc_ratio.py`, `test_unified_mamba_views.py`
- `test_hi_mamba_hicache_tp_transactions.py` (TP consensus),
  `test_hi_mamba_host_watermark_eviction.py`,
  `test_hi_mamba_path_state_cap.py`, `test_mamba_path_state_cap.py`
- `test/registered/radix_cache/test_int8_mamba_checkpoint_e2e.py` (M7)
- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py`
- invariant checker runs (`managers/scheduler_components/invariant_checker.py`
  consumes `session_held_mamba_slots` and allocator sizes; extend it to sweep
  the table per §2's refcount rule under
  `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY`, `scheduler.py:1672-1673`).

### 8.3 Integration probes (exist, scratchpad of the k3-dedup campaign)

- **`mamba_two_phase.py`** — the workload that caught G1/G4/G6: Phase A builds
  400 conversations (host pool below watermark) and revisits 120; Phase B
  builds 600 more (pool exhausts, watermark eviction + load-back churn) and
  revisits the same 120 + 60 late ones, checking codeword recall with
  `temperature=0`. Verdict classes: `clean` / `base-path-corruption` (Phase A
  failures) / `eviction-correlated-corruption` (Phase B-only failures).
  **Green = verdict `clean`, zero revisit failures in both phases**, on a
  fresh TP8 canary with HiCache+dedup+DFLASH+overlap, twice consecutively
  (the failure modes are probabilistic; single-pass green under-detects).
- **`mamba_pressure.py` / `mamba_serial.py` / `mamba_only_repro.py`** —
  narrower reproducers from the same campaign (device-pool pressure loop,
  serialized revisits, no-HiCache control). Green = no wrong-codeword and no
  server assert/hang; the no-HiCache control must stay green to attribute
  failures.
- **Soak**: the overnight canary battery (bench + probe alternation) with
  `SGLANG_MAMBA_LIFETIME_CHECK=1` — green = zero lifetime asserts in logs over
  ≥ 12 h, TPM within noise of the pre-R2 baseline (the §7 no-regression gate).
- Each migration step M2–M7 ships only after: unit suite green + two-phase
  probe `clean` ×2 + soak with strict asserts.

### 8.4 What green does NOT prove (honest limits)

- The probes exercise TP8 with mirrored schedulers; a genuinely divergent-rank
  bug (dropped message, one-rank exception swallowed outside the fail-closed
  regions) surfaces as the desync logs (`unified_radix_cache.py:445-455`,
  `2975-2990`, `3011-3026`) — the symmetry harness (§8.1) covers the table's
  reaction, not the transport.
- PD-disaggregation and unified-memory-pool paths are not exercised by the K3
  battery (dedup canary is colocated, static pool); M7's unified instance
  needs its own probe — OPEN-9, blocked on OPEN-8.

---

## Appendix A — consolidated OPEN items

| ID | Question | Blocking |
|---|---|---|
| OPEN-1 | slot-0 dummy: pseudo-state + read audit | M1 |
| OPEN-2 | lazy ping-pong `-1` sentinel stays req-local? | M1 |
| OPEN-3 | draft-worker pool: separate lifetime instance + id-space audit | M7 |
| OPEN-4 | symmetric op_id scheme `(consensus_seq, node_id)` vs R1's ids | M6 |
| OPEN-5 | `write_pos` donate-cap RMW still syncs (`mamba_component.py:485`) | perf follow-up, not correctness |
| OPEN-6 | narrowing the coarse F-waits-S order would activate T6's dst token | none (assert only) |
| OPEN-7 | WAR direction on slot reuse if clears leave the forward stream | none (assert only) |
| OPEN-8 | unified-pool compaction vs CPU physical-id shadow | M7 |
| OPEN-9 | unified-pool integration probe | post-M7 |
