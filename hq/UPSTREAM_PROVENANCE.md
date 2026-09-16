# CUDA-IPC lease-pool provenance

## Source commits

- `443b62db57` (#33949): stream-ordered CUDA-IPC feature-pool lifecycle.
- `ae2bd5728b` (#37047): multimodal transport failure cleanup; only the
  transport, processor, and selected `schedule_batch` hunks are ported.
- `0a57403468` (#36411): Qwen multimodal transport keys used by the upstream
  lease-pool implementation.
- `upstream/main` at `fb91baedab3e1de668d6e8f391ccd81cab9eacd`: source snapshot
  for the copied transport modules.

## Ported files

- `python/sglang/srt/multimodal/transport/memory_pool.py`
- `python/sglang/srt/multimodal/transport/cuda_ipc.py`
- Selected transport lifecycle hunks in
  `python/sglang/srt/multimodal/processors/base_processor.py`
- Selected proxy cleanup hunks in
  `python/sglang/srt/managers/schedule_batch.py`
- New-pool tests in
  `test/registered/unit/multimodal/test_cuda_ipc_lease_pool.py`
- CPU-only abort cleanup tests in
  `test/registered/unit/managers/test_multimodal_abort_cleanup.py`

SHA-256 comparisons against `upstream/main`:

| File | Upstream | Local |
| --- | --- | --- |
| `multimodal/transport/memory_pool.py` | `56ffd932157fd0f9aeefb68e8b1b84bb0e1e2e696424e99b488b22e5f4e7347c` | `56ffd932157fd0f9aeefb68e8b1b84bb0e1e2e696424e99b488b22e5f4e7347c` |
| `multimodal/transport/cuda_ipc.py` | `6e9f4ba94a9da7850bcf7397e18bbf81e758d1ddbd6f8b0add95f0bb868941b2` | `6e9f4ba94a9da7850bcf7397e18bbf81e758d1ddbd6f8b0add95f0bb868941b2` |

## Drift hunks

- `python/sglang/srt/multimodal/transport/__init__.py:1`: docstring-only
  package initializer. The fork already provides
  `determine_tensor_transport_mode` in `managers/mm_utils.py`; duplicating it
  would create competing transport-mode helpers.
- `python/sglang/srt/environ.py:973`: adds the fork-specific,
  default-off `SGLANG_MM_CUDA_IPC_LEASE_POOL` switch.
- `python/sglang/srt/utils/cuda_ipc_transport_utils.py:1-31`: compatibility
  shim selecting the legacy or stream-ordered implementation at import time;
  `PRECOMPUTED_FEATURE_HASHES_KEY` remains fork-owned.
- `python/sglang/srt/utils/cuda_ipc_transport_utils_legacy.py`: moved without
  content edits so flag-off behavior remains byte-identical to the base branch.
- `python/sglang/srt/multimodal/processors/base_processor.py:364-377`: passes
  the fork's `server_args.tp_size` as the new pool consumer count only when
  the lease-pool flag is enabled; the old three-argument constructor remains
  in the flag-off branch.
- `python/sglang/srt/multimodal/processors/base_processor.py:1353-1427`:
  gates the upstream stream-ordered wrapping and rollback path while retaining
  the old sync-flag wrapping statements in the false branch.
- `python/sglang/srt/managers/schedule_batch.py:422-454,577-626,1716-1722`:
  gates consumer-count resolution, abandoned-proxy cleanup, reconstruction
  rollback, feature release, and request-abort cleanup for the new pool.
- `python/sglang/srt/multimodal/processors/kimi_k3.py`: no upstream hunk was
  applied because this fork no longer has the old per-item wrapping pattern;
  its feature-sink path already returns `mm_items` directly.

## Not ported

- `python/sglang/srt/multimodal/transport/cuda_vmm_transport_utils.py`
- `python/sglang/srt/utils/cuda_vmm_utils.py`: only `check_drv` and
  `_get_cuda_driver` are ported; the VMM transport itself is not ported.
- The `mm_schedule.py` split; the fork retains this code in `mm_utils.py`.
- Scheduler/request-receiver SHM error-consensus portions of #37047.
- Upstream's transport-mode exports in `transport/__init__.py`, for the
  docstring-only rationale above.

## Commit C

- `python/sglang/srt/multimodal/transport/lease_guard.py`: fork-owned
  bounded per-slot generation state and stale-read/write protections.
- `python/sglang/srt/multimodal/transport/cuda_ipc.py`: imports
  `lease_guard`, resolves the consumer rank, and checks the bounded generation
  guard before reconstruct/borrow readiness waits.
- `python/sglang/srt/multimodal/transport/cuda_ipc.py`: overrides
  `_acknowledge_on_stream` to guard own-word and full-group acknowledgements,
  record the generation only after stream writes succeed, and refuse stale
  writes.
- `python/sglang/srt/managers/mm_utils.py`: flag-on deferred cache hits
  acknowledge each rank's own word instead of only rank zero.
- `python/sglang/srt/models/kimi_k25.py` and `kimi_k3.py`: flag-on image
  materialization uses own-word acknowledgement; K3 and K2.5 image-wise DP
  release non-owner acknowledgement words after the DP encoder call.
- `python/sglang/srt/managers/schedule_batch.py` and `scheduler.py`: threaded
  full-group consumer counts through reconstruction/release and broadcast-mode
  entry-rank materialization.

## Commit D

- `python/sglang/srt/managers/schedule_batch.py`: fork-owned prefix-resident
  own-word acknowledgement computed once from each request's matched prefix,
  plus full proxy consumer-count resolution for broadcast materialization.
- `python/sglang/srt/managers/mm_utils.py`: fork-owned per-image cache-hit and
  duplicate-hash own-word acknowledgements before the batched miss encoding
  phase.
- `python/sglang/srt/managers/scheduler.py`: broadcast reconstruction now uses
  the largest consumer group declared by the raw proxy items rather than the
  scheduler broadcast group size.

## Commit E

- `test/registered/unit/multimodal/test_lease_pool_invariants.py`: CPU-only
  mock protocol tests covering eventual recycling/cancellation, non-overlap,
  stale-generation read refusal, monotonic acknowledgement words, and the
  strict upstream leak trace versus own-word guarded recycling.
- `test/registered/unit/multimodal/test_lease_pool_scenarios.py`: CPU-only
  scenario coverage for waiting-timeout aborts, session close ownership,
  n>1 clone detachment, mid-processing sink failures, interleaved
  acknowledgements, orphaned-owner cleanup, and inflight stale generations.
- `test/registered/unit/multimodal/test_lease_lifecycle.py`: verifies the
  legacy shim's base-branch SHA-256, flag-off object identity/import
  isolation, and flag-on transport selection.

## Validation (abl-p01, B300x8 TP8, prod-shaped args/env)

Same-box off1 -> on -> off2, 30 min replay per arm plus hot-image and
abort/timeout/n=2 injection traffic; artifacts under `/results/lane11-ab/`.

- Legacy pool (off arms): full after ~270 s, 2459/2485 CPU fallbacks
  (93.6%), 92/91 slices still occupied after the drain.
- Lease pool (on arm): 2620 wraps, 0 CPU fallbacks, peak 355 MiB of 1 GiB,
  max 29 concurrent leases; occupancy returns to zero between every idle
  gap (180 gaps >= 5 s, including the 80 s drain before the post-drain
  probe), 0 stale-lease refusals, 0 tracebacks.
- 36-row greedy probe identical on all arms; image-turn p50 TTFT
  -83/-92 ms vs off1/off2 (box bias +2 ms), text neutral, TPOT/accept
  unchanged; p90/p99 tails not attributable at this sample size.
