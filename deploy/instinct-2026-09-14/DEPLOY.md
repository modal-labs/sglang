# kimi-k3-fast -> dev/instinct/2026-09-15 @ fd8aff798, 4 PRs of #22 + #26 on

Files in this directory:

- `serve.py` — the prod `kimi-k3-fast` serve.py with exactly three edits
  (see `kimi-k3-fast_to_release-instinct-0914.diff` — historical: the diff of the original 462ade71f
  recipe against the prior prod serve.py; only `RELEASE_REF`/`RELEASE_SHA` have moved since):
  1. fork pin: the two patch steps are replaced by `git fetch engine-<sha9>.bundle <RELEASE_SHA> && git checkout`
     (bundle = fork commits since the image HEAD b9e90a6; the old bump + PR31633 are ancestors).
  2. runtime env: the 4 PRs of #22 opted in (5 assignments):
     `SGLANG_TRTLLM_MLA_VERIFY_FUSED_KV_WRITE=1` (#18), `SGLANG_TRTLLM_MLA_FUSED_CHUNK_KV_PACK=1` (#17),
     `SGLANG_KIMI_ENCODE_FAST_PATH=1` + `SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS=32000000` (#20),
     `SGLANG_K3_MM_USE_RENDERED_INPUT_IDS=1` (#19). No prefix-ack (#15 dropped).
     Plus #26 (dev, CONFIRMED same-box): `SGLANG_K3_SCHED_MM_FASTPATH=1` + `SGLANG_K3_MM_STRIP_PROCESSOR_INPUT_IDS=1`.
     Also in this pin but NOT set (default off, flip in prod after their gates): `SGLANG_MM_CUDA_IPC_LEASE_POOL=1`
     (#33, after the IPC stress soak), `SGLANG_PREFILL_CUDA_GRAPH_MIN_REPLAY_BUCKET=128` (#29) and the other
     default-off dev levers (#37/#38/#43) as their same-box rows land. Rollback of any flip = unset the var.
  3. warmup: one 1280x800 synthetic-PNG image turn after the text greeting (autoinference #516).
  4. HiCache host tier re-sized for the 60-minute session TTL (see "HiCache sizing" below):
     `--enable-mla-hicache-host-dedup --hicache-size 140 --hicache-mamba-ratio 13.5 --hicache-write-policy write_through`
     replaces `--hicache-ratio 3 --hicache-write-policy write_through_selective`.
  Unchanged: app name, model/draft, other server args, volumes, B300:8, min_containers=63,
  routing/kv-aware routing, auth, heartbeat/terminate, JIT-cache salt (`SGLANG_EFFECTIVE_COMMIT`).

## Bundle generation (per validated dev merge)

Bundles are generated locally per `RELEASE_SHA` and are never committed. Each bundle contains every
fork commit since b9e90a6d6, so committing a bundle would embed the previous committed bundle(s) and
grow without bound.

```sh
git update-ref refs/deploy/engine-<sha9> <RELEASE_SHA> && git bundle create deploy/instinct-2026-09-14/engine-<sha9>.bundle b9e90a6d6ef1859830c3b879cef999092975a41a..refs/deploy/engine-<sha9> b9e90a6d6ef1859830c3b879cef999092975a41a..origin/release/2026-09-14
git bundle verify deploy/instinct-2026-09-14/engine-<sha9>.bundle
git bundle list-heads deploy/instinct-2026-09-14/engine-<sha9>.bundle
```

The pin ref is needed because bundles only advertise named refs; serve.py checks that the advertised
head equals `RELEASE_SHA`. For the current pin, use `engine-fd8aff798.bundle` and
`RELEASE_SHA=fd8aff798ca487d72db3341a5057e84394196af5`.

Then run `modal deploy serve.py`. `serve.py` refuses to import if the bundle is missing, fails
`git bundle verify`, or lacks `RELEASE_SHA`; the exception message carries the exact create
command, so a bare `python -c "import serve"` (from this directory, before generating the bundle)
prints the command to run. The git checks run with `cwd` = this directory, so `modal deploy` works
from any cwd.

## Deploy (same workspace/env/profile as the last `kimi-k3-fast` redeploy)

```sh
# 0. capacity: Instinct concurrent-GPU limit vs current usage; need headroom for the
#    green containers to schedule (last time the trough was 34 running containers).
modal deploy serve.py          # in the directory that holds serve.py + generated bundle
```

Image build check (in the build log): `rev-parse HEAD == fd8aff798...`, `status --porcelain` empty,
`sglang.__file__` printed. The A/B endpoint `kimi-k3-ab-instinct` was built with the identical steps.

## During cutover

- Watch green containers schedule; if they don't, scale Instinct down a little to free GPUs.
- Old (blue) containers are hard-terminated 4 h after deploy regardless of green progress.
- Per-container readiness line: `Kimi K3 TP8 DFlash is ready (release fd8aff798). ...` — the
  warmup number now includes the image turn (expect +25-80 s on containers with a cold mm JIT).
- Env check on a running container: `SGLANG_RELEASE_SHA=fd8aff798...` and the 7 flag lines above;
  `SGLANG_ENABLE_MM_CUDA_IPC_PREFIX_ACK` must be absent.

## Rollback

Check out the previous `serve.py` commit `540bbe0021`, regenerate `engine-3ccb60f5b.bundle`
with the same command, then deploy.
Flags-only rollback (keep 3ccb60f5b, drop all 7 env lines — the 5 prod flags and the 2 #26 lines) leaves
the dev levers at their default-off values; at 462ade71f the 5-flags-off configuration was ==
`release/2026-09-14` behavior, greedy-identical 36/36 in every A/B arm, and on 741f05e61 the flags-off
parity leg was 36/36 identical to prod with Δp50 +0 ms.

## DFlash2 draft (kimi-k3-sglang #27 + #40 + #48 step-16000 pin, all in RELEASE_SHA 3ccb60f5b (also in fd8aff798); draft-epoch-2 pin)

`serve.py` points the draft at `/dflash/k3-instinct-v5-dflash2/draft-epoch-2` (`DFLASH2_PINNED_STEP`) on the
existing `dflash_spec` mount (`DFlash2DraftModel`), `--speculative-draft-model-quantization unquant`
(no fp8 activation-scheme arg). The v1 draft dir stays in place.

- Pinned step: `draft-epoch-2` (final checkpoint of the run) on `dflash-experimental`
  (env `cust-instinct`, `outputs/k3-instinct-v5-dflash2-hero-b8-30p2t-success-producer-v2/trainer-0` —
  note trainer-0, the `draft-step-*` checkpoints live under trainer-1). Same
  `target_config_sha256 == 2a5cb51c…` pin; `model.safetensors` 5,677,368,712 B, 96 BF16 tensors,
  sha256 `34aec290324eeafdf49704be9d01de8abb3a407df6e1893315f2b8698280948d`. Copied to
  `dflash_spec:k3-instinct-v5-dflash2/draft-epoch-2` 2026-09-16 19:47Z (size + sha256 of all three files
  verified against the source). Validation (same-box p10,
  engine at dev head e4aeedf60a, 7 deploy flags, epoch-2 vs 16000): 36-row greedy content/tool_calls
  35/36 (row 743 only, a target near-tie row that also flips under target-only kernel changes);
  15-min replay accept length 4.52 vs 4.60, paired TPOT/TTFT deltas not significant (bootstrap CIs
  straddle 0, Wilcoxon p=0.71/0.055), 0 errors both arms; T=0.7 100-request replay 0 errors and 0
  per-rank candidate/output mismatches across 8 TP ranks → SAME.
- Evaluation (integration arm, no copy to `dflash_spec`): deploy with
  `K3_DFLASH2_VOLUME=dflash-experimental K3_DFLASH2_VOLUME_ENV=cust-instinct modal deploy serve.py`.
  The volume is mounted read-only at `/dflash2-eval` and the draft path defaults to
  `/dflash2-eval/outputs/k3-instinct-v5-dflash2-hero-b8-30p2t-success-producer-v2/trainer-0/draft-epoch-2`
  (override with `K3_DFLASH2_PATH=<dir>`). The boot check runs on the active path. Prod never sets
  these variables.
- Ship procedure: (1) **required before deploying this pin:** copy the chosen checkpoint from the training volume into the immutable dir
  `k3-instinct-v5-dflash2/<step>` on `dflash_spec` (done for `draft-epoch-2`; steps rotate on the training volume;
  12000 disappeared once — until the copy exists the boot check fails fast); (2) bump
  `DFLASH2_PINNED_STEP` if a newer step is chosen; (3) integration lane advances `RELEASE_SHA` +
  `rel0914.bundle` to a dev head containing #27.
- Boot: `check_dflash2_checkpoint` runs before the engine starts and aborts the container with a
  `RuntimeError` if the dir is missing/incomplete, `model.safetensors` is truncated (size != safetensors header), `architectures`
  is not `["DFlash2DraftModel"]`, or `target_config_sha256` != the pinned value. Expect
  `DFlash2 draft checkpoint OK: ...` then `DFLASH selector decode (greedy + sampling) folded into the
  draft cuda graph` on every rank; `/metrics` weighted accept length ≈ 4.3 (v1: ≈ 3.4).
- Rollback to v1 (engine may stay): `SPECULATIVE_DRAFT_MODEL_PATH = f"{DFLASH_MOUNT_PATH}/k3-instinct-v5-epoch1"`,
  `DRAFT_QUANTIZATION = "fp8"` + restore `"--speculative-draft-fp8-activation-scheme": "static"`,
  drop the `check_dflash2_checkpoint` call; redeploy.

## HiCache sizing (dedup 140 GB KV / 13.5x Mamba, write_through)

With the router's session affinity TTL at 60 min, the host tier must hold a session's prefix for up to an
hour. At `--hicache-ratio 3` (735 GB host, 8 replicated KV copies) the pool cycles in ~20-30 min. MLA host
dedup keeps one KV copy across the 8 TP ranks, so the same memory holds ~7x more distinct prefix tokens.

Measured (5% prod mirror replay, 1/3 of per-replica session pressure, 60-min sticky affinity, three
simultaneous fresh 8xB300 boxes, 45-min scored window, tree 24423fcdd3; HiCache code is byte-identical to
3ccb60f5b):

| arm | token hit | 30-45 min wake | 45-60 min wake | TTFT p50 | queue-full errors |
|---|---|---|---|---|---|
| ratio 3 selective (previous prod) | 88.0% | 44% | 28% | 4.6 s | 66 |
| dedup 130 GB / mamba 16 / selective | 94.9% | 95% | 79% | 0.4 s | 0 |
| **dedup 140 GB / mamba 13.5 / write_through** | **95.0%** | **97%** | **82%** | **0.4 s** | **0** |

Host memory at steady state: RSS sum ~815 GiB, max rank ~216 GiB (rank 0 owns the dedup pool); pinned
pools ~797 GiB (modelled from the pool sizes), under the fork's 800 GiB host-pool budget. Wakes >= 60 min miss on every arm (the pool has
cycled), so the TTL and the pool horizon are matched. Caveat: at full prod inflow every arm's retention
horizon is shorter than measured; the ranking is unaffected (the gap is ~50x the cross-box band).

Check on a running container: `--enable-mla-hicache-host-dedup` in the server command line and the
HiCache init log reporting the host KV pool at 140 GB with the Mamba host pool sized 13.5x device.

Fallback if rank-0 RSS is a problem: `--hicache-size 130 --hicache-mamba-ratio 16 --hicache-write-policy
write_through_selective` (within 0.05 pt overall, -2 to -3 pt on 20-60 min wakes, ~45 GiB less pinned).

Rollback to the previous tier: drop `--enable-mla-hicache-host-dedup --hicache-size --hicache-mamba-ratio`,
restore `--hicache-ratio 3 --hicache-write-policy write_through_selective`. Cache contents are per-container
and rebuilt from traffic either way; no engine code change is involved (RELEASE_SHA unchanged).

Validation (2026-09-17, L27 gate runs on fresh boxes, same tree as the measured arms):

- Output parity: greedy 36-row probe (29 reasoning, 1 content, 6 tool-call; 8 image rows
  containing literal `<|sep|>`), idle engines, dedup 140/13.5 write_through vs current
  ratio-3 selective: 36/36 byte-identical.
- Full-inflow stability: 15 min at 1x per-replica session inflow (all sessions, 6x pace, no
  flush) on the 140/13.5 arm: health 200 throughout, 0 tracebacks, 0 OOM, rank-0 RSS flat
  at 217 GiB (sum 818 GiB) under both queue saturation and idle, host KV pool 97% full.
  Retention at full inflow could not be scored: 2/3 of sessions arrived cold (~130k uncached
  tokens each) and a single replica ingests ~19k uncached tok/s, so the box stayed
  queue-saturated (running 10-11 / queued 16, TTFT dominated by queue wait). Warm-path
  buckets that did complete stayed at 99%+. The 1/3-slice 45-min comparison above
  therefore remains the retention estimate; horizons shorten at full inflow for every arm.

## Not verified

- A B300:8 container boot with the image warmup turn (#516 is CI-green, not container-tested).
- `modal deploy` from this directory against Instinct's workspace (only run in modal-labs/cust-instinct
  as `kimi-k3-ab-instinct`, 1 container, aws/us-east-1, offline HF layout).
