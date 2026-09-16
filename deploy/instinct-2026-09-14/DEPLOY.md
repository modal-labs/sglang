# kimi-k3-fast -> dev/instinct/2026-09-15 @ 741f05e61, 4 PRs on

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
     Also in this pin but NOT set (default off, flip in prod after their gates): `SGLANG_MM_CUDA_IPC_LEASE_POOL=1`
     (#33, after the IPC stress soak), `SGLANG_PREFILL_CUDA_GRAPH_MIN_REPLAY_BUCKET=128` (#29) and the other
     default-off dev levers (#26/#37/#38/#43) as their same-box rows land. Rollback of any flip = unset the var.
  3. warmup: one 1280x800 synthetic-PNG image turn after the text greeting (autoinference #516).
  Unchanged: app name, model/draft, server args, volumes, B300:8, min_containers=63,
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
head equals `RELEASE_SHA`. For the current pin, use `engine-741f05e61.bundle` and
`RELEASE_SHA=741f05e61c1c8c3efc3bc0d067f2cf788a63b8d3`.

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

Image build check (in the build log): `rev-parse HEAD == 741f05e61...`, `status --porcelain` empty,
`sglang.__file__` printed. The A/B endpoint `kimi-k3-ab-instinct` was built with the identical steps.

## During cutover

- Watch green containers schedule; if they don't, scale Instinct down a little to free GPUs.
- Old (blue) containers are hard-terminated 4 h after deploy regardless of green progress.
- Per-container readiness line: `Kimi K3 TP8 DFlash is ready (release 741f05e61). ...` — the
  warmup number now includes the image turn (expect +25-80 s on containers with a cold mm JIT).
- Env check on a running container: `SGLANG_RELEASE_SHA=741f05e61...` and the 5 flags above;
  `SGLANG_ENABLE_MM_CUDA_IPC_PREFIX_ACK` must be absent.

## Rollback

Check out the previous `serve.py` commit, regenerate its bundle with the same command, then deploy.
Flags-only rollback (keep 741f05e61, drop the 5 env lines) leaves the dev levers at their default-off
values; at 462ade71f that configuration was == `release/2026-09-14` behavior, greedy-identical 36/36
in every A/B arm.

## DFlash2 draft (kimi-k3-sglang #27 + #40, both in RELEASE_SHA 741f05e61)

`serve.py` points the draft at `/dflash/k3-instinct-v5-dflash2/draft-step-16000` (`DFLASH2_PINNED_STEP`) on the
existing `dflash_spec` mount (`DFlash2DraftModel`), `--speculative-draft-model-quantization unquant`
(no fp8 activation-scheme arg). The v1 draft dir stays in place.

- Pinned step: `draft-step-16000` (supplied 2026-09-16 ~06:55Z) on `dflash-experimental`
  (env `cust-instinct`, `outputs/k3-instinct-v5-dflash2-hero-b8-30p2t-success-producer-v2/trainer-1`).
  Same `target_config_sha256 == 2a5cb51c…` pin; the copy step below validates architecture, target hash
  and safetensors length. Previous pin `draft-step-14500` (5,677,368,712 B, 96 tensors) stays in its
  immutable dir as the fallback (`DFLASH2_PINNED_STEP = "draft-step-14500"`). Measured numbers in #27
  are on draft-step-13750; 14500 was validated by the integration arm. **16000 is not yet validated** —
  this pin ships only after the same-box p24 comparison (36-row greedy + 15-min replay: accept length,
  TPOT, TTFT, errors) is posted on its PR and reads BETTER or SAME vs 14500.
- Evaluation (integration arm, no copy to `dflash_spec`): deploy with
  `K3_DFLASH2_VOLUME=dflash-experimental K3_DFLASH2_VOLUME_ENV=cust-instinct modal deploy serve.py`.
  The volume is mounted read-only at `/dflash2-eval` and the draft path defaults to
  `/dflash2-eval/outputs/k3-instinct-v5-dflash2-hero-b8-30p2t-success-producer-v2/trainer-1/draft-step-16000`
  (override with `K3_DFLASH2_PATH=<dir>`). The boot check runs on the active path. Prod never sets
  these variables.
- Ship procedure: (1) **required before deploying this pin:** copy `draft-step-16000` from the training volume into the immutable dir
  `k3-instinct-v5-dflash2/draft-step-16000` on `dflash_spec` (steps rotate on the training volume;
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

## Not verified

- A B300:8 container boot with the image warmup turn (#516 is CI-green, not container-tested).
- `modal deploy` from this directory against Instinct's workspace (only run in modal-labs/cust-instinct
  as `kimi-k3-ab-instinct`, 1 container, aws/us-east-1, offline HF layout).
