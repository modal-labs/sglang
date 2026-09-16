# kimi-k3-fast -> dev/instinct/2026-09-15 @ c10d0d215, 4 PRs on

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
head equals `RELEASE_SHA`. For the current pin, use `engine-c10d0d215.bundle` and
`RELEASE_SHA=c10d0d21512d9df6a85c9968fa69080a9b16059e`.

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

Image build check (in the build log): `rev-parse HEAD == c10d0d215...`, `status --porcelain` empty,
`sglang.__file__` printed. The A/B endpoint `kimi-k3-ab-instinct` was built with the identical steps.

## During cutover

- Watch green containers schedule; if they don't, scale Instinct down a little to free GPUs.
- Old (blue) containers are hard-terminated 4 h after deploy regardless of green progress.
- Per-container readiness line: `Kimi K3 TP8 DFlash is ready (release c10d0d215). ...` — the
  warmup number now includes the image turn (expect +25-80 s on containers with a cold mm JIT).
- Env check on a running container: `SGLANG_RELEASE_SHA=c10d0d215...` and the 5 flags above;
  `SGLANG_ENABLE_MM_CUDA_IPC_PREFIX_ACK` must be absent.

## Rollback

Check out the previous `serve.py` commit, regenerate its bundle with the same command, then deploy.
Flags-only rollback (keep c10d0d215, drop the 5 env lines) leaves the dev levers at their default-off
values; at 462ade71f that configuration was == `release/2026-09-14` behavior, greedy-identical 36/36
in every A/B arm.

## Not verified

- A B300:8 container boot with the image warmup turn (#516 is CI-green, not container-tested).
- `modal deploy` from this directory against Instinct's workspace (only run in modal-labs/cust-instinct
  as `kimi-k3-ab-instinct`, 1 container, aws/us-east-1, offline HF layout).
