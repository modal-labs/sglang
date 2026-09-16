# Instinct `kimi-k3-fast` deploy files for release/instinct/2026-09-14 (462ade71f)

Everything needed to run `modal deploy serve.py` for the 2026-09-15 Instinct cutover, version-controlled
next to the code it deploys. See `DEPLOY.md` for the procedure, checks and rollback.

| file | what |
| --- | --- |
| `serve.py` | the deployed recipe: prod `kimi-k3-fast` serve.py + 3 edits (bundle pin, 5 flags, image warmup turn) |
| `rel0914.bundle` | `git bundle` of fork commits `b9e90a6d6..462ade71f` (+ `release/2026-09-14` @ aba9973b7); `serve.py` `add_local_file`s it and does `git fetch rel0914.bundle 462ade71f && git checkout` inside the image whose HEAD is b9e90a6d6 |
| `kimi-k3-fast_to_release-instinct-0914.diff` | diff of `serve.py` against the previous prod `kimi-k3-fast` serve.py (2c881e2e, patch-based) |
| `DEPLOY.md` | deploy / cutover / rollback notes |
| `serve_probe.py`, `health_probe.diff` | **not deployed** — variant of `serve.py` whose health check exercises the real chat path (pool lane 2 proposal); diff is `serve.py -> serve_probe.py` |

Verify the bundle: `git bundle verify rel0914.bundle` (prerequisite b9e90a6d6 must exist in the repo).
