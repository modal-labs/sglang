# Instinct `kimi-k3-fast` deploy files for dev/instinct/2026-09-15 @ 15fafd786

The deploy recipe and procedure for the 2026-09-15 Instinct cutover are version-controlled next to the
code they deploy. See `DEPLOY.md` for the procedure, checks and rollback.

| file | what |
| --- | --- |
| `serve.py` | the deployed recipe: prod `kimi-k3-fast` serve.py + 3 edits (bundle pin, 5 flags, image warmup turn) |
| `engine-<sha9>.bundle` | generated, gitignored — see `DEPLOY.md` |
| `kimi-k3-fast_to_release-instinct-0914.diff` | diff of `serve.py` against the previous prod `kimi-k3-fast` serve.py (2c881e2e, patch-based) |
| `DEPLOY.md` | deploy / cutover / rollback notes |
| `serve_probe.py`, `health_probe.diff` | **not deployed** — variant of `serve.py` whose health check exercises the real chat path (pool lane 2 proposal); diff is `serve.py -> serve_probe.py` |

Bundles are generated and verified per `DEPLOY.md` (prerequisite b9e90a6d6 must exist in the repo).
