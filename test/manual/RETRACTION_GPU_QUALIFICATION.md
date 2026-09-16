# PD retraction GPU qualification

Qualified engine pin: `5c9375e1308d5921a2e313bffcb2c90aebceda07` (2026-09-16).
This is bounded correctness evidence, not a throughput benchmark.

## Bugs found and fixed

1. **Duplicate allocator-page release:** speculative tail cleanup rounded to the
   physical 64-token page, while DCP8 frees logical 512-token pages. Immediate
   retraction could free the committed boundary page twice. The first forced GPU
   run ended at **-2560 used tokens**. Commit `8b7b2e6cc0` uses allocator page size;
   the repeat completed 18/18 requests and 35 native-request retractions, then
   returned to zero used tokens. This matches existing upstream behavior.

2. **A stale checkpoint boundary:** PD overlap scheduling retracted before
   processing the previous speculative result. GPU KV/Mamba had advanced, but
   CPU output history and the bonus token still lagged. Delayed result processing
   then skipped the retracted request's accepted block. Byte-perfect save/restore
   alone could not detect this inconsistency.
   Commit `5c9375e130` settles pending results before PD offload, filters completed
   requests, rechecks capacity, and clears the outer loop's consumed-result
   bookkeeping. Healthy decode steps retain overlap. Six actual-method CPU
   regressions pass, including natural pressure and partial completion.

## Controlled GPU results

Both roles used DCP8, full DFlash draft KV, and the production-based K3 candidate.
Normal and forced pairs had the same immutable pin and deployment profile.
Only forced decode set `SGLANG_TEST_RETRACT=true`, interval 16, and
`SGLANG_TEST_RETRACT_VERIFY=true`; actual process environments were captured.

Each cohort sent six waves of six concurrent requests: arithmetic, retrieval of
a synthetic identifier, and a synthetic red/blue image. Each request also had to
produce the exact integer sequence 1–96. All raw answers and finish reasons are
retained, including failures.

| Runtime | Normal control | Forced retraction |
|---|---:|---:|
| Before boundary fix (`59355684c3`) | 36/36 exact checks | 31/36 exact checks |
| After boundary fix (`5c9375e130`) | 36/36 exact checks | 36/36 exact checks |

Before the fix, forced requests produced two skipped sequences and two empty
final image answers, all with finish reason `stop`; these were not output-limit
truncations. The fifth strict failure was numerically correct `543.0` versus
`543`, caused by ambiguous fixture wording. Every affected request had actually
been retracted. All 144 requests across the four cohorts returned HTTP 200.

After the fix:

- 21 retractions per TP rank; normal controls had none.
- **168/168 GPU byte comparisons passed**, 21 on each of eight ranks; none skipped.
- Every comparison covered 24 target KV tensors, five persistent KDA Mamba
  tensors, and 12 full-draft KV tensors, with 302–990 saved tokens.
- Each test run ended with decode used tokens, running requests, and all
  waiting/preallocation/transfer/retracted queues at zero.
- No matching ERROR, traceback, CUDA-error, or OOM records in the captured final
  forced-decode runtime log.

## Scope and limits

The verifier is disabled by default and bounded to 32 restores per allocator
and 4096 tokens per restore. It adds copies/synchronization only in explicit QA.
These tests qualify the current K3/DFlash KDA speculative path with full draft
KV. They do not qualify GDN circular ReplaySSM, non-spec linear ReplaySSM,
draft-SWA rings, every model, arbitrary long-context restore, or production
performance. Greedy token identity across separate batches was not used as the
sole correctness criterion: normal runs already exhibited numerical differences.

Durable private evidence, including raw cohorts, runtime logs, process
provenance, CPU regressions, earlier allocator-failure artifacts, and
`SHA256SUMS`:

`/home/ec2-user/k3-productionization-evidence/20260916T070321Z-5c937-retraction/`

The generic pending-result ordering fix is being prepared separately for
upstream. The allocator-page fix already exists upstream. The verifier and
synthetic harness are qualification tooling.

## Isolated decode-role failure and recovery

On the same qualified pin, the idle forced-QA decode engine received one
SIGKILL after assertions verified its app, role, process, pin, and QA flags.
The process-aware heartbeat detected exit code -9 about 3.5 seconds later.
Modal automatically replaced both gang tasks without a manual redeploy.

- Injected failure: 2026-09-16 07:15:16.490 UTC.
- Replacement router, prefill, and decode healthy: 07:20:21.610 UTC.
- Exact text and image smoke checks both HTTP200/pass: 07:20:23.080 UTC.
- Total recovery plus smoke: **306.6 seconds**, within the ten-minute bound.
- Subsequent idle probe: zero decode used tokens, running requests, and queues.

Router `/health` briefly remained200 after decode was unavailable. It proves
router process health, not full PD serving readiness; qualification therefore
checked both engines and real text/image generation. This test does not claim
uninterrupted availability for a single-pair endpoint.

Recovery evidence and checksums:
`/home/ec2-user/k3-productionization-evidence/20260916T071426Z-5c937-role-recovery/`
