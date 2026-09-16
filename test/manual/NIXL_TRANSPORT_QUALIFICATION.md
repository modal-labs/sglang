# NIXL transport qualification — 2026-09-16

## Scope and result

Runtime commit: `5c9375e1308d5921a2e313bffcb2c90aebceda07`.
The isolated app `ap-2hBGjYqV7e9fuH7jHPOJ1Q` used the same DCP8 prefill/decode,
full-draft KV, Mamba, 64-running-request and memory settings as the candidate.
Only UCX logging changed: `UCX_PROTO_INFO=y`, `UCX_LOG_LEVEL=info` on both roles.
No forced retraction, mirrored traffic, transport policy change, or production mutation.

- Prefill: `ta-01M2MJ5PGHHD6R0Q3VPQCEKWSR`, engine PID 10.
- Decode: `ta-01M2MJ5PM1KQ11ZKT7A3PAE7VR`, engine PID 9.
- Actual engine SHA and process environments were read from both containers.
- NIXL/nixl-cu13 1.3.2; bundled UCX reports 1.21.0.
- The app became ready at 07:37:37 UTC, under five minutes after deployment.

**CUDA-to-CUDA zero-copy RDMA was selected for the actual test transfer.**
During a cold 8192-token request at 07:37:38 UTC, the prefill UCX logs included:

    inter-node cfg#2
    remote memory write by ucp_put*(multi) from cuda/GPU4 to cuda/dev[4]
    1..inf | zero-copy | rc_mlx5/mlx5_11:1 50% on path0 and 50% on path1

Clean header-plus-protocol rows survive for ranks 4, 5, 6 and 7, respectively using
mlx5_11, mlx5_15, mlx5_14 and mlx5_6. Rank 2's CUDA-to-CUDA header survives but its
protocol row is truncated. Shared multiprocess stdout truncated/interleaved some
table writes; there is no complete lane table for ranks 0, 1 and 3 in the retained
logs. Do not describe this as eight clean per-rank protocol proofs.

The test completed HTTP 200 in 0.538 seconds, with zero cached tokens and zero
retractions. A second distinct cold 8192-token request completed HTTP 200 in
0.456 seconds. Its stable before/after metric snapshots show exactly one transfer
sample on each of eight ranks:

| Measurement | Result |
| --- | --- |
| Accounted payload per rank | 53.1943359375 MiB |
| Accounted payload across eight ranks | 425.5546875 MiB |
| Reported transfer duration across ranks | 4.040–4.659 ms |
| Reported transfer rate across ranks | 11.149–12.859 GiB/s |
| Rank 0 duration/rate | 4.046 ms / 12.838 GiB/s |

The first request's initial metric scrape was empty while warmup counters were
being created. Its later count includes warmup, so it is not used for a per-request
metric delta. The second request has a populated stable baseline.

This disproves a general inability of this profile to use GPU RDMA. It does not
prove identical lane choice on every live machine, sustained saturation bandwidth,
or that every live latency spike is caused by prefill computation.

## What the existing transfer metrics measure

At this runtime pin:

- `NixlKVSender.send` starts timing immediately before the first chunk enters
  `add_transfer_request`'s worker queue.
- `transfer_worker` consumes chunks, prepares descriptors, posts transfers and
  polls handles to completion; it marks the room successful after its final chunk.
- `NixlKVSender.poll` stops timing only when the scheduler next observes success.

The duration therefore includes worker queueing, descriptor preparation,
inter-chunk gaps while later prefill work runs, transfer completion polling and
scheduler observation delay. It is **not a NIC wire-time measurement**.
The reported rate divides accounted payload by this same wall-clock duration.
A low rate under load alone cannot establish host staging or low NIC bandwidth.

`ReqTimeStats.compute_and_observe_kv_transfer_metrics` divides bytes by 1024 twice
for the quantity named MB, then by 1024 again for the rate named GB/s. The units
are MiB and GiB/s. Each observation is per reporting sender rank. This QA exports
all eight ranks; averaging these observations does not produce node-wide bandwidth.
Summing overlapping per-rank rates is also not a rigorous node throughput estimate.

## Payload accounting audit

`CommonKVSender.get_transfer_metric` counts target page indices times target page
byte strides, plus each state component's own index count times its own byte
strides. The equal DCP8-to-DCP8 topology has one destination per rank, so its
replication factor is one.

`setup_state_kv_args` registers full draft KV separately as `DFLASH_KV`.
The final prefill chunk supplies draft page indices in the logical token domain.
A resolved-runtime check found `speculative_draft_window_size=4096` and
`speculative_dflash_draft_ring=false`: this candidate allocates a full logical
draft pool but transfers only the page-aligned 4096-token draft tail. It does not
transfer the entire draft prefix. With window=None, the same code would transfer
the full prefix. Draft indices/bytes are not divided by the target DCP factor.
Mamba contributes its per-request state-slot bytes. Transferred padding is counted;
small auxiliary metadata and protocol overhead are not.

A read-only CPU check executed the actual generic NIXL descriptor-construction
and metric methods over 20 K3-shaped cases: lengths 1–100000, partial pages,
and contiguous/fragmented allocation. Target-plus-draft metric bytes matched
descriptor bytes in every case. The fixtures use illustrative byte strides, not
an independent read of the running model's tensors. Equal-layout Mamba accounting
was source-audited separately; heterogeneous TP/PP and invalid zero-pointer
components are outside this check.

## Reproducibility and limitations

Durable evidence is recorded in the accompanying evidence directory's
`SHA256SUMS`: process provenance, raw startup/transfer logs, exact synthetic
request driver, response metadata, stable counter deltas, lane excerpts, source
audit inputs and the CPU accounting check. No user request bodies were collected.

UCX's protocol tables report operation, source/remote memory types and selected
size-range protocol/lane configuration. The retained CUDA-to-CUDA tables were
created by the actual first transfer, whereas startup tables mentioning host
memory or self/intra-node endpoints are not sufficient evidence.

Relevant UCX 1.21 source:
- [PROTO_INFO configuration](https://github.com/openucx/ucx/blob/v1.21.0/src/ucp/core/ucp_context.c)
- [Protocol table memory types and lane descriptions](https://github.com/openucx/ucx/blob/v1.21.0/src/ucp/proto/proto_debug.c)

NIXL's exposed `query_xfer_backend` only identifies UCX; it does not expose the
selected transport lane. GPU registration, mapped CUDA/verbs libraries, uverbs
file descriptors and aggregate NIC counters alone cannot distinguish direct GPU
transfer from a host bounce. No live process was attached to mutate UCX state.

The prepared-descriptor fallback remains a separate qualification item.
Historical rank-7 `NIXL_ERR_NOT_FOUND` evidence does not yet have a minimized cause.
Current full-draft state geometry differs from the old combined target/draft
region layout, so an isolated prepared-versus-descriptor A/B can test present need
for the workaround without retroactively claiming the historical cause.

## Cold/warm and allocation-order baseline

The descriptor fallback completed a frozen 23-request synthetic trace:
cold/repeated 8K, cold/two repeated 128K, 16 mixed 4K/8K/16K/32K requests at
concurrency eight with staggered 16/64/128-token outputs, and two fresh 8K requests.
All returned HTTP 200, zero retractions; both roles ended with zero running,
waiting and stage queues, and decode used tokens returned to zero.

| Case | Request wall time | Actual cached tokens | Transfer metric, rank 0 |
| --- | --- | --- | --- |
| Cold 8K | 0.463 s | 0 | 4.059 ms, 53.194 MiB |
| Repeated 8K | 0.455 s | 0 | 2.238 ms, 53.194 MiB |
| Cold 128K | 6.168 s | 0 | 5283.819 ms, 255.694 MiB |
| Repeated 128K, first | 1.084 s | 114688 | 846.756 ms, 255.694 MiB |
| Repeated 128K, second | 0.333 s | 130560 | 172.290 ms, 255.694 MiB |

The repeated 8K case had no measured cache hit and is not evidence of a warm-prefix
benefit. The cold 128K transfer metric reports only 0.047 GiB/s, even on the same
proven GPU RDMA path. Its timer spans later prefill chunks; this directly
demonstrates why low request-level transfer speed is not pure network bandwidth.
The exact split among compute gaps, descriptor work, transfer queueing and
scheduler observation still requires finer timing.

Trace SHA256: `2fefa66392fbe2f19ef2d2270a341e0a564a7349a4fc874c75ffff1202db70b3`.
The prepared-path comparison must use this same trace and inspect actual cache
hits, not assume equal warmth. No prepared-path result has been claimed yet.
