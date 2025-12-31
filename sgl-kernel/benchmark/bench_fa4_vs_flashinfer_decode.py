#!/usr/bin/env python3
"""
Decode attention microbench (paged KV) comparing FA4 vs FlashInfer.

This script is intentionally minimal and focuses on the decode attention kernels.
It does NOT include:
  - QKV projection, rotary, KV write, or logits
  - multi-GPU communication / TP collectives

Examples
--------
# Small-batch decode sweep (common in online serving)
python sgl-kernel/benchmark/bench_fa4_vs_flashinfer_decode.py sweep-shapes \\
  --batch-sizes 1,2,4,8,16,32 \\
  --seqlens-k 1024,2048,4096 \\
  --num-heads 8 --num-kv-heads 1 --head-dim 128 --page-size 128

# GLM-ish per-rank head ratio 12 (e.g. 96 q heads / 8 kv heads with TP=8 -> 12 / 1)
python sgl-kernel/benchmark/bench_fa4_vs_flashinfer_decode.py sweep-shapes \\
  --batch-sizes 8 \\
  --seqlens-k 65536 \\
  --num-heads 12 --num-kv-heads 1 --head-dim 128 --page-size 128 \\
  --fa4-pack-gqa auto,false

# Sweep FA4 num_splits for one shape
python sgl-kernel/benchmark/bench_fa4_vs_flashinfer_decode.py sweep-num-splits \\
  --batch-size 1 --seqlen-k 4096 \\
  --num-heads 8 --num-kv-heads 1 --head-dim 128 --page-size 128 \\
  --num-splits 1,2,4,8,16,32,0
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Optional

# Disable FA4 warmup by default so this benchmark only compiles what it uses.
# You can override this by running with: SGLANG_DISABLE_FA4_WARMUP=0
os.environ.setdefault("SGLANG_DISABLE_FA4_WARMUP", "1")

import torch


def _parse_int_list(csv: str) -> list[int]:
    if not csv:
        return []
    return [int(x) for x in csv.split(",") if x.strip()]


def _parse_choice_list(csv: str) -> list[str]:
    if not csv:
        return []
    return [x.strip() for x in csv.split(",") if x.strip()]


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp8_e4m3", "fp8e4m3", "float8_e4m3", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if name in ("fp8_e5m2", "fp8e5m2", "float8_e5m2"):
        return torch.float8_e5m2
    raise ValueError(f"Unsupported dtype: {name} (use bf16, fp16, fp8_e4m3, or fp8_e5m2)")


def _bench_cuda_events(fn: Callable[[], None], iters: int, repeats: int) -> tuple[float, float]:
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end) / iters)
    return statistics.mean(times_ms), statistics.pstdev(times_ms)


@dataclass(frozen=True)
class PagedKV:
    k_cache: torch.Tensor  # (num_pages_total, page_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor
    page_table: torch.Tensor  # (B, pages_per_seq) page indices into k/v cache
    cache_seqlens: torch.Tensor  # (B,) int32 tokens
    max_seqlen_k: int


def _make_paged_kv(
    batch_size: int,
    seqlen_k: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> PagedKV:
    pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq

    # torch.randn does not support float8 dtypes directly.
    base_dtype = torch.bfloat16 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else dtype
    k_cache = torch.randn(
        (total_pages, page_size, num_kv_heads, head_dim), device=device, dtype=base_dtype
    ).to(dtype)
    v_cache = torch.randn(
        (total_pages, page_size, num_kv_heads, head_dim), device=device, dtype=base_dtype
    ).to(dtype)

    base = torch.arange(pages_per_seq, device=device, dtype=torch.int32)[None, :]
    offsets = (torch.arange(batch_size, device=device, dtype=torch.int32)[:, None] * pages_per_seq)
    page_table = base + offsets

    cache_seqlens = torch.full(
        (batch_size,), seqlen_k, device=device, dtype=torch.int32
    )
    return PagedKV(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=seqlen_k,
    )


def _make_flashinfer_decode_plan(
    batch_size: int,
    seqlen_k: int,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    device: torch.device,
    workspace_mb: int,
    use_tensor_cores: bool,
):
    try:
        import flashinfer  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "flashinfer is not installed. Install flashinfer-python/flashinfer-jit-cache first."
        ) from e

    import flashinfer

    pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    last_page_len = seqlen_k - (pages_per_seq - 1) * page_size

    indptr = (
        torch.arange(0, total_pages + 1, pages_per_seq, device=device, dtype=torch.int32)
        if batch_size > 0
        else torch.zeros((1,), device=device, dtype=torch.int32)
    )
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page = torch.full((batch_size,), last_page_len, device=device, dtype=torch.int32)

    workspace = torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, backend="cutlass", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        indptr,
        indices,
        last_page,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    return wrapper


def _fa4_decode_call(
    q: torch.Tensor,
    paged_kv: PagedKV,
    causal: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    cu_seqlens_q = torch.arange(
        0, q.shape[0] + 1, dtype=torch.int32, device=q.device
    )
    return flash_attn_with_kvcache(
        q=q,
        k_cache=paged_kv.k_cache,
        v_cache=paged_kv.v_cache,
        page_table=paged_kv.page_table,
        cache_seqlens=paged_kv.cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        max_seqlen_k=paged_kv.max_seqlen_k,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=causal,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        ver=4,
    )


def _parse_pack_gqa_choice(choice: str) -> Optional[bool]:
    c = choice.strip().lower()
    if c in ("auto", "none"):
        return None
    if c in ("true", "1", "yes", "on"):
        return True
    if c in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Invalid pack_gqa choice: {choice} (use auto/true/false)")


def _print_env_banner(device: torch.device):
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={props.name} cc={props.major}.{props.minor} sm={props.multi_processor_count} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    try:
        import flashinfer

        print(f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
    except Exception:
        print("flashinfer=not-importable")
    try:
        import sgl_kernel

        print(f"sgl_kernel={getattr(sgl_kernel, '__version__', 'unknown')} path={sgl_kernel.__file__}")
    except Exception:
        print("sgl_kernel=not-importable")


def cmd_sweep_shapes(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    kv_dtype = _parse_dtype(args.dtype)
    torch.manual_seed(args.seed)

    _print_env_banner(device)
    print(
        "backend,batch_size,seqlen_k,num_heads,num_kv_heads,head_dim,page_size,num_splits,pack_gqa,mean_ms,std_ms,tokens_per_s"
    )

    fa4_pack_choices = [_parse_pack_gqa_choice(x) for x in _parse_choice_list(args.fa4_pack_gqa)]
    if not fa4_pack_choices:
        fa4_pack_choices = [None]

    fa4_num_splits = _parse_int_list(args.fa4_num_splits)
    if not fa4_num_splits:
        fa4_num_splits = [0, 1]

    fi_use_tensor_cores = _parse_pack_gqa_choice(args.flashinfer_use_tensor_cores)
    if fi_use_tensor_cores is None:
        gqa_group_size = args.num_heads // args.num_kv_heads
        if kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            fi_use_tensor_cores = True
        elif kv_dtype in (torch.bfloat16, torch.float16):
            fi_use_tensor_cores = gqa_group_size >= 4
        else:
            fi_use_tensor_cores = False

    for batch_size in _parse_int_list(args.batch_sizes):
        for seqlen_k in _parse_int_list(args.seqlens_k):
            paged_kv = _make_paged_kv(
                batch_size=batch_size,
                seqlen_k=seqlen_k,
                page_size=args.page_size,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                dtype=kv_dtype,
                device=device,
            )
            q_base_dtype = (
                torch.bfloat16
                if kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                else kv_dtype
            )
            q_fa4 = torch.randn(
                (batch_size, args.num_heads, args.head_dim), device=device, dtype=q_base_dtype
            ).to(kv_dtype)

            # FlashInfer supports mixed (Q, KV) dtypes; FA4 CuTe requires Q/K/V to match.
            fi_q_dtype = (
                torch.bfloat16
                if kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                else kv_dtype
            )
            q_fi = torch.randn(
                (batch_size, args.num_heads, args.head_dim), device=device, dtype=torch.bfloat16
            ).to(fi_q_dtype)

            fi_wrapper = None
            if not args.skip_flashinfer:
                fi_wrapper = _make_flashinfer_decode_plan(
                    batch_size=batch_size,
                    seqlen_k=seqlen_k,
                    page_size=args.page_size,
                    num_q_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    q_dtype=fi_q_dtype,
                    kv_dtype=kv_dtype,
                    device=device,
                    workspace_mb=args.flashinfer_workspace_mb,
                    use_tensor_cores=bool(fi_use_tensor_cores),
                )

            def _flashinfer_run():
                if fi_wrapper is None:
                    return None
                if kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    return fi_wrapper.run(q_fi, (paged_kv.k_cache, paged_kv.v_cache), k_scale=1.0, v_scale=1.0)
                return fi_wrapper.run(q_fi, (paged_kv.k_cache, paged_kv.v_cache))

            # Warmup
            for _ in range(args.warmup_iters):
                if fi_wrapper is not None:
                    _flashinfer_run()
                for pack_gqa in fa4_pack_choices:
                    for num_splits in fa4_num_splits:
                        _fa4_decode_call(
                            q=q_fa4,
                            paged_kv=paged_kv,
                            causal=args.causal,
                            num_splits=num_splits,
                            pack_gqa=pack_gqa,
                        )
            torch.cuda.synchronize()

            if fi_wrapper is not None:
                mean_ms, std_ms = _bench_cuda_events(
                    _flashinfer_run,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                toks_s = batch_size / (mean_ms * 1e-3)
                print(
                    f"flashinfer,{batch_size},{seqlen_k},{args.num_heads},{args.num_kv_heads},{args.head_dim},{args.page_size},,,"
                    f"{mean_ms:.6f},{std_ms:.6f},{toks_s:.3f}"
                )

            for pack_gqa in fa4_pack_choices:
                for num_splits in fa4_num_splits:
                    mean_ms, std_ms = _bench_cuda_events(
                        lambda: _fa4_decode_call(
                            q=q_fa4,
                            paged_kv=paged_kv,
                            causal=args.causal,
                            num_splits=num_splits,
                            pack_gqa=pack_gqa,
                        ),
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    toks_s = batch_size / (mean_ms * 1e-3)
                    pack_str = "auto" if pack_gqa is None else str(bool(pack_gqa)).lower()
                    print(
                        f"fa4,{batch_size},{seqlen_k},{args.num_heads},{args.num_kv_heads},{args.head_dim},{args.page_size},"
                        f"{num_splits},{pack_str},{mean_ms:.6f},{std_ms:.6f},{toks_s:.3f}"
                    )

            # Reduce allocator noise between shapes
            if args.empty_cache_between_shapes:
                torch.cuda.empty_cache()


def cmd_sweep_num_splits(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    torch.manual_seed(args.seed)

    _print_env_banner(device)
    print(
        "num_splits,batch_size,seqlen_k,num_heads,num_kv_heads,head_dim,page_size,pack_gqa,mean_ms,std_ms,tokens_per_s"
    )

    paged_kv = _make_paged_kv(
        batch_size=args.batch_size,
        seqlen_k=args.seqlen_k,
        page_size=args.page_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    q_base_dtype = (
        torch.bfloat16 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else dtype
    )
    q = torch.randn(
        (args.batch_size, args.num_heads, args.head_dim), device=device, dtype=q_base_dtype
    ).to(dtype)

    pack_gqa = _parse_pack_gqa_choice(args.fa4_pack_gqa_single)
    num_splits_list = _parse_int_list(args.num_splits)

    # Warmup
    for _ in range(args.warmup_iters):
        for ns in num_splits_list:
            _fa4_decode_call(
                q=q, paged_kv=paged_kv, causal=args.causal, num_splits=ns, pack_gqa=pack_gqa
            )
    torch.cuda.synchronize()

    for ns in num_splits_list:
        mean_ms, std_ms = _bench_cuda_events(
            lambda: _fa4_decode_call(
                q=q,
                paged_kv=paged_kv,
                causal=args.causal,
                num_splits=ns,
                pack_gqa=pack_gqa,
            ),
            iters=args.iters,
            repeats=args.repeats,
        )
        toks_s = args.batch_size / (mean_ms * 1e-3)
        pack_str = "auto" if pack_gqa is None else str(bool(pack_gqa)).lower()
        print(
            f"{ns},{args.batch_size},{args.seqlen_k},{args.num_heads},{args.num_kv_heads},{args.head_dim},{args.page_size},"
            f"{pack_str},{mean_ms:.6f},{std_ms:.6f},{toks_s:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_shapes = sub.add_parser("sweep-shapes", help="Sweep (B, seqlen_k) and compare FA4 vs FlashInfer.")
    p_shapes.add_argument("--device", default="cuda:0")
    p_shapes.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp8_e4m3", "fp8_e5m2"],
        help="Data type used for both Q and KV tensors in this microbench.",
    )
    p_shapes.add_argument("--seed", type=int, default=0)

    p_shapes.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p_shapes.add_argument("--seqlens-k", default="1024,2048,4096")
    p_shapes.add_argument("--num-heads", type=int, default=8)
    p_shapes.add_argument("--num-kv-heads", type=int, default=1)
    p_shapes.add_argument("--head-dim", type=int, default=128)
    p_shapes.add_argument("--page-size", type=int, default=128)
    p_shapes.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)

    p_shapes.add_argument("--iters", type=int, default=200)
    p_shapes.add_argument("--repeats", type=int, default=5)
    p_shapes.add_argument("--warmup-iters", type=int, default=10)
    p_shapes.add_argument("--empty-cache-between-shapes", action=argparse.BooleanOptionalAction, default=False)

    p_shapes.add_argument("--skip-flashinfer", action="store_true")
    p_shapes.add_argument("--flashinfer-workspace-mb", type=int, default=256)
    p_shapes.add_argument(
        "--flashinfer-use-tensor-cores",
        default="auto",
        choices=["auto", "true", "false"],
        help="Enable FlashInfer tensor core kernels. Default auto matches SGLang heuristics.",
    )

    p_shapes.add_argument(
        "--fa4-num-splits",
        default="0,1",
        help="Comma-separated list. Use 0 for auto. Example: 1,2,4,8,16,32,0",
    )
    p_shapes.add_argument(
        "--fa4-pack-gqa",
        default="auto",
        help="Comma-separated list from {auto,true,false}. Example: auto,false",
    )
    p_shapes.set_defaults(func=cmd_sweep_shapes)

    p_ns = sub.add_parser("sweep-num-splits", help="Sweep FA4 num_splits for one shape.")
    p_ns.add_argument("--device", default="cuda:0")
    p_ns.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp8_e4m3", "fp8_e5m2"],
        help="Data type used for both Q and KV tensors in this microbench.",
    )
    p_ns.add_argument("--seed", type=int, default=0)

    p_ns.add_argument("--batch-size", type=int, default=1)
    p_ns.add_argument("--seqlen-k", type=int, default=4096)
    p_ns.add_argument("--num-heads", type=int, default=8)
    p_ns.add_argument("--num-kv-heads", type=int, default=1)
    p_ns.add_argument("--head-dim", type=int, default=128)
    p_ns.add_argument("--page-size", type=int, default=128)
    p_ns.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)

    p_ns.add_argument("--iters", type=int, default=200)
    p_ns.add_argument("--repeats", type=int, default=5)
    p_ns.add_argument("--warmup-iters", type=int, default=10)
    p_ns.add_argument(
        "--num-splits",
        default="1,2,4,8,16,32,0",
        help="Comma-separated list. Use 0 for auto.",
    )
    p_ns.add_argument(
        "--fa4-pack-gqa-single",
        default="auto",
        help="One of {auto,true,false}.",
    )
    p_ns.set_defaults(func=cmd_sweep_num_splits)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
