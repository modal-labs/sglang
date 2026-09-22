"""
Serving metadata:
engine: sglang
base_model_repo_id: moonshotai/Kimi-K3
base_model_revision: 9f62e4e9fffbd0a83ddd60e1c209d828994b3569
model_family: kimi_k3
"""

from __future__ import annotations

import base64
import http.client
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import deque
from collections.abc import Callable
from pathlib import Path

import modal
from modal._utils.async_utils import synchronize_api
from modal_proto import api_pb2

MINUTES = 60
HOURS = 60 * MINUTES
PORT = 8000

MODEL_NAME = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
MODEL_PATH = MODEL_NAME
RUNTIME_MODEL_PATH = "/tmp/kimi-k3-model"
DFLASH_VOLUME_NAME = "dflash_spec"
DFLASH_MOUNT_PATH = "/dflash"
# DFlash2 draft (architectures: ["DFlash2DraftModel"]); the engine selects the
# DFlash2 path from the checkpoint config. The chosen step is copied into an
# immutable versioned dir on dflash_spec at ship time and pinned here.
# v1 rollback: f"{DFLASH_MOUNT_PATH}/k3-instinct-v5-epoch1", fp8 + static scheme.
DFLASH2_PINNED_STEP = "draft-epoch-2"  # final checkpoint of the run
DFLASH2_TARGET_CONFIG_SHA256 = (
    "2a5cb51c92f3b64e68f7670042e4d8cfff0acd4bcf1e3c5f7190d69609381de6"
)
DFLASH2_REQUIRED_FILES = ("config.json", "dflash_checkpoint.json", "model.safetensors")
# Evaluation override (read at deploy time, never set in prod): K3_DFLASH2_VOLUME
# mounts that volume read-only at DFLASH2_EVAL_MOUNT_PATH (optionally from
# K3_DFLASH2_VOLUME_ENV) and K3_DFLASH2_PATH points the draft at a dir inside
# the container. The completeness check below runs on whichever path is active.
DFLASH2_EVAL_VOLUME_NAME = os.environ.get("K3_DFLASH2_VOLUME")
DFLASH2_EVAL_VOLUME_ENV = os.environ.get("K3_DFLASH2_VOLUME_ENV")
DFLASH2_EVAL_MOUNT_PATH = "/dflash2-eval"
DFLASH2_TRAINING_SUBDIR = (
    "outputs/k3-instinct-v5-dflash2-hero-b8-30p2t-success-producer-v2/trainer-0"
)  # draft-epoch-2 lives under trainer-0; the draft-step-* checkpoints are under trainer-1
SPECULATIVE_DRAFT_MODEL_PATH = os.environ.get(
    "K3_DFLASH2_PATH",
    f"{DFLASH2_EVAL_MOUNT_PATH}/{DFLASH2_TRAINING_SUBDIR}/{DFLASH2_PINNED_STEP}"
    if DFLASH2_EVAL_VOLUME_NAME
    else f"{DFLASH_MOUNT_PATH}/k3-instinct-v5-dflash2/{DFLASH2_PINNED_STEP}",
)
DRAFT_QUANTIZATION = "unquant"
LOAD_FORMAT = "fastsafetensors"
DRAFT_LOAD_FORMAT = "safetensors"

DRAFT_KV_CACHE_DTYPE = "bf16"
MEM_FRACTION_STATIC = "0.900"
PREFILL_CUDA_GRAPH_MAX_BS = "4096"
PREFILL_CUDA_GRAPH_BS = "128 256 512 768 1024 1536 2048 3072 4096"
DECODE_CUDA_GRAPH_MAX_BS = "48"
K3_MIN_CONTAINERS = int(os.environ.get("K3_MIN_CONTAINERS", "63"))
K3_MAX_CONTAINERS = (
    int(os.environ["K3_MAX_CONTAINERS"]) if "K3_MAX_CONTAINERS" in os.environ else None
)

# HiCache: host-memory (L2) KV cache tier, sized for a 60-minute session TTL.
# MLA host dedup keeps one copy of the (TP-replicated) target KV across the 8
# ranks instead of 8, so the host KV pool is an absolute size (GB, all ranks)
# and the rank-local Mamba/KDA state is sized separately as a ratio of the
# device Mamba pool. The validated baseline is 140 GB KV + 13.5x Mamba =>
# ~800 GiB pinned host memory (SGLANG_HICACHE_HOST_BUDGET_GIB=800), which fits
# the 1 TiB container request.
#
# The request is only a scheduling floor. Once the container lands, the
# memcgroup limit is 10x the request and /proc/meminfo reports the physical
# host (confirmed 2026-09-21 on an 8xB300 AWS host: MemTotal 4.2 TB inside
# the container), and 8-GPU hosts admit no other tasks. plan_hicache_host_tier()
# therefore reads the host size at startup and scales the baseline plan
# (KV size, Mamba ratio and the engine's aggregate cap together, so the
# validated KV/Mamba split is kept) to fill it, leaving
# HICACHE_HOST_FILL_FRACTION headroom plus HICACHE_HOST_RESERVE_GIB for the
# engine's own RSS, residual checkpoint page cache and the host.
# Deploy-time env (forwarded through K3_ENV_OVERRIDES) can replace the
# baseline split; autosizing then scales the replaced baseline. With
# K3_HICACHE_AUTOSIZE=0 the three values are used exactly (experiment arms).
HICACHE_KV_SIZE_GB = os.environ.get("K3_HICACHE_KV_SIZE_GB", "140")  # baseline --hicache-size (GB, one dedup copy)
HICACHE_MAMBA_RATIO = os.environ.get("K3_HICACHE_MAMBA_RATIO", "13.5")  # baseline --hicache-mamba-ratio (per rank)
HICACHE_HOST_BUDGET_GIB = int(os.environ.get("K3_HICACHE_HOST_BUDGET_GIB", "800"))  # baseline SGLANG_HICACHE_HOST_BUDGET_GIB
HICACHE_WRITE_POLICY = "write_through"
# Autosize knobs. Deploy-time env, forwarded into the container through
# K3_ENV_OVERRIDES. K3_HICACHE_AUTOSIZE=0 pins the baseline plan.
HICACHE_AUTOSIZE = os.environ.get("K3_HICACHE_AUTOSIZE", "1") != "0"
HICACHE_HOST_FILL_FRACTION = float(
    os.environ.get("K3_HICACHE_HOST_FILL_FRACTION", "0.90")
)
HICACHE_HOST_RESERVE_GIB = int(os.environ.get("K3_HICACHE_HOST_RESERVE_GIB", "64"))
HICACHE_HOST_MAX_GIB = int(os.environ.get("K3_HICACHE_HOST_MAX_GIB", "0"))  # 0: no cap
# No 8-GPU host has this much RAM; a MemTotal at or above it means the
# sandbox reported the memcgroup limit (10x the request) instead of the
# host, and the plan falls back to the baseline.
HICACHE_HOST_IMPLAUSIBLE_GIB = 8 * 1024
# Container memory cgroup files: (directory, limit files, usage file), v2 then
# v1. /proc/meminfo is not bounded by memory.max (host-wide under runc,
# sentry-reported under gVisor), so the plan reads the limit directly, as
# upstream SGLang (#40135) and vLLM do.
CGROUP_MEMORY_FILES = (
    ("sys/fs/cgroup", ("memory.max", "memory.high"), "memory.current"),
    ("sys/fs/cgroup/memory", ("memory.limit_in_bytes",), "memory.usage_in_bytes"),
)
# cgroup v1 reports "unlimited" as a sentinel near 2**63.
CGROUP_UNLIMITED_BYTES = 1 << 62

SGLANG_BASE_IMAGE = "modalresearch/sglang:kimi-k3-cu13-20260806-b9e90a6d6"
SGLANG_COMMIT = "b9e90a6d6ef1859830c3b879cef999092975a41a"   # HEAD stays here
SGLANG_EFFECTIVE_COMMIT = "2c881e2ed528746312ec326fa89ee6e5e2169adf"  # JIT-cache salt (unchanged: same kernels/ABI)
RELEASE_REF = "release/instinct/2026-09-21"
RELEASE_SHA = os.environ.get(
    "K3_RELEASE_SHA",
    "0cbd91d4f3d0dca86315b20a60c3573d1895362d",
)  # release head = dev/instinct-rca-hotfixes/2026-09-20 @ 0cbd91d4f3: cc7b258e48 + #132 deploy hotfix + #138 KDA padded-row interval zeroing + #139 degenerate verify-row sanitizer/chain-sampler guards/radix-skip + #140 mm embedding-cache correctness + #143 reserved KV slot-0/page-0 guards, zero-on-alloc, allocator hygiene + #142 acceptance-collapse watchdog
RELEASE_BUNDLE = Path(__file__).parent / f"engine-{RELEASE_SHA[:9]}.bundle"  # untracked; regenerate per RELEASE_SHA (see DEPLOY.md)
RELEASE_BUNDLE_IMAGE_PATH = f"/tmp/{RELEASE_BUNDLE.name}"
RELEASE_PIN_REF = f"refs/deploy/{RELEASE_BUNDLE.stem}"  # a bundle only advertises named refs, so pin the SHA under one
RELEASE_BUNDLE_CMD = (
    f"git update-ref {RELEASE_PIN_REF} {RELEASE_SHA} && "
    f"git bundle create {RELEASE_BUNDLE} "
    f"{SGLANG_COMMIT}..{RELEASE_PIN_REF} {SGLANG_COMMIT}..origin/release/2026-09-14"
)


def verify_release_bundle(bundle: Path = RELEASE_BUNDLE, sha: str = RELEASE_SHA) -> None:
    """Deploy-time check (operator machine): the untracked bundle exists, is a valid git bundle
    and carries RELEASE_SHA. Raises with the exact regeneration command otherwise."""
    hint = f"regenerate it with:\n  {RELEASE_BUNDLE_CMD}"
    if not bundle.is_file():
        raise FileNotFoundError(f"{bundle} missing (bundles are not tracked in git); {hint}")
    v = subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=bundle.parent, capture_output=True, text=True)
    if v.returncode != 0:
        raise RuntimeError(f"`git bundle verify {bundle}` failed:\n{v.stderr.strip()}\n{hint}")
    heads = subprocess.run(["git", "bundle", "list-heads", str(bundle)], cwd=bundle.parent, capture_output=True, text=True, check=True).stdout
    if sha not in heads.split():
        raise RuntimeError(f"{bundle} does not contain RELEASE_SHA {sha}; heads:\n{heads.strip()}\n{hint}")


if modal.is_local():  # the bundle only exists on the operator machine, not inside the container
    verify_release_bundle()
SGLANG_SOURCE_PATH = "/sgl-workspace/sglang"
FLASHINFER_TARGET_VERSION = "0.6.16rc5"
FLASHINFER_EXTRA_INDEX = "https://flashinfer.ai/whl"
FLASHINFER_EXTRA_INDEX_CU130 = "https://flashinfer.ai/whl/cu130"
FASTSAFETENSORS_VERSION = "0.3.3"
AUTOINFERENCE_UTILS_VERSION = "0.2.3"

GPU = "B300:8"
TP_SIZE = 8
CPU = 16
MEMORY_MIB = 1024 * 1024  # 1 TiB: Modal's platform maximum

TARGET_CONCURRENCY = 6
UNAUTHENTICATED = False
# Drain window for container stops (scale-in, host reclaim, rolling deploys).
# One constant drives three aligned deadlines so they cannot drift:
#   t+120  sglang abandons the drain itself (SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT)
#   t+150  the exit hook stops waiting and falls through to the 10s kill path
#   t+180  Modal SIGKILLs the container (exit_grace_period)
GRACEFUL_DRAIN_SECONDS = 120

HF_CACHE_PATH = "/cache/huggingface"
JIT_CACHE_MOUNT_PATH = "/root/kimi-k3-jit-cache-volume"
JIT_CACHE_PATH = f"{JIT_CACHE_MOUNT_PATH}/kimi-k3-cu13-sm103-rc5-native-kepoch2-4153db87"

HF_CACHE_VOLUME_NAME = "huggingface-cache"
JIT_CACHE_VOLUME_NAME = "kimi-k3-b300-jit-cache"

hf_cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
jit_cache = modal.Volume.from_name(
    JIT_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
dflash_volume = modal.Volume.from_name(DFLASH_VOLUME_NAME)
server_volumes = {
    HF_CACHE_PATH: hf_cache,
    JIT_CACHE_MOUNT_PATH: jit_cache,
    DFLASH_MOUNT_PATH: dflash_volume.with_mount_options(read_only=True),
}
if DFLASH2_EVAL_VOLUME_NAME:
    server_volumes[DFLASH2_EVAL_MOUNT_PATH] = modal.Volume.from_name(
        DFLASH2_EVAL_VOLUME_NAME, environment_name=DFLASH2_EVAL_VOLUME_ENV
    ).with_mount_options(read_only=True)
# Dev/experiment endpoints only: mount a scratch volume (driver, workload,
# results) at /experiment. Unset in prod.
EXPERIMENT_VOLUME_NAME = os.environ.get("K3_EXPERIMENT_VOLUME", "")
EXPERIMENT_MOUNT_PATH = "/experiment"
if EXPERIMENT_VOLUME_NAME:
    server_volumes[EXPERIMENT_MOUNT_PATH] = modal.Volume.from_name(
        EXPERIMENT_VOLUME_NAME, create_if_missing=True
    )

K3_ENV_OVERRIDES = {
    key: os.environ[key]
    for key in (
        "K3_MIN_CONTAINERS",
        "K3_MAX_CONTAINERS",
        "K3_RELEASE_SHA",
        "K3_HICACHE_AUTOSIZE",
        "K3_HICACHE_HOST_FILL_FRACTION",
        "K3_HICACHE_HOST_RESERVE_GIB",
        "K3_HICACHE_HOST_MAX_GIB",
        "K3_HICACHE_KV_SIZE_GB",
        "K3_HICACHE_MAMBA_RATIO",
        "K3_HICACHE_HOST_BUDGET_GIB",
        "K3_EXPERIMENT_VOLUME",
    )
    if key in os.environ
}

BASE_RUNTIME_ENV = {
    "SYNC_TOKEN_IDS_ACROSS_TP": "1",
    # Engine-side abandon deadline for the drain started by stop(); without it
    # the engine defaults to 0 and never drains on container stop.
    "SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT": str(GRACEFUL_DRAIN_SECONDS),
    "SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES": "4294967296",
    "SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N": "256",
    "SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS": "8192",
    "KIMI_K3_DRAFT_KV_CACHE_DTYPE": DRAFT_KV_CACHE_DTYPE,
    "KIMI_K3_MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
    "KIMI_K3_PREFILL_CUDA_GRAPH_MAX_BS": PREFILL_CUDA_GRAPH_MAX_BS,
    "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(TP_SIZE)),
    "HF_HOME": HF_CACHE_PATH,
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "SGLANG_FASTSAFETENSORS_NOGDS": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_K3_AR_FUSION": "1",
    "SGLANG_TRTLLM_GEN_MOE_SOURCE": "flashinfer",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/nccl_trace_rank_",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
    "CUTE_DSL_ARCH": "sm_103a",
    "FLASH_ATTENTION_ARCH": "sm_103",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{JIT_CACHE_PATH}/flash-attention-cute-dsl",
    "CUTE_DSL_CACHE_DIR": f"{JIT_CACHE_PATH}/cute-dsl",
    "SGLANG_CUTE_AOT_CACHE_DIR": f"{JIT_CACHE_PATH}/cute-aot",
    "SGLANG_CUTE_AOT_ABI_SALT": f"{SGLANG_BASE_IMAGE}+{SGLANG_EFFECTIVE_COMMIT}",
    "SGLANG_EFFECTIVE_COMMIT": SGLANG_EFFECTIVE_COMMIT,
    "TVM_FFI_CACHE_DIR": f"{JIT_CACHE_PATH}/tvm-ffi",
    "SGLANG_CACHE_DIR": f"{JIT_CACHE_PATH}/sglang",
    "SGLANG_DG_CACHE_DIR": f"{JIT_CACHE_PATH}/deep-gemm",
    "FLASHINFER_WORKSPACE_BASE": f"{JIT_CACHE_PATH}/flashinfer-workspace",
    "TRITON_CACHE_DIR": f"{JIT_CACHE_PATH}/triton",
    "TRITON_CACHE_AUTOTUNING": "1",
    "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "1",
    "FLA_CACHE_RESULTS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:False",
    "TORCHINDUCTOR_CACHE_DIR": f"{JIT_CACHE_PATH}/inductor",
    "CUDA_CACHE_PATH": f"{JIT_CACHE_PATH}/cuda",
    "SGLANG_SSE_KEEPALIVE_INTERVAL": "1",
    "SGLANG_K3_TARGET_DENSE_FP8": "wide",
    "SGLANG_K3_TARGET_DENSE_FP8_REPRESENTATION": "tensor_static",
    "SGLANG_K3_TARGET_DENSE_FP8_MEMORY_DIAGNOSTICS": "1",
    "SGLANG_K3_TARGET_DENSE_FP8_RANGE_DIAGNOSTICS": "0",
    "SGLANG_K3_ATTN_RES_FP8_FUSION": "1",
    "SGLANG_VLM_MEDIA_URL_FETCH_ENABLED": "false",
    "SGLANG_OPENAI_MEDIA_URL_FETCH_ENABLED": "false",
    "SGLANG_RELEASE_SHA": RELEASE_SHA,
    **K3_ENV_OVERRIDES,
    # release/instinct/2026-09-14 (PR #22): all default-off, opted in here
    "SGLANG_TRTLLM_MLA_VERIFY_FUSED_KV_WRITE": "1",
    "SGLANG_TRTLLM_MLA_FUSED_CHUNK_KV_PACK": "1",
    "SGLANG_KIMI_ENCODE_FAST_PATH": "1",
    "SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS": "32000000",
    "SGLANG_K3_MM_USE_RENDERED_INPUT_IDS": "1",
    # dev/instinct/2026-09-15 #26: scheduler-side mm padding fast path (CONFIRMED same-box, image p50 -25..-33 ms)
    "SGLANG_K3_SCHED_MM_FASTPATH": "1",
    "SGLANG_K3_MM_STRIP_PROCESSOR_INPUT_IDS": "1",
    # dev/instinct/2026-09-15 #110: bit-identical top-p/top-k renorm across TP ranks
    "SGLANG_RENORM_DETERMINISTIC": "1",
    # dev/instinct/2026-09-15 #113: rank-0 authority D2H write handoff lock
    "SGLANG_ENABLE_HICACHE_ATOMIC_WRITE_HANDOFF": "1",
    # Engine aggregate host-pool cap for the baseline plan (engine default is
    # also 800); startup() raises it together with the pool sizes.
    "SGLANG_HICACHE_HOST_BUDGET_GIB": str(HICACHE_HOST_BUDGET_GIB),
    # dev/instinct/2026-09-15 #119: per-request prefill/decode/queue timings in usage/meta_info
    "SGLANG_ENABLE_REQUEST_METRICS": "1",
}

PREBUILT_JIT_MODULE = "sgl_trtllm_gen_moe_fi_651799c8f7fd_4153db87ecc2"
PREBUILT_JIT_STALE_SECONDS = 3600
PREBUILT_JIT_URL = (
    "https://github.com/modal-projects/flashinfer/releases/download/"
    f"jit-prebuilt-4153db87/{PREBUILT_JIT_MODULE}.tar.gz"
)
PREBUILT_JIT_SHA256 = (
    "6098bacaf89356ed014ee9f6cfd89ffa5de7ff3a9c7a9d3c50604e9b17045b31"
)
PREBUILT_JIT_IMAGE_DIR = "/opt/prebuilt-jit"

serving_image = (
    modal.Image.from_registry(SGLANG_BASE_IMAGE)
    .entrypoint([])
    .uv_pip_install(
        f"autoinference-utils=={AUTOINFERENCE_UTILS_VERSION}",
        f"fastsafetensors=={FASTSAFETENSORS_VERSION}",
    )
    .uv_pip_install(
        f"flashinfer-python=={FLASHINFER_TARGET_VERSION}",
        f"flashinfer-cubin=={FLASHINFER_TARGET_VERSION}",
        f"flashinfer-jit-cache=={FLASHINFER_TARGET_VERSION}",
        extra_index_url=FLASHINFER_EXTRA_INDEX_CU130,
        extra_options=(
            f"--extra-index-url {FLASHINFER_EXTRA_INDEX} "
            "--index-strategy unsafe-best-match"
        ),
    )
    .run_commands(
        f'test "$(git -C {SGLANG_SOURCE_PATH} rev-parse HEAD)" = "{SGLANG_COMMIT}"'
    )
    # --- release source: fork commits since the image's HEAD as a git bundle;
    #     checkout the release sha (it contains the old bump + PR31633)
    .add_local_file(RELEASE_BUNDLE, RELEASE_BUNDLE_IMAGE_PATH, copy=True)
    .run_commands(
        f'set -eu; '
        f'git -C {SGLANG_SOURCE_PATH} fetch -q {RELEASE_BUNDLE_IMAGE_PATH} {RELEASE_SHA}; '
        f'git -C {SGLANG_SOURCE_PATH} checkout -q {RELEASE_SHA}; '
        f'test "$(git -C {SGLANG_SOURCE_PATH} rev-parse HEAD)" = "{RELEASE_SHA}"; '
        f'test -z "$(git -C {SGLANG_SOURCE_PATH} status --porcelain --untracked-files=no)"; '
        f'rm -f {RELEASE_BUNDLE_IMAGE_PATH}; '
        f'python -c "import sglang; print(sglang.__file__)"'
    )
    .run_commands(
        "mkdir -p /opt/prebuilt-jit && cd /opt/prebuilt-jit && "
        f"curl -fsSL -o module.tar.gz {PREBUILT_JIT_URL} && "
        f'echo "{PREBUILT_JIT_SHA256}  module.tar.gz" | sha256sum -c - && '
        "tar xzf module.tar.gz && rm module.tar.gz"
    )
    .env(BASE_RUNTIME_ENV)
)

EXTRA_SERVER_ARGS = {
    "--trust-remote-code": "",
    "--load-format": LOAD_FORMAT,
    "--dist-timeout": "3600",
    "--context-length": "1048576",
    "--moe-runner-backend": "flashinfer_mxfp4",
    "--kv-cache-dtype": "fp8_e4m3",
    "--chunked-prefill-size": "16384",
    "--page-size": "64",
    "--mem-fraction-static": MEM_FRACTION_STATIC,
    "--schedule-policy": "openrouter_slo",
    "--slo-ttft-slope-ms-per-uncached-token": "1.0",
    "--slo-prefill-tokens-per-s": "14000",
    "--attention-backend": "trtllm_mla",
    "--prefill-attention-backend": "trtllm_mla",
    "--decode-attention-backend": "cutedsl_mla",
    "--mamba-ssm-dtype": "bfloat16",
    "--linear-attn-prefill-backend": "triton",
    "--linear-attn-decode-backend": "flashinfer",
    "--linear-attn-verify-backend": "nv_cutedsl",
    "--enable-linear-replayssm-spec": "",
    "--linear-replayssm-cache-len": "32",
    "--max-queued-requests": "16",
    "--max-running-requests": "12",
    "--max-mamba-cache-size": "130",
    "--mamba-max-states-per-path": "4",
    "--mamba-radix-cache-strategy": "extra_buffer_lazy",
    "--cuda-graph-max-bs-decode": DECODE_CUDA_GRAPH_MAX_BS,
    "--cuda-graph-bs-decode": "1 2 4 8 12 16 24 32 48",
    "--enable-cache-report": "",
    "--enable-hierarchical-cache": "",
    "--enable-mla-hicache-host-dedup": "",
    "--hicache-size": HICACHE_KV_SIZE_GB,
    "--hicache-mamba-ratio": HICACHE_MAMBA_RATIO,
    "--hicache-write-policy": HICACHE_WRITE_POLICY,
    "--speculative-algorithm": "DFLASH",
    "--speculative-attention-mode": "decode",
    "--speculative-draft-load-format": DRAFT_LOAD_FORMAT,
    "--speculative-num-steps": "1",
    "--speculative-num-draft-tokens": "8",
    "--speculative-dflash-block-size": "8",
    "--speculative-draft-window-size": "4096",
    "--speculative-eagle-topk": "1",
    "--speculative-draft-attention-backend": "trtllm_mha",
    "--speculative-draft-kv-cache-dtype": DRAFT_KV_CACHE_DTYPE,
    "--speculative-draft-model-quantization": DRAFT_QUANTIZATION,
    "--reasoning-parser": "kimi_k3",
    "--tool-call-parser": "kimi_k3",
    "--stream-response-default-include-usage": "",
    "--cuda-graph-backend-prefill": "breakable",
    "--cuda-graph-max-bs-prefill": PREFILL_CUDA_GRAPH_MAX_BS,
    "--cuda-graph-bs-prefill": PREFILL_CUDA_GRAPH_BS,
    "--enable-metrics": "",
    "--return-input-ids": "",
    "--return-output-ids": "",
}

SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--revision": MODEL_REVISION,
} | EXTRA_SERVER_ARGS

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with a short greeting."}],
    "max_tokens": 64,
    "temperature": 0.7,
}

WARMUP_IMAGE_WIDTH = 1280
WARMUP_IMAGE_HEIGHT = 800


def warmup_image_data_url(width: int, height: int) -> str:
    """Encode a synthetic RGB gradient PNG as an OpenAI image_url data URL."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    row = bytes(
        channel
        for x in range(width)
        for channel in (
            x * 255 // max(width - 1, 1),
            128,
            255 - x * 255 // max(width - 1, 1),
        )
    )
    scanlines = b"".join(b"\x00" + row for _ in range(height))
    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(scanlines, 9)),
            chunk(b"IEND", b""),
        )
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# First image request on a fresh container pays the mm-path JIT/autotune
# (25-80 s observed); take it in warmup instead of on a user request.
WARMUP_IMAGE_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this screenshot in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": warmup_image_data_url(
                            WARMUP_IMAGE_WIDTH, WARMUP_IMAGE_HEIGHT
                        )
                    },
                },
            ],
        }
    ],
    "max_tokens": 64,
    "temperature": 0.7,
}


async def _stop_current_container() -> None:
    from modal.client import _Client

    client = await _Client.from_env()
    await client.stub.ContainerStop(
        api_pb2.ContainerStopRequest(task_id=os.environ["MODAL_TASK_ID"])
    )


stop_current_container = synchronize_api(_stop_current_container)


# Container lifecycle state, guarded by the lock so the Modal exit hook and
# the heartbeat thread cannot both start teardown: RUNNING -> STOPPING marks
# a planned drain (exit hook); RUNNING -> HEARTBEAT_TERMINATING marks
# heartbeat-initiated teardown. The exit hook may also move
# HEARTBEAT_TERMINATING -> STOPPING: a planned drain takes precedence and the
# in-flight heartbeat teardown then yields to it.
_CONTAINER_STATE_LOCK = threading.Lock()
_CONTAINER_STATE = "running"


def _reset_container_state() -> None:
    global _CONTAINER_STATE
    with _CONTAINER_STATE_LOCK:
        _CONTAINER_STATE = "running"


def _container_running() -> bool:
    with _CONTAINER_STATE_LOCK:
        return _CONTAINER_STATE == "running"


def _container_is_draining() -> bool:
    with _CONTAINER_STATE_LOCK:
        return _CONTAINER_STATE == "stopping"


def _set_container_stopping() -> None:
    global _CONTAINER_STATE
    with _CONTAINER_STATE_LOCK:
        _CONTAINER_STATE = "stopping"


def _begin_heartbeat_termination() -> bool:
    """Move RUNNING -> HEARTBEAT_TERMINATING exactly once."""
    global _CONTAINER_STATE
    with _CONTAINER_STATE_LOCK:
        if _CONTAINER_STATE != "running":
            return False
        _CONTAINER_STATE = "heartbeat_terminating"
        return True


def _exit_unless_draining(reason: str) -> None:
    """Force-exit the process unless a planned drain has taken over.

    The lock is held across os._exit so the check and the exit are atomic:
    the exit hook has either already moved the state to STOPPING (we yield
    and let it finish the drain) or is still blocked in
    _set_container_stopping() and has not started draining anything.
    """
    with _CONTAINER_STATE_LOCK:
        if _CONTAINER_STATE == "stopping":
            print(f"planned drain started; skipping forced exit ({reason})")
            return
        os._exit(1)


def terminate_unhealthy_container() -> None:
    """Collect bounded forensics and stop an unhealthy production container."""
    try:
        subprocess.run(
            "for p in $(pgrep 'sglang::schedul'); do "
            "py-spy dump --pid $p --nonblocking; done; "
            "nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv",
            shell=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"forensics dump failed: {error!r}")
    # The forensics dump may run for up to 60 s; a planned drain can have
    # started meanwhile. Yield to it instead of racing ContainerStop and the
    # forced exit against the drain.
    if _container_is_draining():
        print("planned drain started during forensics; skipping ContainerStop")
        return
    try:
        stop_current_container()
    except modal.exception.ClientClosed:
        # The worker already tore down the client: it is draining this
        # container itself, so exit now instead of waiting for it.
        _exit_unless_draining("ClientClosed")
        return
    except Exception as error:
        print(f"ContainerStop failed: {error!r}")
    # Give the platform's stop a minute to land, yielding as soon as the exit
    # hook begins a planned drain.
    for _ in range(60):
        if _container_is_draining():
            print("planned drain started after ContainerStop; skipping forced exit")
            return
        time.sleep(1)
    _exit_unless_draining("ContainerStop timeout")


# Sustained acceptance collapse can leave /health passing while a replica
# generates only its bonus token at every verification step. Measure existing
# completed-request counters over a trailing window, so idle replicas and
# stale acceptance gauges cannot produce a verdict. Keep the calibrated
# five-minute window and thirty ten-second failing polls before retirement.
SPEC_ACCEPT_METRICS_URL = f"http://127.0.0.1:{PORT}/metrics"
SPEC_ACCEPT_WINDOW_SECONDS = 5 * MINUTES
SPEC_ACCEPT_COLLAPSE_THRESHOLD = 1.5
SPEC_ACCEPT_MIN_VERIFY_CALLS = 200
SPEC_ACCEPT_POLL_SECONDS = 10.0
SPEC_ACCEPT_SUSTAINED_POLLS = 30
# Two consecutive counter samples farther apart than this break window
# continuity: the next successful read discards the samples before the gap,
# so a delayed poll or a /metrics outage can extend the effective window by
# at most this tolerance instead of by the whole outage.
SPEC_ACCEPT_MAX_SAMPLE_GAP_SECONDS = 2 * SPEC_ACCEPT_POLL_SECONDS
# Counters read from /metrics, in sample order. Names are pinned to the
# TokenizerMetricsCollector definitions by the deploy watchdog unit test.
_SPEC_ACCEPT_COUNTERS = (
    "generation_tokens_total",
    "spec_verify_calls_total",
)
_SPEC_ACCEPT_COUNTER_RE = re.compile(
    r"^sglang:(" + "|".join(_SPEC_ACCEPT_COUNTERS) + r")(?:\{[^}]*\})?\s+(\S+)",
    re.MULTILINE,
)


class SpecAcceptWatchdog:
    """Per-poll verdict on the trailing completed-request acceptance ratio."""

    def __init__(
        self,
        metrics_url: str = SPEC_ACCEPT_METRICS_URL,
        *,
        window_seconds: float = SPEC_ACCEPT_WINDOW_SECONDS,
        collapse_threshold: float = SPEC_ACCEPT_COLLAPSE_THRESHOLD,
        min_verify_calls: float = SPEC_ACCEPT_MIN_VERIFY_CALLS,
        max_sample_gap: float = SPEC_ACCEPT_MAX_SAMPLE_GAP_SECONDS,
        request_timeout: float = 5.0,
        read_metrics: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.metrics_url = metrics_url
        self.window_seconds = window_seconds
        self.collapse_threshold = collapse_threshold
        self.min_verify_calls = min_verify_calls
        self.max_sample_gap = max_sample_gap
        self.request_timeout = request_timeout
        self._read_metrics = read_metrics or self._fetch_metrics
        self._clock = clock
        # (monotonic time, *_SPEC_ACCEPT_COUNTERS totals)
        self._samples: deque[tuple[float, tuple[float, ...]]] = deque()

    def _fetch_metrics(self) -> str:
        with urllib.request.urlopen(
            self.metrics_url, timeout=self.request_timeout
        ) as response:
            return response.read().decode("utf-8", errors="replace")

    def _read_counters(self) -> tuple[float, ...]:
        """Totals of `_SPEC_ACCEPT_COUNTERS`, summed across label sets."""
        totals = dict.fromkeys(_SPEC_ACCEPT_COUNTERS, 0.0)
        seen = set()
        for name, value in _SPEC_ACCEPT_COUNTER_RE.findall(self._read_metrics()):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}: {value}")
            totals[name] += value
            seen.add(name)
        missing = set(_SPEC_ACCEPT_COUNTERS) - seen
        if missing:
            raise ValueError(f"missing acceptance counters: {sorted(missing)}")
        return tuple(totals[name] for name in _SPEC_ACCEPT_COUNTERS)

    def check(self) -> str | None:
        """None while healthy or without a verdict; an error string on collapse."""
        try:
            counters = self._read_counters()
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            # Unknown data is not a low-acceptance verdict. Invalid or absent
            # series cannot be bridged as zero; transport gaps use the normal
            # sample-gap tolerance on the next successful read.
            if isinstance(error, ValueError):
                self._samples.clear()
            print(f"[spec-accept] metrics unavailable; no verdict: {error!r}", flush=True)
            return None
        now = self._clock()
        if self._samples and now - self._samples[-1][0] > self.max_sample_gap:
            # A poll delayed past the gap tolerance leaves the same hole an
            # outage does; samples before the gap cannot contribute to a
            # trailing window.
            self._samples.clear()
        if self._samples and any(
            c < last for c, last in zip(counters, self._samples[-1][1])
        ):
            self._samples.clear()
        self._samples.append((now, counters))
        while len(self._samples) > 1 and (
            self._samples[1][0] <= now - self.window_seconds
        ):
            self._samples.popleft()
        first_time, first = self._samples[0]
        elapsed = now - first_time
        if elapsed < self.window_seconds:
            return None
        gen, verify_calls = (
            c - f for c, f in zip(counters, first)
        )
        if verify_calls >= self.min_verify_calls:
            accept_length = gen / verify_calls
            if accept_length <= self.collapse_threshold:
                return (
                    f"spec accept collapse: {accept_length:.2f} tokens/verify over "
                    f"{elapsed:.0f}s ({gen:.0f} tokens, "
                    f"{verify_calls:.0f} verify calls) <= {self.collapse_threshold}"
                )
        return None


# Kimi's pinned encoder interprets this reserved token in every string,
# including tool output and reasoning. Only typed image parts may consume an
# image prompt. MODEL_REVISION pins the source; the exact function match below
# prevents a partial or stale edit. (Same rewrite as autoinference's
# deployments/kimi_k3/prod_serve.py.)
_KIMI_K3_VULNERABLE_APPEND_TEXT = """def _append_text(
    segments: list[EncodeSegment],
    text: Any,
    image_state: _ImagePromptState,
) -> None:
    text = str(text)
    if text == "":
        return
    if image_state.image_prompts is None or IMAGE_PLACEHOLDER not in text:
        segments.extend(_text(text))
        return

    parts = text.split(IMAGE_PLACEHOLDER)
    for i, part in enumerate(parts):
        segments.extend(_text(part))
        if i < len(parts) - 1:
            segments.extend(_segment(image_state.next_prompt(),
                                     allow_special=True))
"""
_KIMI_K3_SAFE_APPEND_TEXT = """def _append_text(
    segments: list[EncodeSegment],
    text: Any,
    image_state: _ImagePromptState,
) -> None:
    # Text is always untrusted data. Image prompts are emitted only by the
    # typed image/image_url branch in _render_content_segments.
    segments.extend(_text(text))
"""


def _rewrite_kimi_k3_encoding(source: bytes) -> bytes:
    text = source.decode("utf-8")
    if text.count(_KIMI_K3_VULNERABLE_APPEND_TEXT) != 1:
        raise RuntimeError(
            "Kimi K3 encoder does not contain the reviewed image-placeholder block"
        )
    return text.replace(
        _KIMI_K3_VULNERABLE_APPEND_TEXT,
        _KIMI_K3_SAFE_APPEND_TEXT,
    ).encode("utf-8")


def prepare_model_snapshot() -> str:
    """Materialize the pinned snapshot and patch its encoder.

    The stock encoding_k3.py splits any string -- tool output, reasoning, user
    text -- on the image placeholder and injects image prompt segments. With
    exactly one prompt per typed image, a stray placeholder either crashes
    encoding (prompt exhaustion) or misbinds the image. The patched copy makes
    text always text; typed image/image_url parts emit prompts. Weights are
    symlinked from the HF cache; only small files are copied, so the shared
    cache is never mutated.
    """
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(MODEL_NAME, revision=MODEL_REVISION))
    required = [
        source / "config.json",
        source / "encoding_k3.py",
        source / "kimi_k3_processor.py",
        source / "media_utils.py",
        source / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Kimi K3 snapshot is incomplete or mounted from the wrong cache; "
            f"missing: {', '.join(missing)}"
        )

    destination = Path(RUNTIME_MODEL_PATH)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    materialized_files = 0
    for source_path in source.rglob("*"):
        runtime_path = destination / source_path.relative_to(source)
        if source_path.is_dir():
            runtime_path.mkdir(exist_ok=True)
        elif source_path.name.endswith(".safetensors"):
            runtime_path.symlink_to(source_path)
        else:
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, runtime_path, follow_symlinks=True)
            materialized_files += 1

    custom_code = list(destination.glob("*.py"))
    if not custom_code or any(path.is_symlink() for path in custom_code):
        raise RuntimeError("Kimi K3 custom model code was not materialized")

    encoding_path = destination / "encoding_k3.py"
    encoding_path.write_bytes(_rewrite_kimi_k3_encoding(encoding_path.read_bytes()))
    print(
        f"Prepared {destination} with {materialized_files} local files; "
        "patched encoding_k3.py for type-safe image placeholders"
    )
    return str(destination)

app = modal.App(name="kimi-k3-fast")


def check_dflash2_checkpoint(draft_path: str) -> None:
    """Fail fast if the pinned DFlash2 checkpoint dir is incomplete or rotated away."""
    if "<PINNED>" in draft_path:
        raise RuntimeError(
            f"DFlash2 draft step not pinned: {draft_path} (fill in DFLASH2_PINNED_STEP)"
        )
    root = Path(draft_path)
    if not root.is_dir():
        raise RuntimeError(
            f"DFlash2 draft dir missing: {draft_path} (rotated away or volume not mounted)"
        )
    missing = [name for name in DFLASH2_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"DFlash2 draft dir {draft_path} incomplete, missing {missing}")
    weights = root / "model.safetensors"
    with weights.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    expected_size = 8 + header_len + max(
        entry["data_offsets"][1]
        for name, entry in header.items()
        if name != "__metadata__"
    )
    actual_size = weights.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"DFlash2 model.safetensors truncated: {actual_size} B on disk, "
            f"header declares {expected_size} B"
        )
    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise RuntimeError(
            f"DFlash2 draft config.json architectures={config.get('architectures')!r}"
        )
    meta = json.loads((root / "dflash_checkpoint.json").read_text())
    sha = meta.get("target_config_sha256")
    if sha != DFLASH2_TARGET_CONFIG_SHA256:
        raise RuntimeError(
            f"DFlash2 target_config_sha256 {sha!r} != pinned {DFLASH2_TARGET_CONFIG_SHA256!r}"
        )
    print(f"DFlash2 draft checkpoint OK: {draft_path} (target_config_sha256={sha[:12]})")


def seed_prebuilt_jit(jit_cache_path: str) -> None:
    """Copy the image-baked module into the volume cache if absent.

    Staged into a sibling temp dir and published with one rename so a
    concurrent cold boot never sees a partially copied module. Leftover
    ``.<module>.*`` dirs (a seeder killed mid-copy) are reclaimed once
    their inode change time is older than any plausible copy."""
    src = Path(PREBUILT_JIT_IMAGE_DIR) / PREBUILT_JIT_MODULE
    dst = Path(jit_cache_path) / "tvm-ffi" / PREBUILT_JIT_MODULE
    stale_before = time.time() - PREBUILT_JIT_STALE_SECONDS
    for leftover in dst.parent.glob(f".{PREBUILT_JIT_MODULE}.*"):
        try:
            st = leftover.stat()
            if max(st.st_mtime, st.st_ctime) < stale_before:
                shutil.rmtree(leftover, ignore_errors=True)
        except OSError:
            pass
    if any(dst.glob("*.so")):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{PREBUILT_JIT_MODULE}.", dir=dst.parent)
    )
    quarantine = None
    try:
        shutil.copytree(src, staging, dirs_exist_ok=True)
        try:
            os.rename(staging, dst)
        except OSError:
            if any(dst.glob("*.so")):
                raise
            # An interrupted seed left a partial module (no .so): move it
            # aside and publish over it.
            quarantine = Path(
                tempfile.mkdtemp(prefix=f".{PREBUILT_JIT_MODULE}.stale.", dir=dst.parent)
            )
            os.rename(dst, quarantine / "partial")
            os.rename(staging, dst)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if any(dst.glob("*.so")):
            print(f"Prebuilt JIT module {PREBUILT_JIT_MODULE} already seeded at {dst}")
            return
        raise
    finally:
        if quarantine is not None:
            shutil.rmtree(quarantine, ignore_errors=True)
    print(f"Seeded prebuilt JIT module {PREBUILT_JIT_MODULE} into {dst}")


def export_deployment_identity() -> None:
    """Expose the running app's id to the engine as ``MODAL_APP_ID``.

    The engine's per-request metrics (SGLANG_ENABLE_REQUEST_METRICS) report
    ``deployment_id`` from ``MODAL_APP_ID`` and ``replica_id`` from
    ``MODAL_TASK_ID``. The container runtime only sets the latter; the app id is
    known to the SDK once the container has initialized, so it is exported here
    before the engine subprocess inherits the environment.
    """
    if os.environ.get("MODAL_APP_ID"):
        return
    app_id = app.app_id
    if app_id is None:
        print("MODAL_APP_ID not exported: app id unavailable in this container")
        return
    os.environ["MODAL_APP_ID"] = app_id
    print(f"Exported MODAL_APP_ID={app_id}")


def _read_meminfo_bytes(path: str = "/proc/meminfo") -> dict[str, int]:
    """Parse /proc/meminfo into bytes (the kernel reports kB)."""
    values: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            key, sep, rest = line.partition(":")
            parts = rest.split()
            if not sep or not parts or not parts[0].isdigit():
                continue
            unit = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
            values[key.strip()] = int(parts[0]) * unit
    return values


def _read_int_file(path: str) -> int | None:
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):  # missing file, or "max" (no limit)
        return None


def _read_cgroup_headroom_bytes(
    root: str = "/", fallback_usage: int = 0
) -> tuple[int, int, bool] | None:
    """Bytes this container may still allocate under its memory cgroup.

    None when no finite limit is visible (no cgroupfs in the sandbox, or the
    limit is "max"). Otherwise ``(headroom, limit, usage_known)``: the tightest
    of the limit files, and that limit minus current usage. The limit, not the
    headroom, says whether the cgroup is a credible single-host bound. If the
    usage file cannot be read, ``fallback_usage`` (the caller's own accounting)
    is subtracted instead of assuming zero, which would overstate the headroom.
    """
    for directory, limit_names, usage_name in CGROUP_MEMORY_FILES:
        limits = [
            value
            for name in limit_names
            if (value := _read_int_file(os.path.join(root, directory, name))) is not None
            and value < CGROUP_UNLIMITED_BYTES
        ]
        if limits:
            usage = _read_int_file(os.path.join(root, directory, usage_name))
            usage_known = usage is not None
            if not usage_known:
                usage = fallback_usage
            limit = min(limits)
            return max(0, limit - usage), limit, usage_known
    return None


def _host_identity() -> str:
    """Cloud/region/CPU count, so hit rates can be compared per 8xB300 SKU."""
    cloud = os.environ.get("MODAL_CLOUD_PROVIDER", "?").removeprefix("CLOUD_PROVIDER_").lower()
    region = os.environ.get("MODAL_REGION", "?")
    return f"cloud={cloud} region={region} cpus={os.cpu_count()}"


def plan_hicache_host_tier(
    meminfo: dict[str, int] | None = None,
    cgroup_root: str = "/",
) -> tuple[dict[str, str], dict[str, str]]:
    """Size the HiCache host tier to the physical host this container landed on.

    Runs once per container, before the engine starts. Returns
    ``(server_arg_overrides, env_overrides)``: ``--hicache-size`` and
    ``--hicache-mamba-ratio`` scaled from the validated baseline by
    ``usable / HICACHE_HOST_BUDGET_GIB``, and ``SGLANG_HICACHE_HOST_BUDGET_GIB``
    set to the usable amount so the engine's aggregate preflight enforces the
    same number. ``usable`` is
    ``min(MemTotal, MemAvailable, cgroup headroom) * HICACHE_HOST_FILL_FRACTION
    - HICACHE_HOST_RESERVE_GIB``, optionally capped by HICACHE_HOST_MAX_GIB.
    A host smaller than the baseline scales the plan down rather than starting
    a baseline the container cannot hold; a plan too small to express
    (``--hicache-size`` < 1 GB or ``--hicache-mamba-ratio`` < 1.0) fails startup.
    Falls back to the baseline when autosizing is off, /proc/meminfo is
    unreadable, or neither MemTotal nor the cgroup gives a plausible
    single-host size.
    """
    gib = 1024**3
    baseline_args = {
        "--hicache-size": HICACHE_KV_SIZE_GB,
        "--hicache-mamba-ratio": HICACHE_MAMBA_RATIO,
    }
    baseline_env = {"SGLANG_HICACHE_HOST_BUDGET_GIB": str(HICACHE_HOST_BUDGET_GIB)}
    baseline = (
        f"--hicache-size {HICACHE_KV_SIZE_GB} --hicache-mamba-ratio "
        f"{HICACHE_MAMBA_RATIO} SGLANG_HICACHE_HOST_BUDGET_GIB={HICACHE_HOST_BUDGET_GIB}"
    )
    if not HICACHE_AUTOSIZE:
        print(f"HiCache host tier: autosize off (K3_HICACHE_AUTOSIZE=0), baseline {baseline}")
        return baseline_args, baseline_env
    try:
        if meminfo is None:
            meminfo = _read_meminfo_bytes()
        total = meminfo["MemTotal"]
        available = meminfo.get("MemAvailable", total)
    except (OSError, KeyError, ValueError) as error:
        print(f"HiCache host tier: /proc/meminfo unreadable ({error!r}), baseline {baseline}")
        return baseline_args, baseline_env
    total_gib = total / gib
    available_gib = available / gib
    identity = _host_identity()
    # Read the cgroup before judging plausibility: a bogus MemTotal can still
    # be sized from a real container limit. Unreadable usage falls back to the
    # sandbox's own accounting (MemTotal - MemAvailable), never to zero.
    cgroup = _read_cgroup_headroom_bytes(cgroup_root, fallback_usage=max(0, total - available))
    cgroup_gib = None if cgroup is None else cgroup[0] / gib
    cgroup_limit_gib = None if cgroup is None else cgroup[1] / gib
    cgroup_text = "none"
    if cgroup is not None:
        cgroup_text = f"{cgroup_gib:.0f} GiB of {cgroup_limit_gib:.0f} GiB limit" + (
            "" if cgroup[2] else " (usage unreadable; meminfo used)"
        )
    meminfo_plausible = total_gib < HICACHE_HOST_IMPLAUSIBLE_GIB
    # Plausibility is judged on the limit: usage can pull the headroom of an
    # implausibly large limit below the threshold without making it a host bound.
    if not meminfo_plausible and (
        cgroup_limit_gib is None or cgroup_limit_gib >= HICACHE_HOST_IMPLAUSIBLE_GIB
    ):
        print(
            f"HiCache host tier: MemTotal={total_gib:.0f} GiB is not a single host "
            f"(>= {HICACHE_HOST_IMPLAUSIBLE_GIB} GiB, memcgroup limit reported?) and "
            f"cgroup_headroom={cgroup_text} gives no single-host bound, "
            f"{identity}, baseline {baseline}"
        )
        return baseline_args, baseline_env
    host_gib = min(total_gib, available_gib) if meminfo_plausible else math.inf
    if cgroup_gib is not None:
        host_gib = min(host_gib, cgroup_gib)
    usable_gib = host_gib * HICACHE_HOST_FILL_FRACTION - HICACHE_HOST_RESERVE_GIB
    if HICACHE_HOST_MAX_GIB > 0:
        usable_gib = min(usable_gib, HICACHE_HOST_MAX_GIB)
    memory = (
        f"MemTotal={total_gib:.0f} GiB, MemAvailable={available_gib:.0f} GiB, "
        f"cgroup_headroom={cgroup_text}"
    )
    scale = max(usable_gib, 0.0) / HICACHE_HOST_BUDGET_GIB
    kv_size_gb = int(float(HICACHE_KV_SIZE_GB) * scale)
    mamba_ratio = math.floor(float(HICACHE_MAMBA_RATIO) * scale * 10) / 10
    budget_gib = int(usable_gib)
    if kv_size_gb < 1 or mamba_ratio < 1.0:
        # The engine rejects a zero size/ratio, and a host Mamba pool smaller
        # than the device pool is not a useful tier.
        min_usable_gib = HICACHE_HOST_BUDGET_GIB * max(
            1 / float(HICACHE_KV_SIZE_GB), 1 / float(HICACHE_MAMBA_RATIO)
        )
        raise RuntimeError(
            f"HiCache host tier: usable {usable_gib:.1f} GiB is below the minimum "
            f"{min_usable_gib:.0f} GiB for this baseline (plan would be --hicache-size "
            f"{kv_size_gb} --hicache-mamba-ratio {mamba_ratio}; {memory}, "
            f"fill={HICACHE_HOST_FILL_FRACTION}, reserve={HICACHE_HOST_RESERVE_GIB} GiB, "
            f"cap={HICACHE_HOST_MAX_GIB or 'none'}, {identity}); set K3_HICACHE_AUTOSIZE=0, "
            "raise K3_HICACHE_HOST_MAX_GIB or lower the reserve."
        )
    print(
        f"HiCache host tier plan: {identity}, {memory}, "
        f"fill={HICACHE_HOST_FILL_FRACTION}, reserve={HICACHE_HOST_RESERVE_GIB} GiB, "
        f"cap={HICACHE_HOST_MAX_GIB or 'none'} -> usable={usable_gib:.0f} GiB, "
        f"scale={scale:.2f}x baseline: --hicache-size {kv_size_gb} "
        f"--hicache-mamba-ratio {mamba_ratio} SGLANG_HICACHE_HOST_BUDGET_GIB={budget_gib}"
    )
    return (
        {"--hicache-size": str(kv_size_gb), "--hicache-mamba-ratio": f"{mamba_ratio:g}"},
        {"SGLANG_HICACHE_HOST_BUDGET_GIB": str(budget_gib)},
    )


SERVER_KWARGS = {
    "include_source": True,
    "image": serving_image,
    "gpu": GPU,
    "cpu": CPU,
    "memory": MEMORY_MIB,
    "volumes": server_volumes,
    "min_containers": K3_MIN_CONTAINERS,
    "target_concurrency": TARGET_CONCURRENCY,
    "scaledown_window": 10 * MINUTES,
    "scaleup_window": 5*MINUTES,
    "startup_timeout": 3 * HOURS,
    "port": PORT,
    "unauthenticated": UNAUTHENTICATED,
    "exit_grace_period": GRACEFUL_DRAIN_SECONDS + 60,
    "routing_region": "us-west",
    "experimental_options": {"override_eof_timeout": 1800, "kv_aware_routing": True},
}
if K3_MAX_CONTAINERS is not None:
    SERVER_KWARGS["max_containers"] = K3_MAX_CONTAINERS


@app.server(**SERVER_KWARGS)
class Server:

    @modal.enter()
    def startup(self) -> None:
        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            start_heartbeat_thread,
            warmup_chat_completions,
        )

        # First thing after placement: size the host tier to this host, before
        # anything touches the engine. The env override is inherited by the
        # engine subprocess started by SGLangEndpoint below.
        hicache_args, hicache_env = plan_hicache_host_tier()
        os.environ.update(hicache_env)
        server_args = SERVER_ARGS | hicache_args

        seed_prebuilt_jit(JIT_CACHE_PATH)
        check_dflash2_checkpoint(SPECULATIVE_DRAFT_MODEL_PATH)
        export_deployment_identity()
        _reset_container_state()
        model_path = prepare_model_snapshot()
        started = time.monotonic()

        print(
            "Kimi K3 runtime configuration: "
            f"model_revision={MODEL_REVISION!r}, "
            f"draft_path={SPECULATIVE_DRAFT_MODEL_PATH!r}, "
            f"server_args={server_args!r}"
        )

        self.endpoint = SGLangEndpoint(
            model_path=model_path,
            worker_port=PORT,
            tp=TP_SIZE,
            speculative_model_path=SPECULATIVE_DRAFT_MODEL_PATH,
            extra_server_args=server_args,
            health_timeout=3 * HOURS,
            health_poll_interval=10.0,
            health_request_timeout=30.0,
            log_requests_level=-1,
        )
        self.endpoint.start()
        endpoint_elapsed = time.monotonic() - started

        warmup_chat_completions(
            port=PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=1,
            request_timeout=30 * MINUTES,
        )
        warmup_chat_completions(
            port=PORT,
            payload=WARMUP_IMAGE_PAYLOAD,
            successful_requests=1,
            request_timeout=30 * MINUTES,
        )
        warmup_elapsed = time.monotonic() - started

        def _on_heartbeat_failure() -> None:
            if _begin_heartbeat_termination():
                terminate_unhealthy_container()

        start_heartbeat_thread(
            lambda: self.endpoint.health_check() if _container_running() else None,
            on_failure=_on_heartbeat_failure,
            poll_interval=10.0,
            max_consecutive_failures=6,
        )
        spec_accept_watchdog = SpecAcceptWatchdog()
        start_heartbeat_thread(
            lambda: spec_accept_watchdog.check() if _container_running() else None,
            on_failure=_on_heartbeat_failure,
            poll_interval=SPEC_ACCEPT_POLL_SECONDS,
            max_consecutive_failures=SPEC_ACCEPT_SUSTAINED_POLLS,
        )

        print(
            f"Kimi K3 TP8 DFlash is ready (release {RELEASE_SHA[:9]}). "
            f"endpoint={endpoint_elapsed:.2f}s cumulative, "
            f"warmup={warmup_elapsed:.2f}s cumulative."
        )

    @modal.exit()
    def stop(self) -> None:
        _set_container_stopping()
        if not hasattr(self, "endpoint"):
            return
        # Let the engine drain in-flight requests before endpoint.stop()'s
        # terminate_process (SIGTERM, then SIGKILL after 10s) cuts them off.
        # Under the fork's signal-identity dedup this SIGTERM is NOT a no-op
        # when the platform's stop bundle already reached the engine: a
        # repeated SIGTERM escalates the active drain to force-exit. That is
        # the intended behavior here — this hook only runs once Modal has
        # finished waiting (graceful path: in-flight streams already hit
        # zero; blunt path: streams already severed platform-side), so
        # escalation speeds teardown without cutting anything still alive.
        # If no signal reached the engine (paths that signal only the
        # container main process), this terminate() is what starts the
        # drain, self-bounded via SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT, so
        # wait() normally returns early.
        proc = getattr(self.endpoint, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=GRACEFUL_DRAIN_SECONDS + 30)
            except subprocess.TimeoutExpired:
                pass
        self.endpoint.stop()
