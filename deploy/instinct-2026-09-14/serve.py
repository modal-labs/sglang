"""
Serving metadata:
engine: sglang
base_model_repo_id: moonshotai/Kimi-K3
base_model_revision: 9f62e4e9fffbd0a83ddd60e1c209d828994b3569
model_family: kimi_k3
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import struct
import subprocess
import time
import zlib
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
MEM_FRACTION_STATIC = "0.915"
PREFILL_CUDA_GRAPH_MAX_BS = "4096"
PREFILL_CUDA_GRAPH_BS = "128 256 512 768 1024 1536 2048 3072 4096"
DECODE_CUDA_GRAPH_MAX_BS = "48"

# HiCache: host-memory (L2) KV cache tier, sized for a 60-minute session TTL.
# MLA host dedup keeps one copy of the (TP-replicated) target KV across the 8
# ranks instead of 8, so the host KV pool is an absolute size (GB, all ranks)
# and the rank-local Mamba/KDA state is sized separately as a ratio of the
# device Mamba pool. 140 GB KV + 13.5x Mamba => ~800 GB pinned host memory,
# within the 1 TiB container request (Modal's maximum is 1048576 MiB = 1 TiB).
HICACHE_KV_SIZE_GB = "140"
HICACHE_MAMBA_RATIO = "13.5"
HICACHE_WRITE_POLICY = "write_through"

SGLANG_BASE_IMAGE = "modalresearch/sglang:kimi-k3-cu13-20260806-b9e90a6d6"
SGLANG_COMMIT = "b9e90a6d6ef1859830c3b879cef999092975a41a"   # HEAD stays here
SGLANG_EFFECTIVE_COMMIT = "2c881e2ed528746312ec326fa89ee6e5e2169adf"  # JIT-cache salt (unchanged: same kernels/ABI)
RELEASE_REF = "dev/instinct/2026-09-15"
RELEASE_SHA = "fd8aff798ca487d72db3341a5057e84394196af5"  # dev head: + #58 session fixes, #67 KV-age metrics (off), #69/#72 evict_on_finish (off), #73 HiCache sizing, #74 write-stream hardening
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

BASE_RUNTIME_ENV = {
    "SYNC_TOKEN_IDS_ACROSS_TP": "1",
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
    # release/instinct/2026-09-14 (PR #22): all default-off, opted in here
    "SGLANG_TRTLLM_MLA_VERIFY_FUSED_KV_WRITE": "1",
    "SGLANG_TRTLLM_MLA_FUSED_CHUNK_KV_PACK": "1",
    "SGLANG_KIMI_ENCODE_FAST_PATH": "1",
    "SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS": "32000000",
    "SGLANG_K3_MM_USE_RENDERED_INPUT_IDS": "1",
    # dev/instinct/2026-09-15 #26: scheduler-side mm padding fast path (CONFIRMED same-box, image p50 -25..-33 ms)
    "SGLANG_K3_SCHED_MM_FASTPATH": "1",
    "SGLANG_K3_MM_STRIP_PROCESSOR_INPUT_IDS": "1",
}

PREBUILT_JIT_MODULE = "sgl_trtllm_gen_moe_fi_651799c8f7fd_4153db87ecc2"
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
    try:
        stop_current_container()
    finally:
        time.sleep(60)
        os._exit(1)

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
    """Copy the image-baked module into the volume cache if absent."""
    src = Path(PREBUILT_JIT_IMAGE_DIR) / PREBUILT_JIT_MODULE
    dst = Path(jit_cache_path) / "tvm-ffi" / PREBUILT_JIT_MODULE
    if any(dst.glob("*.so")):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Seeded prebuilt JIT module {PREBUILT_JIT_MODULE} into {dst}")


@app.server(
    include_source=True,
    image=serving_image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    volumes=server_volumes,
    min_containers=63,
    target_concurrency=TARGET_CONCURRENCY,
    scaledown_window=10 * MINUTES,
    scaleup_window=5*MINUTES,
    startup_timeout=3 * HOURS,
    port=PORT,
    unauthenticated=UNAUTHENTICATED,
    exit_grace_period=GRACEFUL_DRAIN_SECONDS + 60,
    routing_region="us-west",
    experimental_options={"override_eof_timeout": 1800, "kv_aware_routing": True},
)
class Server:

    @modal.enter()
    def startup(self) -> None:
        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            start_heartbeat_thread,
            warmup_chat_completions,
        )

        seed_prebuilt_jit(JIT_CACHE_PATH)
        check_dflash2_checkpoint(SPECULATIVE_DRAFT_MODEL_PATH)
        started = time.monotonic()

        print(
            "Kimi K3 runtime configuration: "
            f"model_revision={MODEL_REVISION!r}, "
            f"draft_path={SPECULATIVE_DRAFT_MODEL_PATH!r}, "
            f"server_args={SERVER_ARGS!r}"
        )

        self.endpoint = SGLangEndpoint(
            model_path=MODEL_PATH,
            worker_port=PORT,
            tp=TP_SIZE,
            speculative_model_path=SPECULATIVE_DRAFT_MODEL_PATH,
            extra_server_args=SERVER_ARGS,
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

        start_heartbeat_thread(
            self.endpoint.health_check,
            on_failure=terminate_unhealthy_container,
            poll_interval=10.0,
            max_consecutive_failures=6,
        )

        print(
            f"Kimi K3 TP8 DFlash is ready (release {RELEASE_SHA[:9]}). "
            f"endpoint={endpoint_elapsed:.2f}s cumulative, "
            f"warmup={warmup_elapsed:.2f}s cumulative."
        )

    @modal.exit()
    def stop(self) -> None:
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
