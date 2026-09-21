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
import tempfile
import threading
import time
import urllib.error
import urllib.request
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
RUNTIME_MODEL_PATH = "/tmp/kimi-k3-model"
DFLASH_VOLUME_NAME = "dflash_spec"
DFLASH_MOUNT_PATH = "/dflash"
SPECULATIVE_DRAFT_MODEL_PATH = f"{DFLASH_MOUNT_PATH}/k3-instinct-v5-epoch1"
LOAD_FORMAT = "fastsafetensors"
DRAFT_LOAD_FORMAT = "safetensors"

DRAFT_KV_CACHE_DTYPE = "bf16"
MEM_FRACTION_STATIC = "0.900"
PREFILL_CUDA_GRAPH_MAX_BS = "4096"
PREFILL_CUDA_GRAPH_BS = "128 256 512 768 1024 1536 2048 3072 4096"
DECODE_CUDA_GRAPH_MAX_BS = "48"

# HiCache: host-memory (L2) KV cache tier.
# Host pool size ~= HICACHE_RATIO x GPU KV pool (30.67 GB) x 8 TP ranks.
# Ratio 3 => ~735 GB of host memory, within the 1 TiB container request
# (Modal's maximum memory request is 1048576 MiB = 1 TiB).
HICACHE_RATIO = "3"
HICACHE_WRITE_POLICY = "write_through_selective"

SGLANG_BASE_IMAGE = "modalresearch/sglang:kimi-k3-cu13-20260806-b9e90a6d6"
SGLANG_COMMIT = "b9e90a6d6ef1859830c3b879cef999092975a41a"   # HEAD stays here
SGLANG_EFFECTIVE_COMMIT = "2c881e2ed528746312ec326fa89ee6e5e2169adf"  # JIT-cache salt (unchanged: same kernels/ABI)
RELEASE_REF = "release/instinct/2026-09-14"
RELEASE_SHA = "462ade71f00ee31c60a149bb6b29e77889dcbc91"  # code actually running (PR #22 head)
RELEASE_BUNDLE = Path(__file__).parent / "rel0914.bundle"
RELEASE_BUNDLE_IMAGE_PATH = "/tmp/rel0914.bundle"
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
    # release/instinct/2026-09-14 (PR #22): all default-off, opted in here
    "SGLANG_TRTLLM_MLA_VERIFY_FUSED_KV_WRITE": "1",
    "SGLANG_TRTLLM_MLA_FUSED_CHUNK_KV_PACK": "1",
    "SGLANG_KIMI_ENCODE_FAST_PATH": "1",
    "SGLANG_CHAT_SEGMENT_CACHE_MAX_CHARS": "32000000",
    "SGLANG_K3_MM_USE_RENDERED_INPUT_IDS": "1",
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
    "--hicache-ratio": HICACHE_RATIO,
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
    "--speculative-draft-model-quantization": "fp8",
    "--speculative-draft-fp8-activation-scheme": "static",
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

# Heartbeat probe on the real request path (chat template -> tokenizer ->
# scheduler -> 1 token). SGLang /health only runs a bare 1-token generate and
# stays green when requests stall before the scheduler.
PROBE_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}
PROBE_REQUEST_TIMEOUT_SECONDS = 30.0
PROBE_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def probe_chat_completions() -> str | None:
    """One real-path probe. None = healthy, str = failure reason.

    Any prompt HTTP answer counts as healthy: 503 is the bounded queue
    (--max-queued-requests) rejecting a busy-but-serving container, and a 4xx
    means the API layer is responsive. Only no-answer-in-time and 5xx count.
    """
    body = json.dumps(PROBE_PAYLOAD).encode("utf-8")
    req = urllib.request.Request(
        PROBE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            req, timeout=PROBE_REQUEST_TIMEOUT_SECONDS
        ) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 503 or 400 <= exc.code < 500:
            print(f"[probe] chat/completions answered {exc.code} (busy or 4xx)")
            return None
        return f"chat/completions returned status {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed = time.monotonic() - started
        return f"chat/completions probe failed after {elapsed:.1f}s: {type(exc).__name__}: {exc}"
    return None


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
            "for p in $(pgrep -f '[s]glang.launch_server') "
            "$(pgrep 'sglang::schedul'); do "
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


@app.server(
    include_source=True,
    image=serving_image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    volumes={
        HF_CACHE_PATH: hf_cache,
        JIT_CACHE_MOUNT_PATH: jit_cache,
        DFLASH_MOUNT_PATH: dflash_volume.with_mount_options(read_only=True),
    },
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
        _reset_container_state()
        model_path = prepare_model_snapshot()
        started = time.monotonic()

        print(
            "Kimi K3 runtime configuration: "
            f"model_revision={MODEL_REVISION!r}, "
            f"draft_path={SPECULATIVE_DRAFT_MODEL_PATH!r}, "
            f"server_args={SERVER_ARGS!r}"
        )

        self.endpoint = SGLangEndpoint(
            model_path=model_path,
            worker_port=PORT,
            tp=TP_SIZE,
            speculative_model_path=SPECULATIVE_DRAFT_MODEL_PATH,
            extra_server_args=SERVER_ARGS,
            health_timeout=3 * HOURS,
            health_poll_interval=10.0,
            health_request_timeout=30.0,
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

        def heartbeat() -> str | None:
            if not _container_running():
                return None
            # /health first: raises if the server process exited, and still
            # catches the all-rank engine freeze; then the real request path.
            return self.endpoint.health_check() or probe_chat_completions()

        start_heartbeat_thread(
            heartbeat,
            on_failure=lambda: (
                terminate_unhealthy_container()
                if _begin_heartbeat_termination()
                else None
            ),
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
        _set_container_stopping()
        if not hasattr(self, "endpoint"):
            return
        # Let the engine drain in-flight requests before endpoint.stop()'s
        # terminate_process (SIGTERM, then SIGKILL after 10s) cuts them off.
        # This terminate() starts the engine drain when no platform signal
        # reached it; the drain is self-bounded by
        # SGLANG_GRACEFUL_SHUTDOWN_TIMEOUT, so wait() normally returns early.
        proc = getattr(self.endpoint, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=GRACEFUL_DRAIN_SECONDS + 30)
            except subprocess.TimeoutExpired:
                pass
        self.endpoint.stop()
