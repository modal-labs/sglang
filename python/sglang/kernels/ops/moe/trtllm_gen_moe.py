"""TRT-LLM-gen fused MoE (SiTU) compiled through the sglang JIT system.

Builds the trtllm-gen fused-MoE host/runner sources with sglang's own
tvm-ffi ``load_jit`` from a **self-contained cubin pool**
(``SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL``): a downloadable directory holding

  * the prebuilt SiTU cubins (``local/``) + ``config.json`` +
    ``flashinferMetaInfo.h``,
  * the flat batched-gemm ABI headers (staged into a
    ``trtllmGen_bmm_export/``-shaped include tree at build time),
  * an ``overlay/`` with only the sources/headers that differ from the
    public ``flashinfer`` pip package.

Every unmodified source and the CUTLASS headers come from the installed
``flashinfer`` package's ``data/`` tree (the wheel ships it for its own
JIT), so running this backend needs exactly one download and one env var —
no extra source checkout.

This module vendors glue plus one narrowly scoped routing-source override:

  * header staging: the pool ships the batched-gemm ABI headers flat; they
    are copied into a content-addressed include tree shaped like
    ``flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/``;
  * an opt-in Kimi K3 dynamic-block routing selector for the exact validated
    E=896, top-k=16, TP/BF16/PDL decode envelope;
  * JIT build of the 12 launcher/runner/routing sources with the private
    ABI defines (``TLLM_GEN_LOCAL_CUBINS_ABI`` etc.);
  * the ctypes cubin-loader callback (the .so asks for cubins by absolute
    path + sha256; we read them from the pool);
  * a thin ``trtllm_fp4_block_scale_moe`` wrapper (FromLogits routing,
    ``do_finalize=True``); kernel tile config ("tactic") defaults to the
    runner's built-in heuristic — pass an explicit one for tuned setups.

Validated for the Kimi K3 decode/prefill MoE regime: MxFP4 weights with
bf16 (w4a16) or MxFP8 (w4a8) activations, ``ActivationType.Situ`` (SiTuGlu:
``a*tanh(g/a)*sigmoid(g) * b*tanh(u/b)``), DeepSeekV3/noaux_tc routing.

Two build sources are supported (``SGLANG_TRTLLM_GEN_MOE_SOURCE``):

  * ``pool`` (default when a cubin pool is configured): the private
    self-contained pool described above (pre-fix v0.6.13-era cubins).
  * ``flashinfer``: compiles the bundled rc5-merged launcher
    (``trtllm_gen_moe_k3_data/csrc_rc5/``, our workspace-arena/max-tile-N/PDL
    launcher rebased onto flashinfer v0.6.16rc5) against the installed
    flashinfer package's own JIT source tree. SiTU is upstream in rc5
    (``ActivationType.Situ == 10``); cubins and the batched-gemm ABI headers
    resolve from the flashinfer-cubin wheel through upstream's own loader,
    so the private pool, its overlay, and the K3 dynblock routing patches
    are all bypassed (rc5 ships its own high-expert routing optimization).
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import pathlib
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    override_jit_cuda_arch,
)
from sglang.kernels.ops.moe.trtllm_gen_moe_k3_overlay import (
    stage_k3_dynblock_overlay,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# ActivationType / RoutingMethodType values from trtllm-gen's tllm_enums
# (kept as plain ints here to avoid importing anything for them).
# NOTE: 9 is the PRIVATE cubin-pool ABI value for SiTU. Upstream flashinfer
# >= 0.6.16rc5 has Identity = 9 and Situ = 10; the "flashinfer" source mode
# translates at the wrapper boundary (see _resolve_activation_type).
ACTIVATION_SITU = 9
_UPSTREAM_ACTIVATION_SITU = 10
ROUTING_DEEPSEEK_V3 = 2
_ROUTING_TOPK = 5
_ROUTING_INPUT_FROM_LOGITS = 0
# NOTE: the enum VALUES start at 0; the "Mode 1/2/3" wording in upstream
# comments is documentation numbering, not the enum value.
_ROUTING_INPUT_PACKED = 1

_ROUTING_COMMON_SOURCE = (
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu"
)

# Batched-gemm ABI headers shipped flat in the cubin pool; the launcher
# includes them as flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/<h>.
_BMM_EXPORT_HEADERS = [
    "BatchedGemmEnums.h",
    "BatchedGemmInterface.h",
    "BatchedGemmOptions.h",
    "Enums.h",
    "GemmGatedActOptions.h",
    "GemmOptions.h",
    "KernelParams.h",
    "KernelParamsDecl.h",
    "KernelTraits.h",
    "TmaDescriptor.h",
    "trtllm/gen/CommonUtils.h",
    "trtllm/gen/CudaArchDecl.h",
    "trtllm/gen/CudaKernelLauncher.h",
    "trtllm/gen/DtypeDecl.h",
    "trtllm/gen/MmaDecl.h",
    "trtllm/gen/SfLayoutDecl.h",
    "trtllm/gen/SparsityDecl.h",
]

_SOURCES = [
    "csrc/nv_internal/cpp/kernels/quantization.cu",
    "csrc/nv_internal/cpp/common/envUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/memoryUtils.cu",
    "csrc/trtllm_fused_moe_kernel_launcher.cu",
    "csrc/trtllm_fused_moe_runner.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu",
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
    _ROUTING_COMMON_SOURCE,
    "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu",
    "csrc/trtllm_batched_gemm_runner.cu",
]


logger = logging.getLogger(__name__)
_TRTLLM_MOE_PDL_MAX_TOKENS = envs.SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS.get()

_FP4_WORKSPACE_LAYOUT_VERSION = 1
_FP4_WORKSPACE_LAYOUT_FIELDS = 10
_fp4_workspace_layout_cache: dict[tuple[object, ...], TrtllmFp4WorkspaceLayout] = {}


@dataclass(frozen=True)
class TrtllmFp4WorkspaceLayout:
    """Native layout of one FP4 MoE invocation inside a byte arena."""

    descriptor: tuple[int, ...]
    required_bytes: int
    alignment: int
    tile_n: int
    tactic: int
    gemm2_offset: int
    gemm2_rows: int
    gemm2_size_bytes: int
    expanded_idx_offset: int
    expanded_idx_size_bytes: int

    @classmethod
    def from_native(cls, values: Sequence[int]) -> TrtllmFp4WorkspaceLayout:
        descriptor = tuple(int(value) for value in values)
        if len(descriptor) != _FP4_WORKSPACE_LAYOUT_FIELDS:
            raise RuntimeError(
                "Invalid TRTLLM FP4 workspace layout: expected "
                f"{_FP4_WORKSPACE_LAYOUT_FIELDS} fields, got {len(descriptor)}."
            )
        if descriptor[0] != _FP4_WORKSPACE_LAYOUT_VERSION:
            raise RuntimeError(
                "Unsupported TRTLLM FP4 workspace layout ABI "
                f"{descriptor[0]}; expected {_FP4_WORKSPACE_LAYOUT_VERSION}."
            )
        layout = cls(
            descriptor=descriptor,
            required_bytes=descriptor[1],
            alignment=descriptor[2],
            tile_n=descriptor[3],
            tactic=descriptor[4],
            gemm2_offset=descriptor[5],
            gemm2_rows=descriptor[6],
            gemm2_size_bytes=descriptor[7],
            expanded_idx_offset=descriptor[8],
            expanded_idx_size_bytes=descriptor[9],
        )
        if layout.required_bytes < 0 or layout.alignment <= 0:
            raise RuntimeError(f"Invalid TRTLLM FP4 workspace layout: {layout}.")
        if layout.gemm2_size_bytes % 2 or layout.expanded_idx_size_bytes % 4:
            raise RuntimeError(
                f"TRTLLM FP4 workspace layout has mis-sized typed segments: {layout}."
            )
        return layout

    def typed_views(
        self, workspace: torch.Tensor, hidden_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_workspace_tensor(workspace, self)
        expected_gemm2_bytes = self.gemm2_rows * hidden_size * 2
        if self.gemm2_size_bytes != expected_gemm2_bytes:
            raise RuntimeError(
                "TRTLLM FP4 gemm2 layout mismatch: native segment has "
                f"{self.gemm2_size_bytes} bytes, expected "
                f"{expected_gemm2_bytes} for [{self.gemm2_rows}, {hidden_size}]."
            )
        gemm2_out = (
            workspace.narrow(0, self.gemm2_offset, self.gemm2_size_bytes)
            .view(torch.bfloat16)
            .view(self.gemm2_rows, hidden_size)
        )
        expanded_idx = workspace.narrow(
            0, self.expanded_idx_offset, self.expanded_idx_size_bytes
        ).view(torch.int32)
        return gemm2_out, expanded_idx


def _validate_tactic_cap(
    tactic: Sequence[int], max_tile_n: Optional[int]
) -> tuple[int, int]:
    if len(tactic) != 2:
        raise ValueError(f"tactic must be (tile_N, config), got {tuple(tactic)!r}.")
    tile_n, config = (int(tactic[0]), int(tactic[1]))
    if max_tile_n is not None:
        max_tile_n = int(max_tile_n)
        if max_tile_n <= 0:
            raise ValueError(f"max_tile_n must be positive, got {max_tile_n}.")
        # Check the raw tile even when config == -1. Otherwise native fallback
        # semantics ("either field is -1") could silently turn (256, -1) into
        # a smaller default tile instead of enforcing the caller's cap.
        if tile_n >= 0 and tile_n > max_tile_n:
            raise ValueError(
                f"Explicit FP4 MoE tile_N={tile_n} exceeds max_tile_n={max_tile_n}."
            )
    return tile_n, config


def _validate_workspace_tensor(
    workspace: torch.Tensor, layout: TrtllmFp4WorkspaceLayout
) -> None:
    if workspace.dtype != torch.uint8:
        raise TypeError(
            f"FP4 workspace must have torch.uint8 dtype, got {workspace.dtype}."
        )
    if workspace.ndim != 1 or not workspace.is_contiguous():
        raise ValueError("FP4 workspace must be a contiguous 1-D byte arena.")
    if workspace.device.type != "cuda":
        raise ValueError(f"FP4 workspace must be on CUDA, got {workspace.device}.")
    if workspace.data_ptr() % layout.alignment:
        raise ValueError(f"FP4 workspace base must be {layout.alignment}-byte aligned.")
    if workspace.numel() < layout.required_bytes:
        raise ValueError(
            f"FP4 workspace has {workspace.numel()} bytes, but tile_N="
            f"{layout.tile_n}, tactic={layout.tactic} requires "
            f"{layout.required_bytes} bytes."
        )


def _workspace_layout_op(module):
    # Private cubin-pool builds suffix the op; source-only/unit builds may not.
    for name in (
        "trtllm_fp4_block_scale_moe_workspace_layout_private",
        "trtllm_fp4_block_scale_moe_workspace_layout",
    ):
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    raise AttributeError("TRTLLM FP4 workspace layout query is unavailable.")


def _workspace_moe_op(module):
    for name in (
        "trtllm_fp4_block_scale_moe_workspace_private",
        "trtllm_fp4_block_scale_moe_workspace",
    ):
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    raise AttributeError("TRTLLM FP4 workspace-aware MoE op is unavailable.")


def _legacy_moe_op(module):
    for name in (
        "trtllm_fp4_block_scale_moe_private",
        "trtllm_fp4_block_scale_moe",
    ):
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    raise AttributeError("TRTLLM FP4 legacy MoE op is unavailable.")


def _invoke_fp4_moe(
    module,
    args: tuple[object, ...],
    *,
    workspace: Optional[torch.Tensor],
    workspace_layout: Optional[TrtllmFp4WorkspaceLayout],
    max_tile_n: Optional[int],
):
    if workspace is None and max_tile_n is None:
        return _legacy_moe_op(module)(*args)
    return _workspace_moe_op(module)(
        *args,
        workspace,
        [] if workspace_layout is None else list(workspace_layout.descriptor),
        max_tile_n,
    )


def trtllm_fp4_block_scale_moe_workspace_layout(
    *,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    activation_type: int = ACTIVATION_SITU,
    local_num_experts: Optional[int] = None,
    tactic: Sequence[int] = (-1, -1),
    per_token_scale: Optional[torch.Tensor] = None,
    max_tile_n: Optional[int] = None,
) -> TrtllmFp4WorkspaceLayout:
    """Query and cache the native byte layout for one FP4 MoE shape."""
    module = _jit_trtllm_gen_moe_module()
    activation_type = _resolve_activation_type(activation_type)
    tactic_pair = _validate_tactic_cap(tactic, max_tile_n)
    local_num_experts = num_experts if local_num_experts is None else local_num_experts
    key = (
        id(module),
        hidden_states.shape[0],
        hidden_states.shape[1],
        hidden_states.dtype,
        (
            None
            if hidden_states_scale is None
            else (tuple(hidden_states_scale.shape), hidden_states_scale.dtype)
        ),
        tuple(gemm1_weights_scale.shape),
        gemm1_weights_scale.dtype,
        per_token_scale is not None,
        num_experts,
        top_k,
        intermediate_size,
        local_num_experts,
        activation_type,
        tactic_pair,
        max_tile_n,
    )
    cached = _fp4_workspace_layout_cache.get(key)
    if cached is not None:
        return cached
    values = _workspace_layout_op(module)(
        hidden_states,
        hidden_states_scale,
        gemm1_weights_scale,
        per_token_scale,
        num_experts,
        top_k,
        intermediate_size,
        local_num_experts,
        activation_type,
        list(tactic_pair),
        max_tile_n,
    )
    layout = TrtllmFp4WorkspaceLayout.from_native(values)
    _fp4_workspace_layout_cache[key] = layout
    return layout


def trtllm_fp4_block_scale_moe_workspace_size(**kwargs) -> int:
    """Return the native required arena size in bytes for one FP4 MoE shape."""
    return trtllm_fp4_block_scale_moe_workspace_layout(**kwargs).required_bytes


# The rc5-merged private launcher (workspace arena + max_tile_n + explicit
# PDL plumbed through routing) compiled by the "flashinfer" source mode.
_RC5_LAUNCHER = (
    pathlib.Path(__file__).with_name("trtllm_gen_moe_k3_data")
    / "csrc_rc5"
    / "trtllm_fused_moe_kernel_launcher.cu"
)
_RC5_LAUNCHER_SHA256 = (
    "4153db87ecc2aa0acf1d21927dd4c6ea74fed8a14d4f0d64f59f43c819ce3de3"
)


@cache_once
def moe_source() -> str:
    """Resolve the build source: explicit env wins, else pool-if-configured.

    Cached: env and image contents are fixed for the process lifetime, and
    this is reached from every MoE layer forward via _resolve_activation_type
    (prod stack sampling put the uncached filesystem probes at a double-digit
    percent of scheduler CPU time under gVisor).
    """
    value = (envs.SGLANG_TRTLLM_GEN_MOE_SOURCE.get() or "").strip().lower()
    if value in ("pool", "flashinfer"):
        return value
    if value:
        raise ValueError(
            "SGLANG_TRTLLM_GEN_MOE_SOURCE must be 'pool', 'flashinfer' or "
            f"empty, got {value!r}"
        )
    return "pool" if cubin_pool_dir() is not None else "flashinfer"


def _resolve_activation_type(activation_type: int) -> int:
    """Translate the private SiTU ABI value to upstream's on the rc5 path.

    Idempotent: the upstream value passes through unchanged, so internal
    re-entry (the MoE wrappers call the layout query with an
    already-translated value) is safe.
    """
    if activation_type == ACTIVATION_SITU and moe_source() == "flashinfer":
        return _UPSTREAM_ACTIVATION_SITU
    return activation_type


@cache_once
def cubin_pool_dir() -> Optional[pathlib.Path]:
    # Cached: the pool ships in the image and cannot appear or vanish at
    # runtime; see moe_source() for the hot-path cost rationale.
    p = envs.SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL.get()
    if not p:
        return None
    pool = pathlib.Path(p)
    return pool if pool.is_dir() else None


def _flashinfer_data_dir() -> Optional[pathlib.Path]:
    """The installed public flashinfer package's JIT source tree (ships
    csrc/, include/ and its pinned cutlass), used as the base layer under
    the pool's overlay."""
    try:
        import flashinfer  # noqa: PLC0415
    except ImportError:
        return None
    data = pathlib.Path(flashinfer.__file__).parent / "data"
    return data if (data / "csrc").is_dir() else None


def _flashinfer_bmm_artifact_dir() -> Optional[pathlib.Path]:
    """The trtllm-gen batched-gemm artifact inside the flashinfer-cubin wheel.

    Holds the post-fix cubins plus the paired ``flashinferMetaInfo.h`` and
    flat ABI headers. Returns None when the wheel (or the artifact pin) is
    missing, so availability fails closed.
    """
    try:
        from flashinfer.artifacts import ArtifactPath  # noqa: PLC0415
        from flashinfer.jit import env as fi_jit_env  # noqa: PLC0415
    except ImportError:
        return None
    artifact = (
        pathlib.Path(fi_jit_env.FLASHINFER_CUBIN_DIR) / ArtifactPath.TRTLLM_GEN_BMM
    )
    include = artifact / "include"
    if not (include / "flashinferMetaInfo.h").is_file():
        return None
    if not (include / "trtllmGen_bmm_export").is_dir():
        return None
    return artifact


def _flashinfer_has_upstream_situ() -> bool:
    """Whether the installed flashinfer ships SiTU natively (>= 0.6.16rc5)."""
    fi_data = _flashinfer_data_dir()
    if fi_data is None:
        return False
    runner_header = (
        fi_data / "include" / "flashinfer" / "trtllm" / "fused_moe" / "runner.h"
    )
    try:
        return "Situ = 10," in runner_header.read_text()
    except OSError:
        return False


def _flashinfer_native_available() -> bool:
    return (
        _RC5_LAUNCHER.is_file()
        and _flashinfer_has_upstream_situ()
        and _flashinfer_bmm_artifact_dir() is not None
    )


@cache_once
def available() -> bool:
    # Cached: called as a guard from every SiTU MoE layer forward
    # (mxfp4 apply); the underlying artifacts are image-constant.
    if moe_source() == "flashinfer":
        return _flashinfer_native_available()
    pool = cubin_pool_dir()
    return (
        pool is not None
        and (pool / "flashinferMetaInfo.h").is_file()
        and (pool / "local").is_dir()
        # Modified sources ship in the pool's overlay/, everything else
        # compiles from the installed flashinfer package.
        and (pool / "overlay" / "csrc").is_dir()
        and _flashinfer_data_dir() is not None
    )


def _stage_headers(pool: pathlib.Path) -> pathlib.Path:
    """Copy the pool's ABI headers into a content-addressed include tree."""
    meta = (pool / "flashinferMetaInfo.h").read_bytes()
    tag = hashlib.sha256(meta).hexdigest()[:12]
    cache = pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()
    root = cache / "trtllm_gen_moe_headers" / tag
    dest = root / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export"
    stamp = root / ".staged"
    if not stamp.is_file():
        for name in _BMM_EXPORT_HEADERS:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pool / name, target)
        shutil.copyfile(pool / "flashinferMetaInfo.h", dest / "flashinferMetaInfo.h")
        stamp.touch()
    return root


def _cuda_home() -> pathlib.Path:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        nvcc = shutil.which("nvcc")
        home = str(pathlib.Path(nvcc).parent.parent) if nvcc else "/usr/local/cuda"
    return pathlib.Path(home)


def _cuda_include_dir() -> str:
    return str(_cuda_home() / "include")


def _cuda_stub_ldflags() -> list[str]:
    """-L flags for the libcuda driver stub, so -lcuda links in bare build
    environments (containers without the driver lib on the default linker
    path); the real driver is dlopened at runtime as usual."""
    home = _cuda_home()
    stubs = [
        home / "lib64" / "stubs",
        *home.glob("targets/*/lib/stubs"),
    ]
    return [f"-L{s}" for s in stubs if s.is_dir()]


_CUBIN_CB_KEEPALIVE = {}


def _setup_cubin_loader(so_path: str, pool_local: pathlib.Path) -> None:
    """Register the ctypes callback the .so uses to fetch cubins by name.

    The runner requests ``<TLLM_GEN_GEMM_CUBIN_PATH>/<kernel>`` (absolute,
    because the pool path is baked in at compile time); we read the bytes
    and hand them back via FlashInferSetCurrentCubin.
    """
    if so_path in _CUBIN_CB_KEEPALIVE:
        return
    lib = ctypes.CDLL(so_path)
    cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

    def _get_cubin(name: bytes, sha256: bytes) -> None:
        rel = name.decode()
        path = pathlib.Path(rel)
        if not path.is_absolute():
            path = pool_local / rel
        if path.suffix != ".cubin":
            path = path.with_name(path.name + ".cubin")
        data = path.read_bytes()
        want = sha256.decode()
        if want:
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                raise RuntimeError(
                    f"cubin sha mismatch for {path}: want {want} got {got}"
                )
        lib.FlashInferSetCurrentCubin(
            ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.c_char_p),
            ctypes.c_int(len(data)),
        )

    cb = cb_type(_get_cubin)
    _CUBIN_CB_KEEPALIVE[so_path] = (lib, cb)
    lib.FlashInferSetCubinCallback(cb)


@cache_once
def _jit_trtllm_gen_moe_module() -> Module:
    if moe_source() == "flashinfer":
        return _jit_flashinfer_native_module()
    return _jit_pool_module()


def _stage_flashinfer_headers(artifact_include: pathlib.Path) -> pathlib.Path:
    """Copy the wheel artifact's ABI headers into a content-addressed tree.

    Mirrors ``_stage_headers`` (the pool variant): the launcher includes the
    headers as ``flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/<h>``
    and ``BatchedGemmInterface.h`` includes ``flashinferMetaInfo.h`` relative
    to itself, so the metainfo header is staged into the same directory.
    """
    meta = (artifact_include / "flashinferMetaInfo.h").read_bytes()
    tag = hashlib.sha256(meta).hexdigest()[:12]
    cache = pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()
    root = cache / "trtllm_gen_moe_headers_fi" / tag
    dest = root / "flashinfer" / "trtllm" / "batched_gemm" / "trtllmGen_bmm_export"
    stamp = root / ".staged"
    if not stamp.is_file():
        for name in _BMM_EXPORT_HEADERS:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_include / "trtllmGen_bmm_export" / name, target)
        shutil.copyfile(
            artifact_include / "flashinferMetaInfo.h", dest / "flashinferMetaInfo.h"
        )
        stamp.touch()
    return root


def _jit_flashinfer_native_module() -> Module:
    """Build the rc5-merged launcher against the installed flashinfer tree.

    No private pool, no overlay: every source except the launcher comes from
    the flashinfer wheel's ``data/`` tree, the batched-gemm ABI headers and
    ``flashinferMetaInfo.h`` come from the flashinfer-cubin wheel's pinned
    artifact, and the runtime cubins resolve through upstream's own loader
    callback (``flashinfer.jit.cubin_loader.setup_cubin_loader``).
    """
    from flashinfer.artifacts import ArtifactPath  # noqa: PLC0415
    from flashinfer.jit import env as fi_jit_env  # noqa: PLC0415

    fi_data = _flashinfer_data_dir()
    artifact = _flashinfer_bmm_artifact_dir()
    if fi_data is None or artifact is None or not _flashinfer_has_upstream_situ():
        raise RuntimeError(
            "trtllm-gen MoE flashinfer-native sources not found: needs "
            "flashinfer-python >= 0.6.16rc5 (upstream SiTU) and the matching "
            "flashinfer-cubin wheel."
        )

    logger.info(
        "trtllm-gen MoE build source=flashinfer-native-rc5 (upstream SiTU, "
        "flashinfer-cubin wheel; private pool/overlay bypassed)"
    )
    launcher = _RC5_LAUNCHER.read_bytes()
    launcher_sha256 = hashlib.sha256(launcher).hexdigest()
    if launcher_sha256 != _RC5_LAUNCHER_SHA256:
        raise RuntimeError(
            "Bundled rc5 TRT-LLM-gen launcher failed its integrity check: "
            f"expected {_RC5_LAUNCHER_SHA256}, got {launcher_sha256}."
        )

    cache = pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()
    staged = _stage_flashinfer_headers(artifact / "include")
    meta_tag = staged.name
    launcher_tag = launcher_sha256[:12]
    build_dir = cache / f"sgl_trtllm_gen_moe_fi_{meta_tag}_{launcher_tag}"

    def _fi_source(rel: str) -> str:
        cand = fi_data / rel
        if not cand.is_file():
            raise RuntimeError(f"flashinfer JIT source not found: {cand}")
        return str(cand)

    cpp_files = [_fi_source(s) for s in _SOURCES if s.endswith(".cpp")]
    cuda_files = [
        (
            str(_RC5_LAUNCHER)
            if s == "csrc/trtllm_fused_moe_kernel_launcher.cu"
            else _fi_source(s)
        )
        for s in _SOURCES
        if s.endswith(".cu")
    ]

    # Upstream passes the artifact-relative path (trailing slash included);
    # the runtime callback prepends FLASHINFER_CUBIN_DIR. Keep it identical.
    cubin_path = ArtifactPath.TRTLLM_GEN_BMM
    arch = get_jit_cuda_arch()
    with override_jit_cuda_arch(arch.major, arch.minor, "a"):
        module = load_jit(
            "trtllm_gen_moe_fi",
            meta_tag,
            launcher_tag,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            header_only=False,  # the launcher exports its own tvm-ffi functions
            # c++17 must come last: gcc-13 ICEs (cc1plus segfault) on rc5's
            # trtllm_batched_gemm_runner.cu under the default -std=c++20, and
            # upstream's own JIT compiles these sources as c++17.
            extra_cflags=["-fvisibility=hidden", "-std=c++17"],
            extra_cuda_cflags=[
                "-std=c++17",
                # Match upstream's module recipe (flashinfer/jit/core.py):
                # missing feature defines route FP4/FP8 templates through
                # fallback branches upstream never compiles.
                "-DNDEBUG",
                "-DFLASHINFER_ENABLE_FP8_E8M0",
                "-DFLASHINFER_ENABLE_FP4_E2M1",
                "-DTLLM_GEN_EXPORT_INTERFACE",
                "-DTLLM_GEN_EXPORT_FLASHINFER",
                "-DTLLM_ENABLE_CUDA",
                "-DENABLE_BF16",
                "-DENABLE_FP8",
                "-DENABLE_FP4",
                "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{cubin_path}\\"',
                "-Xcompiler=-fvisibility=hidden",
            ],
            extra_ldflags=[*_cuda_stub_ldflags(), "-lcuda", "-lnvrtc"],
            extra_include_paths=[
                # Vendored CCCL first, mirroring upstream's CTK-override
                # precedence: rc5 sources target these cub/libcudacxx/thrust
                # versions, not the CUDA toolkit's.
                str(fi_data / "cccl" / "cub"),
                str(fi_data / "cccl" / "libcudacxx" / "include"),
                str(fi_data / "cccl" / "thrust"),
                str(staged),
                str(
                    staged
                    / "flashinfer"
                    / "trtllm"
                    / "batched_gemm"
                    / "trtllmGen_bmm_export"
                ),
                str(fi_data / "include"),
                str(fi_data / "csrc"),
                str(fi_data / "csrc" / "nv_internal"),
                str(fi_data / "csrc" / "nv_internal" / "include"),
                str(fi_data / "cutlass" / "include"),
                # rc5's flashinfer/logging.h requires the wheel's bundled spdlog.
                str(fi_data / "spdlog" / "include"),
                _cuda_include_dir(),
            ],
            build_directory=str(build_dir),
        )
    so_files = list(build_dir.glob("*.so"))
    if len(so_files) != 1:
        raise RuntimeError(
            f"expected exactly one built .so under {build_dir}, got {so_files}"
        )
    # Upstream cubin resolution: reads from FLASHINFER_CUBIN_DIR (the
    # flashinfer-cubin wheel) with per-kernel sha256 verification against the
    # hashes baked into flashinferMetaInfo.h.
    from flashinfer.jit.cubin_loader import setup_cubin_loader  # noqa: PLC0415

    setup_cubin_loader(str(so_files[0]))
    _ = fi_jit_env  # imported for its FLASHINFER_CUBIN_DIR side effects above
    return module


def _jit_pool_module() -> Module:
    pool = cubin_pool_dir()
    fi_data = _flashinfer_data_dir()
    if pool is None or not (pool / "overlay" / "csrc").is_dir() or fi_data is None:
        raise RuntimeError(
            "trtllm-gen MoE sources not found: point "
            "SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL at an unpacked cubin pool "
            "(cubins + flat ABI headers + overlay/) and install the public "
            "flashinfer package."
        )

    cache = pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()
    k3_overlay, k3_overlay_tag = stage_k3_dynblock_overlay(pool / "overlay", cache)
    # The staged K3 sources shadow the pool overlay. The pool's remaining
    # modified sources shadow the installed FlashInfer base.
    src_roots = [k3_overlay, pool / "overlay", fi_data]
    include_roots = [k3_overlay, pool / "overlay", fi_data]

    def _resolve_source(rel: str) -> str:
        for root in src_roots:
            cand = root / rel
            if cand.is_file():
                return str(cand)
        raise RuntimeError(f"trtllm-gen MoE source not found in any root: {rel}")

    staged = _stage_headers(pool)
    meta_tag = staged.name
    cubin_path = str((pool / "local").resolve())

    # Flags are not part of load_jit's source hash: fold the pool identity
    # and staged overlay into both the module marker and build directory.
    path_tag = hashlib.sha256(cubin_path.encode()).hexdigest()[:8]
    build_dir = cache / f"sgl_trtllm_gen_moe_{meta_tag}_{path_tag}_{k3_overlay_tag}"

    cpp_files = [_resolve_source(s) for s in _SOURCES if s.endswith(".cpp")]
    cuda_files = [_resolve_source(s) for s in _SOURCES if s.endswith(".cu")]
    # quantization.cu emits fp4 cvt instructions (.e2m1x2) that need the
    # arch-specific feature set: compile for sm_XXXa, not plain sm_XXX.
    # The trtllm-gen cubins themselves are prebuilt (sm100f) and loaded at
    # runtime, unaffected by this flag.
    arch = get_jit_cuda_arch()
    with override_jit_cuda_arch(arch.major, arch.minor, "a"):
        module = load_jit(
            "trtllm_gen_moe",
            meta_tag,
            path_tag,
            k3_overlay_tag,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            header_only=False,  # the launcher exports its own tvm-ffi functions
            extra_cflags=["-fvisibility=hidden"],
            extra_cuda_cflags=[
                "-DTLLM_GEN_EXPORT_INTERFACE",
                "-DTLLM_GEN_EXPORT_FLASHINFER",
                "-DTLLM_ENABLE_CUDA",
                "-DENABLE_BF16",
                "-DENABLE_FP8",
                "-DENABLE_FP4",
                "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                "-DTLLM_GEN_LOCAL_CUBINS_ABI",
                "-DFLASHINFER_PRIVATE_MOE_FFI_NAMES",
                "-DFLASHINFER_PRIVATE_MOE_LEAN_ROUTING",
                f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{cubin_path}\\"',
                "-Xcompiler=-fvisibility=hidden",
            ],
            extra_ldflags=[*_cuda_stub_ldflags(), "-lcuda", "-lnvrtc"],
            extra_include_paths=[
                str(staged),
                str(
                    staged
                    / "flashinfer"
                    / "trtllm"
                    / "batched_gemm"
                    / "trtllmGen_bmm_export"
                ),
                # Per-root include layout: include/, csrc/, csrc/nv_internal/,
                # csrc/nv_internal/include/, plus the flashinfer package's
                # pinned CUTLASS (data/cutlass/). The overlay root comes first
                # so modified headers shadow the public copies.
                *[
                    str(root / sub)
                    for root in include_roots
                    for sub in (
                        "include",
                        "csrc",
                        "csrc/nv_internal",
                        "csrc/nv_internal/include",
                    )
                ],
                *[
                    str(root / "cutlass" / "include")
                    for root in include_roots
                    if (root / "cutlass" / "include").is_dir()
                ],
                # Host .cpp files (g++) need the CUDA headers explicitly; nvcc
                # adds them implicitly for .cu. CUDA 13's bundled CCCL is
                # used as-is (mixing another pinned CCCL with the toolkit's
                # explodes).
                _cuda_include_dir(),
            ],
            build_directory=str(build_dir),
        )
    so_files = list(build_dir.glob("*.so"))
    if len(so_files) != 1:
        raise RuntimeError(
            f"expected exactly one built .so under {build_dir}, got {so_files}"
        )
    _setup_cubin_loader(str(so_files[0]), pool / "local")
    return module


def trtllm_fp4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = ROUTING_DEEPSEEK_V3,
    activation_type: int = ACTIVATION_SITU,
    norm_topk_prob: bool = True,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    tactic: Sequence[int] = (-1, -1),
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    max_tile_n: Optional[int] = None,
) -> torch.Tensor:
    """FP4 block-scale MoE with routing from logits and finalize fused.

    ``hidden_states``: bf16 ``[T, hidden]`` (w4a16) or MxFP8-packed uint8
    with ``hidden_states_scale`` (w4a8). Weights are trtllm-gen shuffled
    MxFP4 (uint8 packed, fp8 block scales, MajorK). ``tactic`` is
    ``(tile_N, config)``; ``(-1, -1)`` selects the runner heuristic.
    ``workspace`` is an optional caller-owned contiguous CUDA uint8 arena.
    """
    module = _jit_trtllm_gen_moe_module()
    activation_type = _resolve_activation_type(activation_type)
    tactic_pair = _validate_tactic_cap(tactic, max_tile_n)
    # The FFI launcher reads these as dense row-major; a strided slice
    # (e.g. a fused-GEMM split) would silently mis-route.
    routing_logits = routing_logits.contiguous()
    hidden_states = hidden_states.contiguous()
    num_tokens = routing_logits.shape[0]
    hidden_size = hidden_states.shape[-1]
    if hidden_states.dtype == torch.uint8:
        hidden_size *= 2
    device = hidden_states.device
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
    topk_weights = torch.empty(
        num_tokens, top_k, dtype=routing_logits.dtype, device=device
    )
    if output is None:
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    workspace_layout = None
    if workspace is not None:
        workspace_layout = trtllm_fp4_block_scale_moe_workspace_layout(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights_scale=gemm1_weights_scale,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            local_num_experts=local_num_experts,
            tactic=tactic_pair,
            max_tile_n=max_tile_n,
        )
        _validate_workspace_tensor(workspace, workspace_layout)
        if workspace.device != hidden_states.device:
            raise ValueError(
                "FP4 workspace and hidden_states must be on the same device."
            )
    _invoke_fp4_moe(
        module,
        (
            _ROUTING_INPUT_FROM_LOGITS,
            routing_logits,
            topk_ids,
            topk_weights,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            None,  # gemm1_bias
            gemm1_alpha,
            gemm1_beta,
            None,  # gemm1_clamp_limit
            gemm2_weights,
            gemm2_weights_scale,
            None,  # gemm2_bias
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            None,  # per_token_scale
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            num_experts if local_num_experts is None else local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            True,  # do_finalize
            num_tokens <= _TRTLLM_MOE_PDL_MAX_TOKENS,  # enable_pdl
            activation_type,
            output,
            list(tactic_pair),
            norm_topk_prob,
            None,  # routing_replay_out
        ),
        workspace=workspace,
        workspace_layout=workspace_layout,
        max_tile_n=max_tile_n,
    )
    return output


def trtllm_fp4_block_scale_routed_moe(
    packed_topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    activation_type: int = ACTIVATION_SITU,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    tactic: Sequence[int] = (-1, -1),
    output: Optional[torch.Tensor] = None,
    do_finalize: bool = True,
    workspace: Optional[torch.Tensor] = None,
    max_tile_n: Optional[int] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP4 block-scale MoE with PRECOMPUTED routing (PackedPrecomputed).

    ``packed_topk_ids``: int32 ``[T, top_k]`` with ``(expert_id << 16) |
    bf16-weight-bits`` (PackTopkIds layout) — selection and weights come
    from the caller's router, the in-op routing kernels are skipped. This
    is the fast path at small T, where the in-op single-CTA routing kernel
    (~22 µs at 896 experts) costs more than an external radix router.

    ``do_finalize=False`` skips the in-op finalize (top-k weighted
    unpermute) and returns its inputs instead:
    ``(gemm2_output [padded_rows, hidden] bf16 in permuted layout,
    topk_weights [T, top_k] bf16 unpacked from packed_topk_ids,
    expanded_idx_to_permuted_idx [T*top_k] int32 with -1 = dropped slot)``.
    ``output`` is left unwritten in that mode.

    ``workspace`` is an optional caller-owned contiguous CUDA uint8 arena.
    When finalize is deferred, the returned GEMM2/index tensors are typed
    PyTorch views of that arena, so their lifetime remains anchored by the
    caller-owned tensor.
    """
    module = _jit_trtllm_gen_moe_module()
    activation_type = _resolve_activation_type(activation_type)
    tactic_pair = _validate_tactic_cap(tactic, max_tile_n)
    hidden_states = hidden_states.contiguous()
    num_tokens = packed_topk_ids.shape[0]
    hidden_size = hidden_states.shape[-1]
    if hidden_states.dtype == torch.uint8:
        hidden_size *= 2
    device = hidden_states.device
    # Mode 2 unpacks the weights in-kernel; this is its output buffer.
    topk_weights = torch.empty(num_tokens, top_k, dtype=torch.bfloat16, device=device)
    if output is None:
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    workspace_layout = None
    arena_deferred_views = None
    if workspace is not None:
        workspace_layout = trtllm_fp4_block_scale_moe_workspace_layout(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights_scale=gemm1_weights_scale,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            local_num_experts=local_num_experts,
            tactic=tactic_pair,
            max_tile_n=max_tile_n,
        )
        _validate_workspace_tensor(workspace, workspace_layout)
        if workspace.device != hidden_states.device:
            raise ValueError(
                "FP4 workspace and hidden_states must be on the same device."
            )
        if not do_finalize:
            arena_deferred_views = workspace_layout.typed_views(workspace, hidden_size)
    result = _invoke_fp4_moe(
        module,
        (
            _ROUTING_INPUT_PACKED,
            None,  # routing_logits
            packed_topk_ids.contiguous(),
            topk_weights,
            None,  # routing_bias (already applied by the external router)
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            None,  # gemm1_bias
            gemm1_alpha,
            gemm1_beta,
            None,  # gemm1_clamp_limit
            gemm2_weights,
            gemm2_weights_scale,
            None,  # gemm2_bias
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            None,  # per_token_scale
            num_experts,
            top_k,
            None,  # n_group
            None,  # topk_group
            intermediate_size,
            local_expert_offset,
            num_experts if local_num_experts is None else local_num_experts,
            1.0,  # routed_scaling_factor (already applied by the router)
            _ROUTING_TOPK,  # routing_method_type (unused for precomputed)
            do_finalize,
            num_tokens <= _TRTLLM_MOE_PDL_MAX_TOKENS,  # enable_pdl
            activation_type,
            output,
            list(tactic_pair),
            True,  # norm_topk_prob (unused for precomputed)
            None,  # routing_replay_out
        ),
        workspace=workspace,
        workspace_layout=workspace_layout,
        max_tile_n=max_tile_n,
    )
    if do_finalize:
        return output
    if arena_deferred_views is not None:
        gemm2_out, expanded_idx = arena_deferred_views
        return gemm2_out, topk_weights, expanded_idx
    # Deferred: [gemm2_output, expert_weights (None in packed mode — the
    # weights live in the topk_weights buffer mode 2 unpacked into),
    # expanded_idx_to_permuted_idx]. Index access — iterating the tvm-ffi
    # Array yields one-shot dlpack capsules.
    return result[0], topk_weights, result[2]
