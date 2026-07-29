"""Minimal, fail-closed Kimi K3 routing overlay for TRT-LLM-gen MoE."""

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class SourcePatch:
    path: str
    sha256: str
    replacements: tuple[tuple[str, str], ...]

    def apply(self, source: bytes) -> bytes:
        actual_sha256 = hashlib.sha256(source).hexdigest()
        if actual_sha256 != self.sha256:
            raise RuntimeError(
                f"TRT-LLM-gen source {self.path} does not match the reviewed "
                f"base: expected {self.sha256}, got {actual_sha256}."
            )

        text = source.decode()
        for old, new in self.replacements:
            matches = text.count(old)
            if matches != 1:
                raise RuntimeError(
                    f"TRT-LLM-gen source patch for {self.path} expected one "
                    f"match, found {matches}."
                )
            text = text.replace(old, new)
        return text.encode()


_COMMON_PATH = "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu"
_CUSTOM_PATH = "csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu"
_POLICY_PATH = "include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"

_ENABLE_K3_DYNBLOCK = """\
namespace moe::dev::routing {
namespace {

bool enableKimiK3DynBlockRouting() {
  static bool const enabled = [] {
    char const* value = std::getenv("FLASHINFER_K3_ROUTING_DYNBLOCK");
    return value != nullptr && std::atoi(value) != 0;
  }();
  return enabled;
}

}  // namespace

namespace routingCustom {"""

_SELECT_K3_DYNBLOCK = """\
  bool const useGenericDynBlock =
      !useStaticBlock && data.mNumTokens <= routingCustom::DynBlockKernelMaxNumTokens &&
      dispatchedMaxExperts <= routingCustom::DynBlockKernelMaxNumExperts;
  // The exact K3 packed/plain-TP BF16 PDL path has a validated 512-thread
  // Tier<1024, 16> specialization. Everything else keeps the generic policy.
  bool const useKimiK3DynBlock =
      !useStaticBlock && enableKimiK3DynBlockRouting() && smMajor >= 10 &&
      data.mNumTokens <= 16 && data.mNumExperts == 896 && data.mTopK == 16 &&
      data.mPtrScores == nullptr && data.mPtrTopKIds == nullptr &&
      data.mPtrTopKPacked != nullptr && data.mPtrTopKWeights != nullptr &&
      data.mDtypeOutput == tg::Dtype::Bfloat16 && data.mUsePdl &&
      data.mTileTokensDim == 8 && data.mPaddingLog2 == 3 &&
      data.mLocalExpertsStartIdx == 0 && data.mLocalExpertsStrideLog2 == 0 &&
      data.mNumLocalExperts == 896 && data.mNumFusedSharedExperts == 0 &&
      dispatchedMaxExperts == routingCustom::NumExperts1024Experts;
  bool const useDynBlock = useGenericDynBlock || useKimiK3DynBlock;"""

_DYNBLOCK_LAUNCH_CONFIG = """\
template <int MaxNumExperts, int MaxNumTopExperts>
struct DynBlockRoutingLaunchConfig {
  static int blockDim(Data const& /*data*/, int numThreads) {
    if constexpr (MaxNumExperts == 1024) {
      return 512;
    }
    return numThreads;
  }

  static int gridDim(Data const& /*data*/, int numBlocks, int /*blockDim*/) {
    return numBlocks;
  }
};

void launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream) {"""

_CUSTOM_LAUNCH_DISPATCH = r"""// Per-kernel launch geometry. The generic dispatcher retains its established
// tier-sized block, while the K3 dynblock kernel can safely stripe E=1024
// across 512 threads.
#define LAUNCH_ROUTING_CUSTOM_WITH_CONFIG(data, coopLaunch, kernel, numBlocks, numThreads,       \
                                          smemSize, stream, LaunchConfig)                        \
  dispatchRoutingPolicy(data, [&](auto preProc_, auto postProc_, char const* policyName_) {      \
    using PreProc_ = decltype(preProc_);                                                         \
    using PostProc_ = decltype(postProc_);                                                       \
    using Pairs_ = typename PolicyTraits<PreProc_, PostProc_>::Pairs;                            \
    bool dispatched_ =                                                                           \
        dispatchTierPairs(static_cast<Pairs_*>(nullptr), data, [&](auto eTag_, auto kTag_) {     \
          using LaunchConfig_ = LaunchConfig<decltype(eTag_)::value, decltype(kTag_)::value>;    \
          int const effectiveThreads_ =                                                          \
              LaunchConfig_::blockDim(data, static_cast<int>(numThreads));                       \
          int const effectiveBlocks_ =                                                           \
              LaunchConfig_::gridDim(data, static_cast<int>(numBlocks), effectiveThreads_);      \
          LAUNCH_ROUTING_WITH_POLICIES(data, coopLaunch, kernel, effectiveBlocks_,               \
                                       effectiveThreads_, smemSize, stream, PreProc_, PostProc_, \
                                       decltype(eTag_)::value, decltype(kTag_)::value);          \
        });                                                                                      \
    if (!dispatched_) {                                                                          \
      FLASHINFER_WARN(                                                                           \
          "No compiled tier covers numExperts=%d topK=%d for policy %s. "                        \
          "Add a Tier<%d, %d> to the corresponding PolicyTraits in RoutingCustomPolicy.cuh.",    \
          data.mNumExperts, data.mTopK, policyName_, getMaxNumExperts(data.mNumExperts),         \
          data.mTopK);                                                                           \
    }                                                                                            \
  })

"""

SOURCE_PATCHES = (
    SourcePatch(
        path=_COMMON_PATH,
        sha256="df7605a8e4274cf4e1786d931a48beab4fc928ade029222f56311dabe2e905c2",
        replacements=(
            (
                '#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"',
                "#include <cstdlib>\n\n"
                '#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"',
            ),
            (
                "namespace moe::dev::routing {\nnamespace routingCustom {",
                _ENABLE_K3_DYNBLOCK,
            ),
            (
                """\
  bool const useDynBlock = !useStaticBlock &&
                           data.mNumTokens <= routingCustom::DynBlockKernelMaxNumTokens &&
                           dispatchedMaxExperts <= routingCustom::DynBlockKernelMaxNumExperts;""",
                _SELECT_K3_DYNBLOCK,
            ),
        ),
    ),
    SourcePatch(
        path=_CUSTOM_PATH,
        sha256="b2f31d8e91b398460170897df321e02b2e4a26b1d1c159d1fe864f54907a0e15",
        replacements=(
            (
                "  static constexpr int NumThreadsExperts = "
                "MaxNumExperts <= 1024 ? MaxNumExperts : 1024;",
                """\
  static constexpr int NumThreadsExperts =
      MaxNumExperts == 1024 ? 512 : (MaxNumExperts <= 1024 ? MaxNumExperts : 1024);""",
            ),
            (
                "void launchDynBlockKernel(Data const& data, "
                "uint32_t numThreadsHist, void* stream) {",
                _DYNBLOCK_LAUNCH_CONFIG,
            ),
            (
                """\
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesDynBlockKernel, 1, threads, smemSize, stream);""",
                """\
  LAUNCH_ROUTING_CUSTOM_WITH_CONFIG(data, false, routingIndicesDynBlockKernel, 1, threads,
                                    smemSize, stream, DynBlockRoutingLaunchConfig);""",
            ),
        ),
    ),
    SourcePatch(
        path=_POLICY_PATH,
        sha256="7acd9b3eb05e7f8e88de990c9003ebab7f59e0425ef3c6ecd37b3a01d978f968",
        replacements=(
            (
                '#include "RoutingKernel.cuh"',
                '#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"',
            ),
            (
                "#define LAUNCH_ROUTING_CUSTOM(data, coopLaunch, kernel, "
                "numBlocks, numThreads, smemSize, stream)",
                _CUSTOM_LAUNCH_DISPATCH
                + "#define LAUNCH_ROUTING_CUSTOM(data, coopLaunch, kernel, "
                "numBlocks, numThreads, smemSize, stream)",
            ),
        ),
    ),
)


def stage_k3_dynblock_overlay(
    pool_overlay: pathlib.Path,
    cache: pathlib.Path,
) -> tuple[pathlib.Path, str]:
    patched_sources = []
    digest = hashlib.sha256()
    for patch in SOURCE_PATCHES:
        patched = patch.apply((pool_overlay / patch.path).read_bytes())
        patched_sources.append((patch.path, patched))
        digest.update(patch.path.encode())
        digest.update(b"\0")
        digest.update(patched)
        digest.update(b"\0")

    tag = digest.hexdigest()[:16]
    root = cache / "trtllm_gen_moe_k3_overlay" / tag
    for relative_path, source in patched_sources:
        destination = root / relative_path
        if destination.is_file() and destination.read_bytes() == source:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.{os.getpid()}.",
            delete=False,
        ) as temporary:
            temporary.write(source)
            temporary_path = pathlib.Path(temporary.name)
        try:
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return root, tag
