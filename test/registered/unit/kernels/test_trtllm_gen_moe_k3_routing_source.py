import hashlib

import pytest

from sglang.kernels.ops.moe import trtllm_gen_moe_k3_overlay as overlay
from sglang.kernels.ops.moe.trtllm_gen_moe_k3_overlay import SOURCE_PATCHES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _patch_text(path_suffix):
    patch = next(patch for patch in SOURCE_PATCHES if patch.path.endswith(path_suffix))
    return "\n".join(new for _, new in patch.replacements)


def test_k3_dynblock_selector_is_exactly_guarded():
    source = _patch_text("trtllm_fused_moe_routing_common.cu")

    assert 'std::getenv("FLASHINFER_K3_ROUTING_DYNBLOCK")' in source
    for predicate in (
        "data.mNumTokens <= 16",
        "data.mNumExperts == 896",
        "data.mTopK == 16",
        "data.mPtrTopKPacked != nullptr",
        "data.mDtypeOutput == tg::Dtype::Bfloat16",
        "data.mUsePdl",
        "data.mLocalExpertsStartIdx == 0",
        "data.mLocalExpertsStrideLog2 == 0",
        "data.mNumLocalExperts == data.mNumExperts",
        "dispatchedMaxExperts == routingCustom::NumExperts1024Experts",
    ):
        assert predicate in source


def test_k3_dynblock_1024_tier_uses_512_threads():
    source = _patch_text("trtllm_fused_moe_routing_custom.cu")

    assert "MaxNumExperts == 1024 ? 512" in source
    assert "if constexpr (MaxNumExperts == 1024)" in source
    assert "return 512;" in source
    assert "LAUNCH_ROUTING_CUSTOM_WITH_CONFIG" in source
    assert "DynBlockRoutingLaunchConfig" in source


def test_k3_dynblock_policy_supports_per_kernel_geometry():
    source = _patch_text("RoutingCustomPolicy.cuh")

    assert '#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"' in source
    assert "LAUNCH_ROUTING_CUSTOM_WITH_CONFIG" in source
    assert "LaunchConfig_::blockDim(data, static_cast<int>(numThreads))" in source
    assert "LAUNCH_ROUTING_WITH_POLICIES" in source


def test_k3_dynblock_overlay_hash_locks_all_three_sources():
    assert len(SOURCE_PATCHES) == 3
    assert all(len(patch.sha256) == 64 for patch in SOURCE_PATCHES)


def test_private_launcher_integrity_and_workspace_exports():
    source = overlay._REVIEWED_LAUNCHER.read_bytes()

    assert hashlib.sha256(source).hexdigest() == overlay._REVIEWED_LAUNCHER_SHA256
    text = source.decode()
    for export in (
        "trtllm_fp4_block_scale_moe_private",
        "trtllm_fp4_block_scale_moe_workspace_private",
        "trtllm_fp4_block_scale_moe_workspace_layout_private",
    ):
        assert export in text


def test_private_launcher_staging_is_fail_closed(monkeypatch, tmp_path):
    base = b"reviewed private launcher base"
    pool_overlay = tmp_path / "pool" / "overlay"
    launcher = pool_overlay / overlay._PRIVATE_LAUNCHER_PATH
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(base)

    monkeypatch.setattr(overlay, "SOURCE_PATCHES", ())
    monkeypatch.setattr(
        overlay, "_PRIVATE_LAUNCHER_BASE_SHA256", hashlib.sha256(base).hexdigest()
    )
    staged, _ = overlay.stage_k3_dynblock_overlay(pool_overlay, tmp_path / "cache")
    assert (staged / overlay._PRIVATE_LAUNCHER_PATH).read_bytes() == (
        overlay._REVIEWED_LAUNCHER.read_bytes()
    )

    launcher.write_bytes(base + b" drift")
    with pytest.raises(RuntimeError, match="does not match the reviewed base"):
        overlay.stage_k3_dynblock_overlay(pool_overlay, tmp_path / "other-cache")
