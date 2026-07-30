"""Regression tests for the K3 TRT-LLM-gen MoE workspace tactic policy.

These tests intentionally model the host launcher's inexpensive shape math.
They do not need a GPU or the private cubin pool, so a production-critical
workspace/candidate regression is caught by the base CPU suite.
"""

from __future__ import annotations

from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_K3_NUM_TOKENS = 16_384
_K3_TOP_K = 16
_K3_NUM_EXPERTS = 896
_K3_HIDDEN_SIZE = 3_584
_K3_INTERMEDIATE_SIZE = 384
_FP4_TILE_LADDER = (8, 16, 32, 64, 128, 256)
_THREE_GIB = 3 * 1024**3
_FOUR_GIB = 4 * 1024**3
_ARENA_ALIGNMENT = 256
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LAUNCHER_SOURCE = (
    _REPO_ROOT
    / "python/sglang/kernels/ops/moe/trtllm_gen_moe_k3_data/csrc"
    / "trtllm_fused_moe_kernel_launcher.cu"
)


def _next_power_of_two(value: float) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def _selected_tiles(
    supported_tiles: tuple[int, ...],
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
) -> set[int]:
    """Mirror computeSelectedTileN's center, +1, +2, and -1 policy."""
    center = _next_power_of_two(num_tokens * top_k / num_experts)
    center = min(max(center, supported_tiles[0]), supported_tiles[-1])
    center_index = supported_tiles.index(center)

    selected = {center}
    selected.update(supported_tiles[center_index + 1 : center_index + 3])
    if center_index:
        selected.add(supported_tiles[center_index - 1])
    return selected


def _align_up(size: int, alignment: int = _ARENA_ALIGNMENT) -> int:
    return (size + alignment - 1) // alignment * alignment


def _k3_workspace_bytes(tile_n: int) -> int:
    """Exact launcher-owned arena size for the production K3 prefill shape.

    This covers FP4BlockScaleLauncher's routing buffers, MxFP8 GEMM1 output
    and block scales, and BF16 GEMM2 output. The native path separately
    validates that the supplied arena's base address has this alignment. Both
    batched-GEMM tactic workspaces are zero bytes for the audited K3 cubins.
    """
    expanded_tokens = _K3_NUM_TOKENS * _K3_TOP_K
    experts_filled = min(_K3_NUM_EXPERTS, expanded_tokens)
    remaining_tokens = expanded_tokens - experts_filled
    max_ctas = experts_filled + remaining_tokens // tile_n
    max_padded_tokens = max_ctas * tile_n

    # computeSwizzledLayoutSFSize uses SWIZZLED_128x4 for tile_n >= 128.
    scale_rows = _align_up(max_padded_tokens, 128)
    scale_cols = _align_up(_K3_INTERMEDIATE_SIZE // 32, 4)

    region_sizes = (
        _K3_NUM_EXPERTS * 4,  # num_tokens_per_expert (int32)
        4,  # total_num_padded_tokens (int32)
        expanded_tokens * 4,  # expanded_idx_to_permuted_idx (int32)
        max_padded_tokens * 4,  # permuted_idx_to_token_idx (int32)
        max(_K3_NUM_EXPERTS * 2, 512) * 4,  # expert_count_histogram
        max_ctas * 4,  # cta_idx_xy_to_batch_idx (int32)
        max_ctas * 4,  # cta_idx_xy_to_mn_limit (int32)
        4,  # num_non_exiting_ctas (int32)
        max_padded_tokens * _K3_INTERMEDIATE_SIZE,  # MxFP8 GEMM1 output
        scale_rows * scale_cols,  # GEMM1 block scales (uint8)
        max_padded_tokens * _K3_HIDDEN_SIZE * 2,  # BF16 GEMM2 output
    )
    return sum(_align_up(size) for size in region_sizes)


def _cpp_function_body(
    signature_fragment: str, *, after_fragment: str | None = None
) -> str:
    """Extract one C++ body for focused source-policy assertions."""
    source = _LAUNCHER_SOURCE.read_text(encoding="utf-8")
    search_start = 0 if after_fragment is None else source.index(after_fragment)
    signature_start = source.index(signature_fragment, search_start)
    body_start = source.index("{", signature_start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start : index + 1]
    raise AssertionError(f"unterminated C++ body for {signature_fragment!r}")


def test_k3_fallback_resolves_on_full_tile_ladder_before_cap():
    selected = _selected_tiles(
        _FP4_TILE_LADDER,
        num_tokens=_K3_NUM_TOKENS,
        top_k=_K3_TOP_K,
        num_experts=_K3_NUM_EXPERTS,
    )

    assert selected == {128, 256}
    assert min(selected) == 128
    assert {tile for tile in selected if tile <= 128} == {128}

    # Applying max_tile_n=128 to the supported ladder before resolving the
    # fallback changes the center and silently regresses the fallback to 64.
    prefiltered = tuple(tile for tile in _FP4_TILE_LADDER if tile <= 128)
    assert _selected_tiles(
        prefiltered,
        num_tokens=_K3_NUM_TOKENS,
        top_k=_K3_TOP_K,
        num_experts=_K3_NUM_EXPERTS,
    ) == {64, 128}


def test_k3_three_gib_requires_tile_128_but_four_gib_covers_tile_256():
    tile_128_bytes = _k3_workspace_bytes(128)
    tile_256_bytes = _k3_workspace_bytes(256)

    assert tile_128_bytes == 2_846_167_040
    assert tile_256_bytes == 3_713_148_928
    assert tile_128_bytes < _THREE_GIB < tile_256_bytes
    assert tile_256_bytes < _FOUR_GIB
    assert _FOUR_GIB - tile_256_bytes > 512 * 1024**2


def test_native_cap_checks_explicit_tactic_before_full_ladder_fallback():
    body = _cpp_function_body("resolveCappedFp4TileAndConfig(")

    explicit_cap_check = body.index("requested_tile <= max_tile_n.value()")
    full_ladder_resolution = body.index("resolveMoeTileAndConfig(")
    resolved_cap_check = body.index(
        "TVM_FFI_ICHECK_LE(resolved.first, max_tile_n.value())"
    )

    assert explicit_cap_check < full_ladder_resolution < resolved_cap_check
    assert "config_index, supported_tile_nums" in body


def test_python_cap_rejects_tile_256_only_when_policy_is_128():
    from sglang.kernels.ops.moe.trtllm_gen_moe import _validate_tactic_cap

    assert _validate_tactic_cap((-1, -1), 128) == (-1, -1)
    assert _validate_tactic_cap((256, 17), 256) == (256, 17)

    for tactic in ((256, 17), (256, -1)):
        try:
            _validate_tactic_cap(tactic, 128)
        except ValueError as error:
            assert "tile_N=256 exceeds max_tile_n=128" in str(error)
        else:
            raise AssertionError(f"cap accepted prohibited tactic {tactic}")


def test_native_launcher_candidate_enumeration_honors_cap():
    body = _cpp_function_body("Array<Tensor> trtllm_fp4_block_scale_moe_workspace(")

    resolve = body.index("resolveCappedFp4TileAndConfig(")
    candidate_loop = body.index("for (int32_t curr_tile_N : mSupportedTileN)")
    cap_filter = body.index(
        "max_tile_n.has_value() && curr_tile_N > max_tile_n.value()",
        candidate_loop,
    )
    launcher_construction = body.index(
        "std::make_unique<FP4BlockScaleLauncher>", candidate_loop
    )

    assert resolve < candidate_loop < cap_filter < launcher_construction


def test_native_capacity_check_precedes_any_workspace_kernel_launch():
    source = _LAUNCHER_SOURCE.read_text(encoding="utf-8")
    prepare_body = _cpp_function_body("void prepare_external_workspace(")
    run_body = _cpp_function_body(
        "Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true,",
        after_fragment="Optional<TensorView> workspace_arena_;",
    )

    assert "constexpr int64_t kFp4WorkspaceAlignment = 256;" in source
    assert "arena.numel(), workspace_layout_.required_bytes" in prepare_body
    assert "validateExpectedFp4WorkspaceLayout(" in prepare_body
    workspace_prepare = run_body.index("prepare_external_workspace(moe_tactic)")
    first_kernel_launch = run_body.index("routing_runner.run(")
    assert workspace_prepare < first_kernel_launch
