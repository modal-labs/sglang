from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.fla import chunk_delta_h, fused_norm_gate
from sglang.kernels.ops.mamba import causal_conv1d_triton
from sglang.kernels.ops.memory import common as memory_common
from sglang.kernels.ops.speculative import fused_kv_materialize
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _jit_does_not_specialize(kernel, arg_name: str) -> bool:
    while not hasattr(kernel, "do_not_specialize") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    assert hasattr(kernel, "do_not_specialize")
    arg_index = kernel.arg_names.index(arg_name)
    return arg_name in kernel.do_not_specialize or arg_index in kernel.do_not_specialize


@pytest.mark.parametrize(
    ("kernel", "arg_names"),
    [
        (
            chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            ("T", "stride_init_state"),
        ),
        (memory_common.get_last_loc_kernel, ("num_tokens",)),
        (memory_common._get_last_loc_safe_kernel, ("num_tokens",)),
        (
            causal_conv1d_triton._causal_conv1d_fwd_kernel,
            ("seqlen", "stride_o_token"),
        ),
        (
            fused_kv_materialize._fused_norm_rope_kernel_stacked,
            ("total_ctx", "k_out_stride_layer", "v_out_stride_layer"),
        ),
        (fused_norm_gate.layer_norm_gated_fwd_kernel, ("T",)),
    ],
)
def test_k3_runtime_shape_scalars_do_not_create_jit_variants(kernel, arg_names):
    assert all(_jit_does_not_specialize(kernel, name) for name in arg_names)


def test_fused_kv_precompile_uses_compile_only_live_geometry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fused_kv_materialize._fused_norm_rope_kernel_stacked,
        "warmup",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    helper = object.__new__(fused_kv_materialize.FusedKVMaterializeHelper)
    helper.n_layers = 6
    helper.num_kv_heads = 8
    helper.head_dim = 128
    helper.kv_size = 1024
    helper.layer_out_dim = 2048
    helper.rotary_dim = 128
    helper.device = torch.device("cpu")
    helper.max_position_hint = 0
    helper._reserved_rope_cache_len = 1
    helper.flat_kv_weight_t = torch.empty((1, 6 * 2048), dtype=torch.bfloat16)
    helper.k_norm_weights = torch.empty((6, 128), dtype=torch.bfloat16)
    helper.eps_values = torch.empty((6,), dtype=torch.float32)
    helper.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty((1, 128)))

    helper.precompile()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].shape == (1, 6, 2048)
    assert args[5].shape == (6, 1, 8, 128)
    assert args[7:9] == (12288, 2048)
    assert args[11:14] == (1024, 1024, 128)
    assert kwargs["grid"] == (1, 8, 6)


def test_paged_allocator_uses_one_config_derived_batch_bound():
    allocator = object.__new__(PagedTokenToKVPoolAllocator)
    allocator._triton_batch_size_upper_bound = None

    allocator.set_triton_batch_size_upper_bound(17)

    assert allocator.triton_batch_size_upper_bound(1) == 32
    assert allocator.triton_batch_size_upper_bound(4) == 32
    assert allocator.triton_batch_size_upper_bound(16) == 32
    assert allocator.triton_batch_size_upper_bound(17) == 32
    allocator.set_triton_batch_size_upper_bound(8)
    assert allocator.triton_batch_size_upper_bound(17) == 32
    with pytest.raises(RuntimeError, match="runtime batch exceeds"):
        allocator.triton_batch_size_upper_bound(33)


def test_assign_uses_req_pool_capacity_instead_of_runtime_batch(monkeypatch):
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args):
                launches.append(SimpleNamespace(grid=grid, args=args))

            return launch

    monkeypatch.setattr(allocation, "_is_cpu", False)
    monkeypatch.setattr(allocation, "assign_req_to_token_pool", FakeKernel())

    req_to_token = torch.empty((33, 128), dtype=torch.int32)
    allocation.assign_req_to_token_pool_func(
        req_pool_indices=torch.empty(4, dtype=torch.int64),
        req_to_token=req_to_token,
        start_offset=torch.empty(4, dtype=torch.int64),
        end_offset=torch.empty(4, dtype=torch.int64),
        out_cache_loc=torch.empty(4, dtype=torch.int64),
        batch_size=4,
    )

    assert launches[0].grid == (4,)
    assert launches[0].args[-1] == 32


def test_assign_rejects_batch_larger_than_req_pool(monkeypatch):
    monkeypatch.setattr(allocation, "_is_cpu", False)

    with pytest.raises(RuntimeError, match="exceeds req-to-token pool capacity"):
        allocation.assign_req_to_token_pool_func(
            req_pool_indices=torch.empty(5, dtype=torch.int64),
            req_to_token=torch.empty((5, 128), dtype=torch.int32),
            start_offset=torch.empty(5, dtype=torch.int64),
            end_offset=torch.empty(5, dtype=torch.int64),
            out_cache_loc=torch.empty(5, dtype=torch.int64),
            batch_size=5,
        )
