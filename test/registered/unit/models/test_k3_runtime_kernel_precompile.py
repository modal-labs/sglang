from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.kernels.jit import utils as jit_utils
from sglang.kernels.ops.kvcache import kv_indices, mla_buffer
from sglang.kernels.ops.memory import allocator, common
from sglang.srt.mem_cache import allocation, mamba_slot_fused
from sglang.srt.model_executor import k3_runtime_kernel_precompile as precompile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner(*, req_width=4096, max_running_requests=32):
    req_to_token = torch.empty(
        (max_running_requests + 1, req_width),
        dtype=torch.int32,
    )
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_pages=torch.empty((128,), dtype=torch.int64),
            page_size=64,
            triton_batch_size_upper_bound=lambda _batch_size: 32,
        ),
        max_running_requests=max_running_requests,
    )


def test_request_path_precompile_uses_compile_only_warmup_and_stable_bound(
    monkeypatch,
):
    alloc_warmup = Mock()
    assign_warmup = Mock()
    last_loc_warmup = Mock()
    monkeypatch.setattr(
        allocator.alloc_extend_kernel,
        "warmup",
        alloc_warmup,
    )
    monkeypatch.setattr(
        allocation.assign_req_to_token_pool,
        "warmup",
        assign_warmup,
    )
    monkeypatch.setattr(
        common.get_last_loc_kernel,
        "warmup",
        last_loc_warmup,
    )

    compiled = precompile._precompile_request_table_kernels(_runner())

    assert compiled == (
        "alloc_extend_kernel",
        "assign_req_to_token_pool",
        "get_last_loc_kernel",
    )
    assert alloc_warmup.call_count == 2
    assert {call.args[0].dtype for call in alloc_warmup.call_args_list} == {
        torch.int32,
        torch.int64,
    }
    assert {call.args[-2] for call in alloc_warmup.call_args_list} == {32}
    assert assign_warmup.call_args.args[-1] == 32
    assert all(call.kwargs["grid"] == (1,) for call in alloc_warmup.call_args_list)
    assert assign_warmup.call_args.kwargs["grid"] == (1,)
    assert last_loc_warmup.call_count == 2
    assert {call.args[2].dtype for call in last_loc_warmup.call_args_list} == {
        torch.int32,
        torch.int64,
    }
    assert all(call.kwargs["grid"] == (1,) for call in last_loc_warmup.call_args_list)


def test_target_precompile_covers_prefix_mla_read_and_mamba_cow(monkeypatch):
    prefix_warmup = Mock()
    mla_warmup = Mock()
    slot_clear_warmup = Mock()
    slot_copy_warmup = Mock()
    monkeypatch.setattr(
        kv_indices.create_chunked_prefix_cache_kv_indices,
        "warmup",
        prefix_warmup,
    )
    monkeypatch.setattr(
        mla_buffer.get_mla_kv_buffer_kernel,
        "warmup",
        mla_warmup,
    )
    monkeypatch.setattr(
        mamba_slot_fused._fused_slot_clear_kernel,
        "warmup",
        slot_clear_warmup,
    )
    monkeypatch.setattr(
        mamba_slot_fused._fused_slot_copy_kernel,
        "warmup",
        slot_copy_warmup,
    )

    desc = SimpleNamespace(
        ptr=torch.empty((3,), dtype=torch.int64),
        feat=torch.empty((3,), dtype=torch.int64),
        layer_stride=torch.empty((3,), dtype=torch.int64),
        slot_stride=torch.empty((3,), dtype=torch.int64),
        num_layers=61,
        max_feat_blocks=2,
    )
    req_pool = SimpleNamespace(
        req_to_token=torch.empty((33, 4096), dtype=torch.int32),
        mamba_pool=SimpleNamespace(
            _should_fuse_slot_ops=lambda: True,
            _conv_slot_desc=desc,
        ),
    )
    kv_buffer = torch.empty((64, 576), dtype=torch.bfloat16)
    kv_pool = SimpleNamespace(
        get_key_buffer=lambda _layer_id: kv_buffer,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
        start_layer=0,
    )
    runner = SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool=kv_pool,
    )

    compiled = precompile._precompile_kimi_hybrid_kernels(runner)

    assert compiled == (
        "create_chunked_prefix_cache_kv_indices",
        "get_mla_kv_buffer_kernel",
        "_fused_slot_clear_kernel",
        "_fused_slot_copy_kernel",
    )
    prefix_warmup.assert_called_once()
    mla_warmup.assert_called_once()
    slot_clear_warmup.assert_called_once()
    slot_copy_warmup.assert_called_once()
    assert slot_clear_warmup.call_args.kwargs["grid"] == (1, 3, 61)
    assert slot_copy_warmup.call_args.kwargs["grid"] == (1, 3, 61)


def test_mla_write_precompile_uses_live_fused_projection_pitch(monkeypatch):
    set_mla_warmup = Mock()
    monkeypatch.setattr(
        mla_buffer.set_mla_kv_buffer_kernel,
        "warmup",
        set_mla_warmup,
    )
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", lambda: True)
    monkeypatch.setattr(
        precompile,
        "get_parallel",
        lambda: SimpleNamespace(attn_dcp_rank=0, attn_dcp_size=1),
    )

    kv_buffer = torch.empty((64, 1, 576), dtype=torch.uint8)
    mla_pool = SimpleNamespace(
        kv_buffer=[kv_buffer],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(q_lora_rank=1536))
    )

    assert precompile._precompile_mla_fused_projection_write(runner, mla_pool) == (
        "set_mla_kv_buffer_kernel",
    )
    assert set_mla_warmup.call_count == 2
    assert [call.args[5] for call in set_mla_warmup.call_args_list] == [512, 2112]
    assert [call.args[6] for call in set_mla_warmup.call_args_list] == [64, 2112]
    assert all(call.kwargs["BLOCK"] == 1024 for call in set_mla_warmup.call_args_list)
    assert all(call.kwargs["grid"] == (1, 1) for call in set_mla_warmup.call_args_list)
    assert all(
        call.kwargs["launch_pdl"] is True for call in set_mla_warmup.call_args_list
    )


def test_tp_precompile_serializes_cold_compile_then_peer_load(monkeypatch):
    tp_group = Mock()
    tp_group.broadcast_object.side_effect = lambda value, src: value
    tp_group.all_gather_object.return_value = [None] * 8
    runner = SimpleNamespace(tp_group=tp_group)
    compile_fn = Mock(return_value=("kernel",))

    monkeypatch.setattr(
        precompile,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_size=8),
    )
    assert precompile._compile_on_rank_zero(runner, compile_fn) == ("kernel",)
    compile_fn.assert_called_once()
    tp_group.broadcast_object.assert_called_once_with(None, src=0)
    tp_group.all_gather_object.assert_called_once_with(None)

    tp_group.reset_mock()
    tp_group.broadcast_object.side_effect = lambda value, src: value
    tp_group.all_gather_object.return_value = [None] * 8
    compile_fn.reset_mock()
    monkeypatch.setattr(
        precompile,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=3, tp_size=8),
    )
    assert precompile._compile_on_rank_zero(runner, compile_fn) == ()
    compile_fn.assert_called_once()
    tp_group.broadcast_object.assert_called_once_with(None, src=0)
    tp_group.all_gather_object.assert_called_once_with(None)


def test_tp_precompile_propagates_rank_zero_failure(monkeypatch):
    tp_group = Mock()
    tp_group.broadcast_object.side_effect = lambda value, src: value
    runner = SimpleNamespace(tp_group=tp_group)
    compile_fn = Mock(side_effect=ValueError("bad compile"))
    monkeypatch.setattr(
        precompile,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_size=8),
    )

    try:
        precompile._compile_on_rank_zero(runner, compile_fn)
    except RuntimeError as exc:
        assert "rank 0" in str(exc)
        assert "ValueError: bad compile" in str(exc)
    else:
        raise AssertionError("rank-zero precompile failure did not propagate")

    tp_group.all_gather_object.assert_not_called()
