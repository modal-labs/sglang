import gc
import os
import tempfile
import weakref
from contextlib import contextmanager, nullcontext
from copy import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.nn import Parameter

from sglang.srt.layers.quantization import fp8
from sglang.srt.model_loader import loader, weight_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_source_allocation_uses_one_thread_local_private_pool():
    sentinel = object()
    pool = Mock()
    active_pool = []

    @contextmanager
    def use_mem_pool(selected_pool, *, device):
        assert selected_pool is pool
        assert device == 3
        active_pool.append(selected_pool)
        try:
            yield
        finally:
            active_pool.pop()

    config = fp8.Fp8Config(use_online_weight_staging_pool=True)
    with (
        patch.object(fp8, "_is_cuda", True),
        patch.object(fp8.torch.cuda, "init") as cuda_init,
        patch.object(fp8.torch.cuda, "current_device", return_value=3),
        patch.object(fp8.torch.cuda, "get_allocator_backend", return_value="native"),
        patch.object(fp8.torch.cuda, "empty_cache") as empty_cache,
        patch.object(fp8.torch.cuda, "MemPool", return_value=pool) as mem_pool,
        patch.object(fp8.torch.cuda, "use_mem_pool", side_effect=use_mem_pool),
        patch.object(
            fp8.torch,
            "empty",
            side_effect=lambda *args, **kwargs: (
                sentinel if active_pool == [pool] else pytest.fail("wrong pool")
            ),
        ) as empty,
    ):
        config.begin_online_weight_staging()
        first = config.allocate_online_source_weight(4, 8, torch.bfloat16)
        second = config.allocate_online_source_weight(4, 8, torch.bfloat16)
        config.abort_online_weight_staging()

    assert first is sentinel
    assert second is sentinel
    cuda_init.assert_called_once_with()
    empty_cache.assert_called_once_with()
    mem_pool.assert_called_once_with(use_on_oom=False, no_split=False)
    assert empty.call_count == 2
    empty.assert_called_with(4, 8, dtype=torch.bfloat16)


def test_source_pool_context_exits_when_allocation_fails():
    pool = Mock()

    @contextmanager
    def use_mem_pool(*args, **kwargs):
        yield

    config = fp8.Fp8Config(use_online_weight_staging_pool=True)
    with (
        patch.object(fp8, "_is_cuda", True),
        patch.object(fp8.torch.cuda, "init"),
        patch.object(fp8.torch.cuda, "current_device", return_value=0),
        patch.object(fp8.torch.cuda, "get_allocator_backend", return_value="native"),
        patch.object(fp8.torch.cuda, "MemPool", return_value=pool),
        patch.object(fp8.torch.cuda, "use_mem_pool", side_effect=use_mem_pool),
        patch.object(fp8.torch, "empty", side_effect=RuntimeError("allocation failed")),
        pytest.raises(RuntimeError, match="allocation failed"),
    ):
        config.begin_online_weight_staging()
        config.allocate_online_source_weight(4, 8, torch.bfloat16)


def test_source_pool_rejects_cuda_malloc_async():
    config = fp8.Fp8Config(use_online_weight_staging_pool=True)
    with (
        patch.object(fp8, "_is_cuda", True),
        patch.object(fp8.torch.cuda, "init"),
        patch.object(
            fp8.torch.cuda,
            "get_allocator_backend",
            return_value="cudaMallocAsync",
        ),
        pytest.raises(RuntimeError, match="native CUDA allocator"),
    ):
        config.begin_online_weight_staging()
        config.allocate_online_source_weight(4, 8, torch.bfloat16)


def test_shallow_config_copies_share_one_transient_pool_manager():
    config = fp8.Fp8Config(use_online_weight_staging_pool=True)
    copied = copy(config)

    assert copied._online_weight_staging is config._online_weight_staging


def test_online_static_uses_unit_activation_scale_and_fused_weight_quant():
    config = fp8.Fp8Config(
        activation_scheme="static",
        use_online_weight_staging_pool=True,
    )
    method = object.__new__(fp8.Fp8LinearMethod)
    method.quant_config = config
    method.block_quant = False
    method.is_checkpoint_fp8_serialized = False
    method.cutlass_fp8_supported = False
    method.use_marlin = False
    method.use_online_weight_staging_pool = True
    method.online_static = True

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)),
    )
    source_parameter_ref = weakref.ref(layer.weight)
    quantized = torch.zeros((2, 4), dtype=torch.uint8)
    weight_scale = torch.tensor([0.25], dtype=torch.float32)

    with (
        patch.object(fp8, "_use_aiter", False),
        patch.object(fp8, "_is_cuda", True),
        patch.object(
            fp8,
            "scaled_fp8_quant",
            return_value=(quantized, weight_scale),
        ) as scaled_fp8_quant,
    ):
        method.process_weights_after_loading(layer)

    scaled_fp8_quant.assert_called_once()
    scaled_fp8_quant.reset_mock()
    gc.collect()
    assert layer.weight.shape == (4, 2)
    assert layer.weight_scale.item() == pytest.approx(0.25)
    assert layer.input_scale.item() == pytest.approx(1.0)
    assert source_parameter_ref() is None


def test_online_static_selects_per_tensor_backend():
    config = fp8.Fp8Config(activation_scheme="static")
    with (
        patch.object(fp8, "_is_cuda", True),
        patch.object(fp8, "cutlass_fp8_supported", return_value=True),
        patch.object(fp8, "can_auto_enable_marlin_fp8", return_value=False),
        patch.object(fp8, "get_bool_env_var", return_value=False),
        patch.object(fp8, "dispatch_w8a8_block_fp8_linear", return_value=Mock()),
    ):
        method = fp8.Fp8LinearMethod(config)

    assert method.online_static
    assert not method.cutlass_fp8_supported


def test_draft_fp8_quant_config_is_online_only_and_preserves_scheme():
    with tempfile.TemporaryDirectory() as model_path:
        model_config = SimpleNamespace(
            quantization="fp8",
            is_draft_model=True,
            draft_fp8_activation_scheme="static",
            hf_config=SimpleNamespace(),
            model_path=model_path,
            revision=None,
        )
        load_config = SimpleNamespace(
            model_loader_extra_config=None,
            download_dir=None,
        )

        with patch.object(
            weight_utils, "get_quantization_config", return_value=fp8.Fp8Config
        ):
            config = weight_utils.get_quant_config(model_config, load_config, {})

    assert config.activation_scheme == "static"
    assert not config.is_checkpoint_fp8_serialized
    assert config.use_online_weight_staging_pool


def test_source_pool_teardown_releases_pool_without_global_empty_cache():
    class FakePool:
        def use_count(self):
            return 1

        def snapshot(self, include_traces=True):
            assert not include_traces
            return [
                {
                    "total_size": 64,
                    "allocated_size": 0,
                    "active_size": 0,
                    "is_expandable": False,
                }
            ]

    config = fp8.Fp8Config(use_online_weight_staging_pool=True)
    pool = FakePool()
    pool_ref = weakref.ref(pool)
    with (
        patch.object(fp8, "_is_cuda", True),
        patch.object(fp8.torch.cuda, "device", return_value=nullcontext()),
        patch.object(fp8.torch.cuda, "synchronize") as synchronize,
        patch.object(
            fp8,
            "_cuda_memory_snapshot",
            side_effect=[
                {"free_bytes": 10},
                {"free_bytes": 74},
            ],
        ),
        patch.object(fp8.torch.cuda, "empty_cache") as empty_cache,
    ):
        config.begin_online_weight_staging()
        config._online_weight_staging.pool = pool
        config._online_weight_staging.device = 0
        del pool
        config.finish_online_weight_staging()

    assert pool_ref() is None
    assert config._online_weight_staging.pool is None
    assert synchronize.call_count == 2
    empty_cache.assert_not_called()


@pytest.mark.parametrize(
    ("allocated_size", "use_count", "is_expandable"),
    [
        (1, 1, False),
        (0, 2, False),
        (0, 1, True),
    ],
)
def test_source_pool_teardown_rejects_unsafe_ownership(
    allocated_size, use_count, is_expandable
):
    pool = Mock()
    pool.use_count.return_value = use_count
    pool.snapshot.return_value = [
        {
            "total_size": 64,
            "allocated_size": allocated_size,
            "active_size": allocated_size,
            "is_expandable": is_expandable,
        }
    ]
    config = fp8.Fp8Config(use_online_weight_staging_pool=True)
    with (
        patch.object(fp8, "_is_cuda", True),
        patch.object(fp8.torch.cuda, "device", return_value=nullcontext()),
        patch.object(fp8.torch.cuda, "synchronize"),
        pytest.raises(RuntimeError, match="staging pool still owns live sources"),
    ):
        config.begin_online_weight_staging()
        config._online_weight_staging.pool = pool
        config._online_weight_staging.device = 0
        config.finish_online_weight_staging()

    config.abort_online_weight_staging()
    assert config._online_weight_staging.pool is None


def test_memory_diagnostics_report_pool_recovery_once():
    before = {
        "free_bytes": 100,
        "total_bytes": 1000,
        "allocated_bytes": 300,
        "reserved_bytes": 400,
    }
    before_destroy = {
        "free_bytes": 20,
        "total_bytes": 1000,
        "allocated_bytes": 320,
        "reserved_bytes": 500,
    }
    after_destroy = {
        "free_bytes": 84,
        "total_bytes": 1000,
        "allocated_bytes": 320,
        "reserved_bytes": 436,
    }

    pool = Mock()
    pool.use_count.return_value = 1
    pool.snapshot.return_value = [
        {
            "total_size": 64,
            "allocated_size": 0,
            "active_size": 0,
            "is_expandable": False,
        }
    ]
    with (
        patch.dict(
            os.environ,
            {"SGLANG_DRAFT_FP8_MEMORY_DIAGNOSTICS": "1"},
        ),
        patch.object(fp8, "_is_cuda", True),
        patch.object(
            fp8,
            "_cuda_memory_snapshot",
            side_effect=[before_destroy, after_destroy],
        ),
        patch.object(fp8.torch.cuda, "device", return_value=nullcontext()),
        patch.object(fp8.torch.cuda, "synchronize"),
        patch.object(fp8, "log_info_on_rank0") as log_info,
    ):
        config = fp8.Fp8Config(use_online_weight_staging_pool=True)
        config.begin_online_weight_staging()
        config._online_weight_staging.pool = pool
        config._online_weight_staging.device = 0
        config._online_weight_staging.before = before
        config.record_online_quantization(
            source_bytes=64,
            destination=torch.empty(32, dtype=torch.uint8),
            scale=torch.empty(1, dtype=torch.float32),
        )
        config.finish_online_weight_staging()

    log_info.assert_called_once()
    message = log_info.call_args.args[1]
    assert "layers=1" in message
    assert "source_bytes=64" in message
    assert "fp8_destination_bytes=32" in message
    assert "scale_bytes=4" in message
    assert "pool_reserved_bytes=64" in message
    assert "pool_free_recovered=+64" in message


def test_loader_abort_preserves_original_error():
    quant_config = SimpleNamespace(
        begin_online_weight_staging=Mock(),
        finish_online_weight_staging=Mock(),
        abort_online_weight_staging=Mock(side_effect=RuntimeError("abort failed")),
    )
    with (
        pytest.raises(ValueError, match="original failure"),
        loader._online_quantization_staging(quant_config),
    ):
        raise ValueError("original failure")

    quant_config.begin_online_weight_staging.assert_called_once_with()
    quant_config.abort_online_weight_staging.assert_called_once_with()
    quant_config.finish_online_weight_staging.assert_not_called()
