from __future__ import annotations

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers.k3_target_fp8 import (
    K3TargetFP8Linear,
    K3TargetFP8MemoryStats,
    K3TargetFP8SourceGroup,
    K3TargetFP8State,
    kda_qkvg_role,
    moe_front_role,
    rebind_weight_aliases,
)


class TestK3TargetFP8Accounting(unittest.TestCase):
    def test_exact_tensor_static_wide_replacement_bytes(self):
        front_layers = 92
        kda_layers = 69
        k = 7168
        front_weights = front_layers * 6016 * k
        qkvg_weights = kda_layers * 6144 * k
        weights = front_weights + qkvg_weights
        linears = front_layers + kda_layers
        stats = K3TargetFP8MemoryStats(
            scope="wide",
            representation="tensor_static",
            staged_source_bytes=weights * 2,
            replaced_source_bytes=weights * 2,
            final_weight_bytes=weights,
            final_scale_bytes=linears * 8,
            replaced_linears=linears,
        )

        self.assertEqual(front_weights * 2, 7_934_574_592)
        self.assertEqual(qkvg_weights * 2, 6_077_546_496)
        self.assertEqual(stats.final_scale_bytes, 1_288)
        self.assertEqual(stats.saved_bytes, 7_006_059_256)
        self.assertAlmostEqual(stats.saved_bytes / 2**30, 6.524901144206524)

    def test_exact_channel_static_front_bytes(self):
        layers = 92
        n = 6016
        k = 7168
        weights = layers * n * k
        stats = K3TargetFP8MemoryStats(
            scope="front",
            representation="channel_static",
            staged_source_bytes=weights * 2,
            replaced_source_bytes=weights * 2,
            final_weight_bytes=weights,
            final_scale_bytes=layers * (n * 4 + 4),
            replaced_linears=layers,
        )

        self.assertEqual(stats.final_bytes // layers, 43_146_756)
        self.assertEqual(stats.saved_bytes // layers, 43_098_620)

    def test_component_aliases_no_longer_hold_bf16_storage(self):
        modules = [
            nn.Linear(4, 2, bias=False),
            nn.Linear(4, 1, bias=False),
            nn.Linear(4, 3, bias=False),
        ]
        for module in modules:
            module.weight.requires_grad_(False)
        old_storages = {
            module.weight.untyped_storage().data_ptr() for module in modules
        }
        replacement = torch.empty((6, 4), dtype=torch.float8_e4m3fn)

        rebind_weight_aliases(modules, [2, 1, 3], replacement)

        replacement_storage = replacement.untyped_storage().data_ptr()
        self.assertTrue(
            all(
                module.weight.untyped_storage().data_ptr() == replacement_storage
                for module in modules
            )
        )
        self.assertTrue(old_storages.isdisjoint({replacement_storage}))
        self.assertEqual(
            [tuple(module.weight.shape) for module in modules],
            [(2, 4), (1, 4), (3, 4)],
        )
        self.assertTrue(
            all(module.weight.dtype == torch.float8_e4m3fn for module in modules)
        )

    def test_alias_rows_must_cover_replacement(self):
        modules = [nn.Linear(4, 2, bias=False)]
        with self.assertRaisesRegex(ValueError, "do not cover"):
            rebind_weight_aliases(
                modules,
                [1],
                torch.empty((2, 4), dtype=torch.float8_e4m3fn),
            )


class TestK3TargetFP8Linear(unittest.TestCase):
    def test_cpu_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA-resident source"):
            K3TargetFP8Linear.from_bf16(
                torch.empty((16, 16), dtype=torch.bfloat16),
                representation="tensor_static",
            )

    def test_runtime_and_logical_weight_share_storage(self):
        runtime_weight = torch.empty((4, 6), dtype=torch.float8_e4m3fn)
        linear = K3TargetFP8Linear(
            runtime_weight,
            torch.ones(1, dtype=torch.float32),
            representation="tensor_static",
        )

        self.assertEqual(tuple(linear.weight.shape), (4, 6))
        self.assertEqual(tuple(linear.logical_weight.shape), (6, 4))
        self.assertEqual(
            linear.weight.untyped_storage().data_ptr(),
            linear.logical_weight.untyped_storage().data_ptr(),
        )
        self.assertEqual(linear.resident_scale_bytes, 8)
        self.assertFalse(hasattr(linear, "quant_method"))

    def test_empty_input_and_sharded_save_fail_closed(self):
        linear = K3TargetFP8Linear(
            torch.empty((16, 32), dtype=torch.float8_e4m3fn),
            torch.ones(1, dtype=torch.float32),
            representation="tensor_static",
        )
        output = linear(torch.empty((0, 16), dtype=torch.bfloat16))
        self.assertEqual(tuple(output.shape), (0, 32))
        self.assertEqual(output.dtype, torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "cannot be saved/reloaded"):
            linear.state_dict()

    def test_channel_representation_scale_accounting(self):
        linear = K3TargetFP8Linear(
            torch.empty((4, 6), dtype=torch.float8_e4m3fn),
            torch.ones((1, 6), dtype=torch.float32),
            representation="channel_static",
        )
        self.assertEqual(linear.resident_scale_bytes, 28)
        self.assertFalse(linear.supports_prequantized_static_input)
        with self.assertRaisesRegex(RuntimeError, "requires tensor_static"):
            linear.forward_prequantized(
                torch.empty((2, 4), dtype=torch.float8_e4m3fn),
                output_dtype=torch.bfloat16,
            )

    @patch("sglang.srt.layers.k3_target_fp8.torch._scaled_mm")
    def test_tensor_static_prequantized_input_bypasses_quant(self, scaled_mm):
        runtime_weight = torch.empty((4, 6), dtype=torch.float8_e4m3fn)
        weight_scale = torch.full((1,), 0.25, dtype=torch.float32)
        linear = K3TargetFP8Linear(
            runtime_weight,
            weight_scale,
            representation="tensor_static",
        )
        qinput = torch.empty((2, 3, 4), dtype=torch.float8_e4m3fn)
        scaled_mm.return_value = torch.empty((6, 6), dtype=torch.bfloat16)

        output = linear.forward_prequantized(
            qinput,
            output_dtype=torch.bfloat16,
        )

        self.assertTrue(linear.supports_prequantized_static_input)
        self.assertEqual(tuple(output.shape), (2, 3, 6))
        self.assertEqual(output.dtype, torch.bfloat16)
        scaled_mm.assert_called_once()
        args, kwargs = scaled_mm.call_args
        self.assertEqual(tuple(args[0].shape), (6, 4))
        self.assertEqual(args[0].data_ptr(), qinput.data_ptr())
        self.assertIs(args[1], linear.weight)
        self.assertIs(kwargs["scale_a"], linear.input_scale)
        self.assertIs(kwargs["scale_b"], linear.weight_scale)
        self.assertIs(kwargs["out_dtype"], torch.bfloat16)

    def test_prequantized_input_contract_fails_closed(self):
        linear = K3TargetFP8Linear(
            torch.empty((4, 6), dtype=torch.float8_e4m3fn),
            torch.ones(1, dtype=torch.float32),
            representation="tensor_static",
        )
        with self.assertRaisesRegex(TypeError, "must be float8_e4m3fn"):
            linear.forward_prequantized(
                torch.empty((2, 4), dtype=torch.bfloat16),
                output_dtype=torch.bfloat16,
            )
        with self.assertRaisesRegex(ValueError, "incompatible shape"):
            linear.forward_prequantized(
                torch.empty((2, 8), dtype=torch.float8_e4m3fn),
                output_dtype=torch.bfloat16,
            )
        with self.assertRaisesRegex(TypeError, "must be BF16/FP16"):
            linear.forward_prequantized(
                torch.empty((2, 4), dtype=torch.float8_e4m3fn),
                output_dtype=torch.float32,
            )


class TestK3TargetFP8Scope(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "SGLANG_K3_TARGET_DENSE_FP8": "front",
            "SGLANG_K3_TARGET_DENSE_FP8_REPRESENTATION": "channel_static",
        },
    )
    def test_from_env_reads_registered_scope_and_representation(self):
        state = K3TargetFP8State.from_env()
        self.assertEqual(state.scope, "front")
        self.assertEqual(state.representation, "channel_static")
        self.assertTrue(state.role_enabled(moe_front_role()))
        self.assertFalse(state.role_enabled(kda_qkvg_role()))

    @patch("torch.cuda.is_available", return_value=False)
    def test_wide_scope_is_parseable_without_cuda(self, _):
        state = K3TargetFP8State(" WIDE ", " TENSOR_STATIC ")
        self.assertTrue(state.enabled)
        self.assertTrue(state.role_enabled(moe_front_role()))
        self.assertTrue(state.role_enabled(kda_qkvg_role()))
        self.assertIsNone(state.new_source_group(moe_front_role(), 1))

    def test_invalid_scope_and_representation_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "TARGET_DENSE_FP8"):
            K3TargetFP8State("all", "tensor_static")
        with self.assertRaisesRegex(ValueError, "REPRESENTATION"):
            K3TargetFP8State("front", "mxfp8")

    @patch("sglang.srt.layers.k3_target_fp8.torch.cuda.MemPool")
    @patch("sglang.srt.layers.k3_target_fp8.torch.cuda.is_available")
    def test_source_pool_is_private_and_ids_are_unique(
        self,
        is_available,
        mem_pool,
    ):
        is_available.return_value = True
        mem_pool.return_value = MagicMock()

        state = K3TargetFP8State("wide", "tensor_static")
        group = state.new_source_group(moe_front_role(), 1)

        self.assertIsNotNone(group)
        mem_pool.assert_called_once_with(use_on_oom=False)
        with self.assertRaisesRegex(RuntimeError, "CPU offload"):
            group.stage_linear_weight(nn.Linear(4, 4, bias=False))
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            state.new_source_group(moe_front_role(), 1)
        with self.assertRaisesRegex(RuntimeError, "live BF16 source pools"):
            state.finalize()

    def test_source_group_release_avoids_gc_and_default_allocator_flush(self):
        state = K3TargetFP8State("wide", "tensor_static")
        pool = MagicMock()
        pool.use_count.return_value = 1
        group = K3TargetFP8SourceGroup(
            owner=state,
            role=moe_front_role(),
            identifier=1,
            pool=pool,
        )

        with (
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.synchronize"
            ) as synchronize,
            patch("sglang.srt.layers.k3_target_fp8.gc.collect") as collect,
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            group.release()
            group.release()

        synchronize.assert_called_once_with()
        pool.use_count.assert_called_once_with()
        collect.assert_not_called()
        empty_cache.assert_not_called()
        self.assertTrue(group.released)
        self.assertIsNone(group._pool)

    def test_source_group_release_gc_fallback_recovers_delayed_pool_reference(self):
        state = K3TargetFP8State("wide", "tensor_static")
        pool = MagicMock()
        pool.use_count.side_effect = (2, 1)
        group = K3TargetFP8SourceGroup(
            owner=state,
            role=kda_qkvg_role(),
            identifier=1,
            pool=pool,
        )

        with (
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.synchronize"
            ) as synchronize,
            patch("sglang.srt.layers.k3_target_fp8.gc.collect") as collect,
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            group.release()

        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(pool.use_count.call_count, 2)
        collect.assert_called_once_with()
        empty_cache.assert_not_called()
        self.assertTrue(group.released)
        self.assertIsNone(group._pool)

    def test_source_group_release_retains_pool_when_gc_fallback_fails(self):
        state = K3TargetFP8State("wide", "tensor_static")
        pool = MagicMock()
        pool.use_count.return_value = 2
        group = K3TargetFP8SourceGroup(
            owner=state,
            role=kda_qkvg_role(),
            identifier=1,
            pool=pool,
        )

        with (
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.synchronize"
            ) as synchronize,
            patch("sglang.srt.layers.k3_target_fp8.gc.collect") as collect,
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.empty_cache"
            ) as empty_cache,
            self.assertRaisesRegex(RuntimeError, "live users"),
        ):
            group.release()

        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(pool.use_count.call_count, 2)
        collect.assert_called_once_with()
        empty_cache.assert_not_called()
        self.assertFalse(group.released)
        self.assertIs(group._pool, pool)

    def test_finalize_performs_one_cleanup_and_logs_aggregate_elapsed_time(self):
        state = K3TargetFP8State("wide", "tensor_static")

        with (
            patch(
                "sglang.srt.layers.k3_target_fp8.envs."
                "SGLANG_K3_TARGET_DENSE_FP8_MEMORY_DIAGNOSTICS.get",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.k3_target_fp8.time.perf_counter",
                side_effect=(10.0, 13.25),
            ),
            patch("sglang.srt.layers.k3_target_fp8.gc.collect") as collect,
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.empty_cache"
            ) as empty_cache,
            patch(
                "sglang.srt.layers.k3_target_fp8.torch.cuda.synchronize"
            ) as synchronize,
            patch("sglang.srt.layers.k3_target_fp8.rank0_log") as rank0_log,
        ):
            state.begin_conversion()
            state.finalize()

        collect.assert_called_once_with()
        empty_cache.assert_called_once_with()
        synchronize.assert_called_once_with()
        rank0_log.assert_called_once()
        self.assertIn(
            "conversion_elapsed_seconds=3.250",
            rank0_log.call_args.args[0],
        )

    def test_config_derived_layer_ids_fail_closed(self):
        state = K3TargetFP8State("wide", "tensor_static")
        state._created_ids_by_role = defaultdict(
            set,
            {
                moe_front_role(): {1, 2},
                kda_qkvg_role(): {1},
            },
        )
        with self.assertRaisesRegex(RuntimeError, "model config"):
            state.configure_expected_layer_ids(
                local_front_ids=[1, 2],
                local_kda_ids=[1, 3],
                configured_kda_count=69,
            )

    def test_checkpoint_components_must_all_be_loaded(self):
        state = K3TargetFP8State("wide", "tensor_static")
        state._created_ids_by_role = defaultdict(
            set,
            {
                moe_front_role(): {1},
                kda_qkvg_role(): {1},
            },
        )
        state.configure_expected_layer_ids(
            local_front_ids=[1],
            local_kda_ids=[1],
            configured_kda_count=1,
        )
        state.begin_checkpoint_load()
        for component in ("shared_gate", "shared_up", "router", "latent_down"):
            state.mark_checkpoint_component(
                moe_front_role(),
                1,
                component,
            )
        for component in ("q", "k", "v"):
            state.mark_checkpoint_component(
                kda_qkvg_role(),
                1,
                component,
            )

        with self.assertRaisesRegex(RuntimeError, "kda_qkvg:1"):
            state.finish_checkpoint_load()

    def test_range_diagnostics_preserve_component_boundaries(self):
        state = K3TargetFP8State("front", "tensor_static")
        source = torch.tensor(
            [[-1.0, 0.5], [2.0, -0.5], [0.25, -3.0]],
            dtype=torch.bfloat16,
        )
        result = state._source_range_diagnostic(
            source,
            moe_front_role(),
            component_names=("shared", "router"),
            component_rows=(2, 1),
        )

        self.assertEqual(result["source_absmax"], 3.0)
        self.assertEqual(
            [component["absmax"] for component in result["components"]],
            [2.0, 3.0],
        )


if __name__ == "__main__":
    unittest.main()
