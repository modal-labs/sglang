"""Regression tests for multimodal IDs and Engram masking during graph replay."""

import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as dsv4
from sglang.srt.layers.engram import engram_gate
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepseekV41BCGMultimodal(CustomTestCase):
    def test_replay_uses_prepared_image_ids_without_mutating_scheduler_ids(self):
        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                scheduler_ids = torch.tensor([7, dsv4.MM_PAD_SHIFT_VALUE + 1, 9, 0])
                static_ids = scheduler_ids.clone()
                image_id = 42
                layer = SimpleNamespace(forward=None)
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                runner._is_full_backend = False
                runner.is_dsv41_model = True
                runner._qwen_bcg_hc_sidechannel = False
                runner._use_draft_input_embeds = False
                runner.layer_model = layer
                runner.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
                runner._prefill_forward_context = lambda *a, **kw: nullcontext()
                runner.backend = SimpleNamespace(
                    replay=lambda *a, **kw: static_ids.clone()
                )

                def outer_forward(ids, positions, batch):
                    model = SimpleNamespace(
                        vision=object(), config=SimpleNamespace(image_token_id=image_id)
                    )
                    mapped, _ = (
                        dsv4.DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, batch
                        )
                    )
                    if keyword:
                        return layer.forward(
                            input_ids=mapped, positions=positions, forward_batch=batch
                        )
                    return layer.forward(mapped, positions, batch)

                runner.model_runner = SimpleNamespace(
                    pp_group=SimpleNamespace(is_first_rank=True),
                    model=SimpleNamespace(forward=outer_forward),
                )
                live = SimpleNamespace(input_ids=scheduler_ids, mm_input_embeds=None)
                static = SimpleNamespace(
                    input_ids=static_ids,
                    positions=torch.arange(4),
                    mm_input_embeds=None,
                    mm_inputs=None,
                    forward_mode=ForwardMode.EXTEND,
                )
                actual = runner._execute_body_capture(live, static, 4, 3, None)
                torch.testing.assert_close(actual, torch.tensor([7, image_id, 9, 0]))
                self.assertEqual(scheduler_ids[1].item(), dsv4.MM_PAD_SHIFT_VALUE + 1)

    def test_image_engram_mask_does_not_depend_on_capture_batch_mm_metadata(self):
        image_id = 42
        ids = torch.tensor([7, image_id, 9])
        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        kv = torch.ones(3, 4)
        weight = torch.ones(1, 2)

        class Engram:
            layer_hash_index = 0

            def __call__(self, residual, *args, input_ids, **kwargs):
                return engram_gate(
                    residual, kv, weight, weight, 1e-6, 1e-6, input_ids, image_id
                )

        layer = SimpleNamespace(
            engram=Engram(),
            hc_cfg=None,
            forward_hc_pre_from_prev=lambda **kw: kw["state"],
        )
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=1),
            engram_hasher=lambda *a: torch.zeros((3, 1), dtype=torch.long),
            late_layer_start=None,
            decoder_replay_tail=None,
            start_layer=0,
            end_layer=1,
            layers=[layer],
            config=SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=1, image_token_id=image_id
            ),
        )
        expected = engram_gate(hidden, kv, weight, weight, 1e-6, 1e-6)
        self.assertFalse(torch.equal(expected[1], hidden[1]))
        expected[1] = hidden[1]
        for contains_mm in (False, True):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND, contains_mm_inputs=lambda: contains_mm
            )
            with (
                patch.object(dsv4, "is_cp_active", return_value=False),
                patch.object(dsv4, "is_in_breakable_cuda_graph", return_value=False),
                patch.object(
                    dsv4, "get_forward", return_value=SimpleNamespace(sp_active=False)
                ),
                patch.object(dsv4, "check_cuda_graph_backend", return_value=True),
                patch.object(dsv4, "_is_hip", False),
            ):
                actual, _, _ = dsv4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                    model, torch.arange(3), hidden, batch, ids, ids, False, []
                )
            torch.testing.assert_close(actual, expected)


class TestDeepseekV4BCGAttentionOutput(CustomTestCase):
    def run_attention(self, output, real_rows, forward):
        cache_loc = torch.arange(output.shape[0])
        batch = SimpleNamespace(
            global_num_token_non_padded_cpu=real_rows, out_cache_loc=cache_loc
        )
        context = SimpleNamespace(forward_batch=batch, attention_layers=[object()])
        with (
            patch.object(
                dsv4, "get_tc_piecewise_forward_context", return_value=context
            ),
            patch.object(
                dsv4, "get_attn_backend", return_value=SimpleNamespace(forward=forward)
            ),
        ):
            try:
                inspect.unwrap(dsv4.deepseek_v4_attention_with_output)(
                    torch.ones_like(output),
                    torch.ones_like(output),
                    output,
                    0,
                    0,
                    torch.zeros(output.shape[1]),
                    False,
                )
            finally:
                self.assertIs(batch.out_cache_loc, cache_loc)

    def test_backend_output_reuse_avoids_copy_and_clears_padding(self):
        """BCG must not copy an already-written attention result into itself."""
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                output = torch.full((4, 2, 512), float("nan"), dtype=torch.bfloat16)

                def forward(q, out=None, **kwargs):
                    result = out if reuse and out is not None else torch.empty_like(q)
                    return result.fill_(7)

                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    self.run_attention(output, 2, forward)
                torch.testing.assert_close(output[:2], torch.full_like(output[:2], 7))
                torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))
                copies = sum(
                    event.count
                    for event in profile.key_averages()
                    if event.key == "aten::copy_"
                )
                self.assertEqual(copies, 0 if reuse else 1)

    def test_empty_and_failed_attention_preserve_cache_locations(self):
        def fail(**kwargs):
            raise RuntimeError("backend failed")

        output = torch.full((4, 2, 512), float("nan"))
        self.run_attention(output, 0, fail)
        torch.testing.assert_close(output, torch.zeros_like(output))
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            self.run_attention(output, 2, fail)

    def test_trtllm_prefill_only_reuses_compatible_unpadded_output(self):
        """An incompatible or padded destination must not reach the VarSeq kernel."""
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as trtllm

        backend = trtllm.DeepseekV4TrtllmAttnBackend.__new__(
            trtllm.DeepseekV4TrtllmAttnBackend
        )
        backend.cuda_int32_kwargs = {"dtype": torch.int32, "device": "cpu"}
        backend.trtllm_workspace_buffer = torch.empty(1)
        backend.softmax_scale = 1.0
        core = SimpleNamespace(
            trtllm_prefill_qmeta=(torch.tensor([0, 2]), 2, 2, torch.tensor([2])),
            trtllm_prefill_swa_lens=torch.ones(2, dtype=torch.int32),
        )
        backend.forward_metadata = SimpleNamespace(core_attn_metadata=core)
        good = torch.empty(2, 2, 512, dtype=torch.bfloat16)
        for destination, rows, reuse in (
            (good, 2, True),
            (None, 2, False),
            (good.float(), 2, False),
            (good.view(2, 1, 1024), 2, False),
            (torch.empty(2, 512, 2, dtype=torch.bfloat16).transpose(1, 2), 2, False),
            (good, 3, False),
        ):
            with self.subTest(
                shape=None if destination is None else destination.shape,
                rows=rows,
                reuse=reuse,
            ):

                def launch(query, out, **kwargs):
                    result = (
                        out
                        if out is not None
                        else torch.empty(query.shape, dtype=torch.bfloat16)
                    )
                    return result.fill_(7)

                with (
                    patch.object(
                        backend, "_trtllm_kv_cache_views", return_value=(None, None)
                    ),
                    patch.object(trtllm, "_check_trtllm_query_rows"),
                    patch.object(
                        trtllm, "_get_trtllm_kv_counter_buffer", return_value=None
                    ),
                    patch.object(
                        trtllm,
                        "trtllm_batch_decode_sparse_mla_dsv4",
                        side_effect=launch,
                        create=True,
                    ),
                ):
                    result = backend._forward_trtllm_prefill(
                        q=torch.ones(rows, 2, 512),
                        layer=SimpleNamespace(layer_id=0, k_scale_float=None),
                        compress_ratio=0,
                        forward_batch=None,
                        attn_sink=torch.zeros(2),
                        swa_page_indices=torch.zeros(
                            rows, trtllm.SWA_WINDOW, dtype=torch.int32
                        ),
                        extra_indices=None,
                        extra_topk_lengths=None,
                        out=destination,
                    )
                torch.testing.assert_close(result[:2], torch.full_like(good, 7))
                if destination is not None:
                    self.assertEqual(result.data_ptr() == destination.data_ptr(), reuse)
                if rows == 3:
                    torch.testing.assert_close(result[2:], torch.zeros_like(result[2:]))


if __name__ == "__main__":
    unittest.main()
