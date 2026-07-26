"""CPU regressions for DFlash with lazy Mamba extra-buffer tracking."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _precompile_fused_kv_helper_tp,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashFusedKVPrecompile(unittest.TestCase):
    @staticmethod
    def _group(*, broadcast_result=None, gathered=None):
        return SimpleNamespace(
            world_size=8,
            broadcast_object=Mock(return_value=broadcast_result),
            all_gather_object=Mock(
                side_effect=lambda value: gathered or [None] * 7 + [value]
            ),
        )

    def test_rank_zero_compile_failure_is_broadcast_and_raised(self):
        helper = SimpleNamespace(precompile=Mock(side_effect=ValueError("compile")))
        group = self._group(broadcast_result="ValueError: compile")

        with self.assertRaisesRegex(RuntimeError, "failed on TP rank 0"):
            _precompile_fused_kv_helper_tp(helper, 0, group)

        group.broadcast_object.assert_called_once_with("ValueError: compile", src=0)
        group.all_gather_object.assert_not_called()

    def test_peer_artifact_load_failure_is_raised(self):
        helper = SimpleNamespace(precompile=Mock(side_effect=ValueError("load")))
        group = self._group(
            broadcast_result=None,
            gathered=[None, "rank=1 ValueError: load"] + [None] * 6,
        )

        with self.assertRaisesRegex(RuntimeError, "artifact load failed"):
            _precompile_fused_kv_helper_tp(helper, 1, group)

        helper.precompile.assert_called_once()


class TestDFlashMambaLazyValidation(CustomTestCase):
    @staticmethod
    def _view(**overrides):
        values = {
            "linear_attn_backend": "triton",
            "mamba_radix_cache_strategy": "extra_buffer_lazy",
            "disaggregation_mode": "null",
            "speculative_algorithm": "DFLASH",
            "speculative_num_draft_tokens": 8,
            "mamba_track_interval": 16,
            "page_size": None,
            "chunked_prefill_size": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _validate(self, view):
        server_args = object.__new__(ServerArgs)
        with patch("sglang.srt.server_args.is_cuda", return_value=True):
            server_args._validate_mamba_extra_buffer(
                view, "KimiK3ForConditionalGeneration"
            )

    def test_extra_buffer_lazy_allows_dflash(self):
        self._validate(self._view())

    def test_extra_buffer_lazy_still_rejects_pd_disaggregation(self):
        with self.assertRaisesRegex(AssertionError, "PD disaggregation"):
            self._validate(self._view(disaggregation_mode="decode"))


class TestDFlashMambaLazyVerifyOrdering(CustomTestCase):
    def test_track_plan_is_rebuilt_before_target_forward_batch(self):
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(DFlashWorkerV2.forward_batch_generation))
        )

        track_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_mamba_track_for_verify"
        ]
        prepare_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare_for_verify"
        ]
        target_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward_batch_generation"
            and any(
                keyword.arg == "is_verify"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
        ]

        self.assertEqual(len(track_calls), 1)
        self.assertEqual(len(prepare_calls), 1)
        self.assertEqual(len(target_calls), 1)
        self.assertEqual(ast.unparse(track_calls[0].args[0]), "batch")
        self.assertEqual(ast.unparse(prepare_calls[0].args[0]), "batch")
        self.assertLess(track_calls[0].lineno, prepare_calls[0].lineno)
        self.assertLess(prepare_calls[0].lineno, target_calls[0].lineno)

    def test_commit_uses_accepted_post_verify_lengths_for_track_step(self):
        update = Mock()
        model = object()
        track_indices = torch.tensor([10, 11, 12], dtype=torch.int64)
        worker = SimpleNamespace(
            _need_mamba_verify_commit=True,
            server_args=SimpleNamespace(mamba_track_interval=256),
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(
                        update_mamba_state_after_mtp_verify=update
                    ),
                    model=model,
                )
            ),
        )
        batch = SimpleNamespace(mamba_track_indices=track_indices)

        DFlashWorkerV2._update_target_mamba_state_after_verify(
            worker,
            batch=batch,
            seq_lens_pre_verify=torch.tensor([255, 250, 256], dtype=torch.int32),
            commit_lens=torch.tensor([1, 8, 1], dtype=torch.int32),
        )

        update.assert_called_once()
        call = update.call_args.kwargs
        torch.testing.assert_close(
            call["last_correct_step_indices"],
            torch.tensor([0, 7, 0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            call["mamba_steps_to_track"],
            torch.tensor([0, 5, -1], dtype=torch.int64),
        )
        self.assertIs(call["mamba_track_indices"], track_indices)
        self.assertIs(call["model"], model)


if __name__ == "__main__":
    unittest.main()
