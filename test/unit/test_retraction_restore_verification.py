"""The QA verifier detects changed target/draft/state bytes without logging them."""

import copy
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.retraction_debug import (
    _assert_snapshot_equal,
    verify_retraction_restore,
)


class RetractionRestoreVerificationTest(unittest.TestCase):
    def snapshot(self):
        return (
            [[torch.arange(8, dtype=torch.int64)]],
            ([torch.tensor([1.0, float("nan")])], torch.tensor([3], dtype=torch.int32)),
            [[torch.tensor([4.0, 5.0], dtype=torch.bfloat16)]],
        )

    def test_actual_pool_recapture_checks_each_component(self):
        expected = self.snapshot()
        observed = []
        indices = torch.tensor([1024, 1025, 2050])
        mamba = torch.tensor(7)
        allocator = SimpleNamespace(
            get_cpu_copy=lambda loc, mamba_indices: (
                observed.append((loc, mamba_indices)) or copy.deepcopy(expected)
            )
        )
        with envs.SGLANG_TEST_RETRACT.override(True):
            event = verify_retraction_restore(
                allocator, expected, indices, mamba, "test"
            )
        self.assertEqual(event["status"], "passed")
        self.assertEqual(set(event["components"]), {"target", "mamba", "draft"})
        self.assertGreater(event["components"]["draft"]["bytes"], 0)
        self.assertIs(observed[0][0], indices)
        self.assertIs(observed[0][1], mamba)

    def test_corrupt_target_mamba_and_draft_each_fail(self):
        for component in range(3):
            expected = self.snapshot()
            actual = copy.deepcopy(expected)
            tensor = (
                actual[0][0][0]
                if component == 0
                else actual[1][0][0]
                if component == 1
                else actual[2][0][0]
            )
            tensor[0] = 42
            allocator = SimpleNamespace(get_cpu_copy=lambda *a, **kw: actual)
            with (
                self.subTest(component=component),
                envs.SGLANG_TEST_RETRACT.override(True),
            ):
                with self.assertRaisesRegex(RuntimeError, "byte verification failed"):
                    verify_retraction_restore(
                        allocator, expected, torch.tensor([1]), None, "test"
                    )

    def test_bounds_skip_without_recapture(self):
        for count, tokens in ((32, 3), (0, 4097)):
            allocator = SimpleNamespace(
                _retraction_verify_count=count,
                get_cpu_copy=lambda *a, **k: self.fail("must not recapture"),
            )
            with envs.SGLANG_TEST_RETRACT.override(True):
                event = verify_retraction_restore(
                    allocator, None, torch.arange(tokens), None, "test"
                )
            self.assertEqual(event["status"], "skipped")

    def test_requires_explicit_forced_qa(self):
        allocator = SimpleNamespace(
            get_cpu_copy=lambda *a, **k: self.fail("must not recapture")
        )
        with envs.SGLANG_TEST_RETRACT.override(False):
            with self.assertRaisesRegex(RuntimeError, "requires SGLANG_TEST_RETRACT"):
                verify_retraction_restore(
                    allocator, None, torch.tensor([1]), None, "test"
                )

    def test_server_configuration_rejects_non_qa_roles(self):
        from sglang.srt.server_args import ServerArgs

        for forced, role in ((False, "decode"), (True, "prefill"), (True, "null")):
            args = ServerArgs(model_path="dummy", disaggregation_mode=role)
            with (
                self.subTest(forced=forced, role=role),
                envs.SGLANG_TEST_RETRACT.override(forced),
                envs.SGLANG_TEST_RETRACT_VERIFY.override(True),
                self.assertRaisesRegex(ValueError, "forced-retraction decode"),
            ):
                args._handle_debug_utils()
        args = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        with (
            envs.SGLANG_TEST_RETRACT.override(True),
            envs.SGLANG_TEST_RETRACT_VERIFY.override(True),
        ):
            args._handle_debug_utils()

    def test_byte_comparison_handles_fp8_and_detects_shape_change(self):
        value = torch.arange(8, dtype=torch.float32).to(torch.float8_e4m3fn)
        self.assertEqual(_assert_snapshot_equal(value, value.clone())["bytes"], 8)
        with self.assertRaisesRegex(AssertionError, "shape or dtype"):
            _assert_snapshot_equal(value, value.reshape(2, 4))


if __name__ == "__main__":
    unittest.main()
