"""
Test SPEC V2 with tree speculative decoding (topk > 1).

This test suite validates correctness of speculative decoding implementations,
comparing V1 (baseline) vs V2 (overlap scheduling) and spec vs non-spec outputs.

================================================================================
MODELS USED
================================================================================
- Target: Qwen/Qwen3-8B
- Draft: Tengyunw/qwen3_8b_eagle3

================================================================================
TEST CLASSES
================================================================================

TestDeterminism
    Tests that engines produce deterministic output across runs.
    - test_reference_deterministic: Non-speculative engine determinism
    - test_v1_topk1_deterministic: V1 chain spec (topk=1) determinism
    - test_v1_topk2_deterministic: V1 tree spec (topk=2) determinism

TestV1TreeSpecBehavior
    Tests V1 tree spec produces valid output with good acceptance rate.
    - test_v1_tree_spec_acceptance: Validates acceptance rate > 1

TestV2VsV1
    Tests V2 matches V1 behavior (key implementation validation).
    - test_v2_matches_v1_topk1: V2 chain spec should match V1 chain spec
    - test_v2_matches_v1_topk2: V2 tree spec should match V1 tree spec

TestSpecVsNonSpec (base class)
    Compares speculative decoding vs non-speculative decoding.
    Subclasses configure USE_V2 (V1/V2) and USE_TREE (chain/tree):
    - TestSpecVsNonSpecV1Chain: V1 chain (topk=1) vs non-spec
    - TestSpecVsNonSpecV1Tree: V1 tree (topk=2) vs non-spec
    - TestSpecVsNonSpecV2Chain: V2 chain (topk=1) vs non-spec
    - TestSpecVsNonSpecV2Tree: V2 tree (topk=2) vs non-spec

================================================================================
HOW TO RUN TESTS
================================================================================

Run all tests:
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v

Run a specific test class:
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py::TestDeterminism -v
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py::TestV2VsV1 -v
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py::TestSpecVsNonSpecV1Chain -v

Run a specific test method:
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py::TestDeterminism::test_reference_deterministic -v
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py::TestV2VsV1::test_v2_matches_v1_topk1 -v

Run tests by keyword pattern:
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "deterministic"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "V1"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "V2"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "topk1"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "topk2"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "Chain"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "Tree"
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -k "SpecVsNonSpec"

Run with stdout output (see print statements):
    python -m pytest test/registered/spec/eagle/test_eagle_v2_tree_spec.py -v -s

================================================================================
CONFIGURATION
================================================================================

All tests use:
- page_size=1 (baseline validation, page_size > 1 not yet supported for tree spec)
- attention_backend="flashinfer"
- temperature=0, sampling_seed=42
- Shallow tree configs (topk=2, num_steps=2) to avoid crashes
"""

import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

# Test models - Qwen3-8B with EAGLE3 draft
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "Tengyunw/qwen3_8b_eagle3"


class TestDeterminism(CustomTestCase):
    """
    Test that engines produce deterministic output across runs.
    This is a prerequisite for any comparison tests.
    """

    PROMPTS = [
        "Hello, my name is",
        "The capital of France is",
        "In one sentence, explain speculative decoding.",
    ]

    SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 64, "sampling_seed": 42}

    def test_reference_deterministic(self):
        """Non-speculative engine should be deterministic."""
        print(f"\n{'='*60}")
        print("Testing reference engine determinism")
        print(f"{'='*60}")

        outputs_run1 = []
        outputs_run2 = []

        # Run 1
        engine1 = sgl.Engine(model_path=TARGET_MODEL, mem_fraction_static=0.5, page_size=1, attention_backend="flashinfer")
        try:
            for prompt in self.PROMPTS:
                output = engine1.generate(prompt, self.SAMPLING_PARAMS)["text"]
                outputs_run1.append(output)
                print(f"  Run1 '{prompt}': '{output[:40]}...'")
        finally:
            engine1.shutdown()

        # Run 2
        engine2 = sgl.Engine(model_path=TARGET_MODEL, mem_fraction_static=0.5, page_size=1, attention_backend="flashinfer")
        try:
            for prompt in self.PROMPTS:
                output = engine2.generate(prompt, self.SAMPLING_PARAMS)["text"]
                outputs_run2.append(output)
                print(f"  Run2 '{prompt}': '{output[:40]}...'")
        finally:
            engine2.shutdown()

        for i, (o1, o2) in enumerate(zip(outputs_run1, outputs_run2)):
            self.assertEqual(
                o1, o2,
                f"\nReference engine should be deterministic for prompt {i}:\n"
                f"  Prompt: '{self.PROMPTS[i]}'\n"
                f"  Run 1: '{o1}'\n"
                f"  Run 2: '{o2}'"
            )

    def test_v1_topk1_deterministic(self):
        """V1 with topk=1 should be deterministic across runs."""
        print(f"\n{'='*60}")
        print("Testing V1 topk=1 determinism")
        print(f"{'='*60}")

        config = {
            "model_path": TARGET_MODEL,
            "speculative_draft_model_path": DRAFT_MODEL,
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "page_size": 1,
            "mem_fraction_static": 0.5,
            "attention_backend": "flashinfer",
        }

        outputs_run1 = []
        outputs_run2 = []

        # Run 1
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            engine1 = sgl.Engine(**config, log_level="warning")
            try:
                for prompt in self.PROMPTS:
                    output = engine1.generate(prompt, self.SAMPLING_PARAMS)["text"]
                    outputs_run1.append(output)
                    print(f"  Run1 '{prompt}': '{output[:40]}...'")
            finally:
                engine1.shutdown()

        # Run 2
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            engine2 = sgl.Engine(**config, log_level="warning")
            try:
                for prompt in self.PROMPTS:
                    output = engine2.generate(prompt, self.SAMPLING_PARAMS)["text"]
                    outputs_run2.append(output)
                    print(f"  Run2 '{prompt}': '{output[:40]}...'")
            finally:
                engine2.shutdown()

        for i, (o1, o2) in enumerate(zip(outputs_run1, outputs_run2)):
            self.assertEqual(
                o1, o2,
                f"\nV1 topk=1 should be deterministic for prompt {i}:\n"
                f"  Prompt: '{self.PROMPTS[i]}'\n"
                f"  Run 1: '{o1}'\n"
                f"  Run 2: '{o2}'"
            )

    def test_v1_topk2_deterministic(self):
        """V1 with topk=2 should be deterministic across runs."""
        print(f"\n{'='*60}")
        print("Testing V1 topk=2 determinism")
        print(f"{'='*60}")

        config = {
            "model_path": TARGET_MODEL,
            "speculative_draft_model_path": DRAFT_MODEL,
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": 2,
            "speculative_eagle_topk": 2,
            "speculative_num_draft_tokens": 3,
            "page_size": 1,
            "mem_fraction_static": 0.5,
            "attention_backend": "flashinfer",
        }

        outputs_run1 = []
        outputs_run2 = []

        # Run 1
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            engine1 = sgl.Engine(**config, log_level="warning")
            try:
                for prompt in self.PROMPTS:
                    output = engine1.generate(prompt, self.SAMPLING_PARAMS)["text"]
                    outputs_run1.append(output)
                    print(f"  Run1 '{prompt}': '{output[:40]}...'")
            finally:
                engine1.shutdown()

        # Run 2
        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            engine2 = sgl.Engine(**config, log_level="warning")
            try:
                for prompt in self.PROMPTS:
                    output = engine2.generate(prompt, self.SAMPLING_PARAMS)["text"]
                    outputs_run2.append(output)
                    print(f"  Run2 '{prompt}': '{output[:40]}...'")
            finally:
                engine2.shutdown()

        for i, (o1, o2) in enumerate(zip(outputs_run1, outputs_run2)):
            self.assertEqual(
                o1, o2,
                f"\nV1 topk=2 should be deterministic for prompt {i}:\n"
                f"  Prompt: '{self.PROMPTS[i]}'\n"
                f"  Run 1: '{o1}'\n"
                f"  Run 2: '{o2}'"
            )


class TestV1TreeSpecBehavior(CustomTestCase):
    """Test V1 tree spec produces valid output with good acceptance."""

    PROMPTS = [
        "Hello, my name is",
        "The capital of France is",
    ]

    SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 32, "sampling_seed": 42}

    def test_v1_tree_spec_acceptance(self):
        """V1 with topk>1 should have acceptance rate > 1."""
        print(f"\n{'='*60}")
        print("Testing V1 tree spec acceptance rate")
        print(f"{'='*60}")

        config = {
            "model_path": TARGET_MODEL,
            "speculative_draft_model_path": DRAFT_MODEL,
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": 2,
            "speculative_eagle_topk": 2,
            "speculative_num_draft_tokens": 3,
            "page_size": 1,
            "mem_fraction_static": 0.5,
            "attention_backend": "flashinfer",
        }

        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
            engine = sgl.Engine(**config, log_level="warning")
            try:
                outputs = engine.generate(self.PROMPTS, self.SAMPLING_PARAMS)

                for i, output in enumerate(outputs):
                    output_text = output["text"]
                    print(f"  Prompt: '{self.PROMPTS[i]}'")
                    print(f"    Output: '{output_text[:50]}...'")
                    self.assertGreater(len(output_text), 10, "Output too short")

                info = engine.get_server_info()["internal_states"][0]
                avg_acc = info.get("avg_spec_accept_length", 1.0)
                print(f"  Avg accept length: {avg_acc:.2f}")
                self.assertGreater(avg_acc, 1.0, "Tree spec should have acceptance > 1")
            finally:
                engine.shutdown()


class TestV2VsV1(CustomTestCase):
    """
    Test V2 matches V1 behavior.
    This is the key test for the implementation.
    """

    PROMPTS = [
        "Hello, my name is",
        "The capital of France is",
        "In one sentence, explain speculative decoding.",
    ]

    SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 64, "sampling_seed": 42}

    def _create_engine(self, topk, num_steps, num_draft_tokens, use_v2):
        """Create engine with consistent config."""
        config = {
            "model_path": TARGET_MODEL,
            "speculative_draft_model_path": DRAFT_MODEL,
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": num_steps,
            "speculative_eagle_topk": topk,
            "speculative_num_draft_tokens": num_draft_tokens,
            "page_size": 1,
            "mem_fraction_static": 0.5,
            "attention_backend": "flashinfer",
        }

        if use_v2:
            with envs.SGLANG_ENABLE_SPEC_V2.override(True):
                return sgl.Engine(**config, log_level="warning")
        else:
            with envs.SGLANG_ENABLE_SPEC_V2.override(False):
                return sgl.Engine(**config, log_level="warning")

    def test_v2_matches_v1_topk1(self):
        """V2 with topk=1 should match V1 with topk=1."""
        topk, num_steps, num_draft_tokens = 1, 3, 4

        print(f"\n{'='*60}")
        print(f"Testing V2 vs V1: topk={topk}")
        print(f"{'='*60}")

        # Get V1 outputs
        v1_outputs = []
        v1_engine = self._create_engine(topk, num_steps, num_draft_tokens, use_v2=False)
        try:
            for prompt in self.PROMPTS:
                output = v1_engine.generate(prompt, self.SAMPLING_PARAMS)["text"]
                v1_outputs.append(output)
        finally:
            v1_engine.shutdown()

        # Get V2 outputs
        v2_outputs = []
        v2_engine = self._create_engine(topk, num_steps, num_draft_tokens, use_v2=True)
        try:
            for prompt in self.PROMPTS:
                output = v2_engine.generate(prompt, self.SAMPLING_PARAMS)["text"]
                v2_outputs.append(output)
        finally:
            v2_engine.shutdown()

        # Compare
        for i, (v1_out, v2_out) in enumerate(zip(v1_outputs, v2_outputs)):
            print(f"  Prompt: '{self.PROMPTS[i]}'")
            print(f"    V1: '{v1_out[:40]}...'")
            print(f"    V2: '{v2_out[:40]}...'")
            self.assertEqual(
                v1_out, v2_out,
                f"\nV2 should match V1 for topk=1, prompt {i}:\n"
                f"  Prompt: '{self.PROMPTS[i]}'\n"
                f"  V1 (baseline): '{v1_out}'\n"
                f"  V2 (overlap):  '{v2_out}'"
            )

    def test_v2_matches_v1_topk2(self):
        """V2 with topk=2 should match V1 with topk=2."""
        topk, num_steps, num_draft_tokens = 2, 2, 3

        print(f"\n{'='*60}")
        print(f"Testing V2 vs V1: topk={topk}")
        print(f"{'='*60}")

        # Get V1 outputs
        v1_outputs = []
        v1_engine = self._create_engine(topk, num_steps, num_draft_tokens, use_v2=False)
        try:
            for prompt in self.PROMPTS:
                output = v1_engine.generate(prompt, self.SAMPLING_PARAMS)["text"]
                v1_outputs.append(output)
            v1_info = v1_engine.get_server_info()["internal_states"][0]
            v1_acc = v1_info.get("avg_spec_accept_length", 1.0)
        finally:
            v1_engine.shutdown()

        # Get V2 outputs
        v2_outputs = []
        v2_engine = self._create_engine(topk, num_steps, num_draft_tokens, use_v2=True)
        try:
            for prompt in self.PROMPTS:
                output = v2_engine.generate(prompt, self.SAMPLING_PARAMS)["text"]
                v2_outputs.append(output)
            v2_info = v2_engine.get_server_info()["internal_states"][0]
            v2_acc = v2_info.get("avg_spec_accept_length", 1.0)
        finally:
            v2_engine.shutdown()

        # Compare outputs
        for i, (v1_out, v2_out) in enumerate(zip(v1_outputs, v2_outputs)):
            print(f"  Prompt: '{self.PROMPTS[i]}'")
            print(f"    V1: '{v1_out[:40]}...'")
            print(f"    V2: '{v2_out[:40]}...'")
            self.assertEqual(
                v1_out, v2_out,
                f"\nV2 should match V1 for topk=2, prompt {i}:\n"
                f"  Prompt: '{self.PROMPTS[i]}'\n"
                f"  V1 (baseline): '{v1_out}'\n"
                f"  V2 (overlap):  '{v2_out}'"
            )

        print(f"  V1 acceptance: {v1_acc:.2f}")
        print(f"  V2 acceptance: {v2_acc:.2f}")


class TestSpecVsNonSpec(CustomTestCase):
    """
    Compare speculative decoding vs non-speculative decoding.

    Note: Speculative decoding (especially tree spec with topk > 1) may produce
    different outputs than non-speculative decoding. This test documents the
    behavior and checks for reasonable acceptance rates.

    Configure via class attributes or subclassing:
    - USE_V2: False (default) or True for overlap scheduling
    - USE_TREE: False (default, topk=1) or True (topk=2) for tree spec
    """

    # Configuration - override in subclasses to test different modes
    USE_V2 = False  # V1 by default
    USE_TREE = False  # Chain (topk=1) by default

    PROMPTS = [
        "Hello, my name is",
        "The capital of France is",
        "In one sentence, explain speculative decoding.",
    ]

    SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 64, "sampling_seed": 42}

    def _get_config(self):
        """Get spec config based on USE_TREE setting."""
        if self.USE_TREE:
            return {
                "topk": 2,
                "num_steps": 2,
                "num_draft_tokens": 3,
            }
        else:
            return {
                "topk": 1,
                "num_steps": 3,
                "num_draft_tokens": 4,
            }

    def _create_reference_engine(self):
        """Create non-speculative reference engine."""
        return sgl.Engine(model_path=TARGET_MODEL, mem_fraction_static=0.5, page_size=1, attention_backend="flashinfer")

    def _create_spec_engine(self):
        """Create speculative decoding engine."""
        spec_config = self._get_config()
        config = {
            "model_path": TARGET_MODEL,
            "speculative_draft_model_path": DRAFT_MODEL,
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": spec_config["num_steps"],
            "speculative_eagle_topk": spec_config["topk"],
            "speculative_num_draft_tokens": spec_config["num_draft_tokens"],
            "page_size": 1,
            "mem_fraction_static": 0.5,
            "attention_backend": "flashinfer",
        }

        if self.USE_V2:
            with envs.SGLANG_ENABLE_SPEC_V2.override(True):
                return sgl.Engine(**config, log_level="warning")
        else:
            with envs.SGLANG_ENABLE_SPEC_V2.override(False):
                return sgl.Engine(**config, log_level="warning")

    def test_spec_vs_nonspec(self):
        """Compare speculative vs non-speculative decoding."""
        spec_config = self._get_config()
        mode = "V2" if self.USE_V2 else "V1"
        tree_mode = "tree" if self.USE_TREE else "chain"

        print(f"\n{'='*60}")
        print(f"Testing {mode} {tree_mode} spec (topk={spec_config['topk']}) vs non-spec")
        print(f"{'='*60}")

        # Get reference outputs
        ref_outputs = []
        ref_engine = self._create_reference_engine()
        try:
            for prompt in self.PROMPTS:
                output = ref_engine.generate(prompt, self.SAMPLING_PARAMS)["text"]
                ref_outputs.append(output)
        finally:
            ref_engine.shutdown()

        # Get speculative outputs
        spec_outputs = []
        spec_engine = self._create_spec_engine()
        try:
            for prompt in self.PROMPTS:
                output = spec_engine.generate(prompt, self.SAMPLING_PARAMS)["text"]
                spec_outputs.append(output)
            info = spec_engine.get_server_info()["internal_states"][0]
            avg_acc = info.get("avg_spec_accept_length", 1.0)
        finally:
            spec_engine.shutdown()

        # Report results
        matches = 0
        for i, (ref_out, spec_out) in enumerate(zip(ref_outputs, spec_outputs)):
            match = ref_out == spec_out
            matches += int(match)
            match_str = "MATCH" if match else "DIFFER"

            print(f"\n  Prompt {i}: '{self.PROMPTS[i]}'")
            print(f"    Reference: '{ref_out[:50]}...'")
            print(f"    Spec:      '{spec_out[:50]}...'")
            print(f"    Status:    {match_str}")

        print(f"\n  Summary:")
        print(f"    Matches: {matches}/{len(self.PROMPTS)}")
        print(f"    Avg acceptance length: {avg_acc:.2f}")

        # Assertions
        # For chain spec (topk=1), outputs SHOULD match reference
        if not self.USE_TREE:
            for i, (ref_out, spec_out) in enumerate(zip(ref_outputs, spec_outputs)):
                self.assertEqual(
                    ref_out, spec_out,
                    f"\nChain spec (topk=1) should match reference for prompt {i}:\n"
                    f"  Prompt: '{self.PROMPTS[i]}'\n"
                    f"  Reference (non-spec): '{ref_out}'\n"
                    f"  Speculative (chain):  '{spec_out}'"
                )

        # For tree spec (topk>1), we just check acceptance rate is reasonable
        # (outputs may differ from reference, which is expected)
        self.assertGreater(avg_acc, 1.0, "Speculative decoding should have acceptance > 1")

        # Check outputs are not corrupted (basic sanity)
        for i, spec_out in enumerate(spec_outputs):
            self.assertGreater(len(spec_out), 10, f"Output {i} too short")
            # Check for obvious corruption patterns
            self.assertNotIn("!!", spec_out, f"Output {i} may be corrupted (double !)")


class TestSpecVsNonSpecV1Chain(TestSpecVsNonSpec):
    """V1 chain spec (topk=1) vs non-spec - should match exactly."""
    USE_V2 = False
    USE_TREE = False


class TestSpecVsNonSpecV1Tree(TestSpecVsNonSpec):
    """V1 tree spec (topk=2) vs non-spec - may differ."""
    USE_V2 = False
    USE_TREE = True


class TestSpecVsNonSpecV2Chain(TestSpecVsNonSpec):
    """V2 chain spec (topk=1) vs non-spec - should match exactly."""
    USE_V2 = True
    USE_TREE = False


class TestSpecVsNonSpecV2Tree(TestSpecVsNonSpec):
    """V2 tree spec (topk=2) vs non-spec - may differ (and may be broken)."""
    USE_V2 = True
    USE_TREE = True


if __name__ == "__main__":
    unittest.main()
