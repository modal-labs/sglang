import os
import random
import unittest

import sglang as sgl
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    CustomTestCase,
)

register_cuda_ci(est_time=561, suite="stage-b-test-large-1-gpu")


class TestDFlashEngine(CustomTestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_TARGET_MODEL_DFLASH,
        "speculative_draft_model_path": DEFAULT_DRAFT_MODEL_DFLASH,
        "speculative_algorithm": "DFLASH",
        "speculative_num_draft_tokens": 16,
        "cuda_graph_max_bs": 5,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    }
    NUM_CONFIGS = 2

    THRESHOLDS = {
        "batch_avg_accept_len": 2.09,
        "accept_len": 5.61,
    }

    def setUp(self):
        self.prompt = "Today is a sunny day and I like"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            ref_engine = sgl.Engine(
                model_path=self.BASE_CONFIG["model_path"],
                cuda_graph_max_bs=1,
            )
            self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)[
                "text"
            ]
            ref_engine.shutdown()
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    def test_correctness(self):
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            configs = [
                # Basic config
                self.BASE_CONFIG,
                # Chunked prefill
                {**self.BASE_CONFIG, "chunked_prefill_size": 4},
            ]

            for i, config in enumerate(configs[: self.NUM_CONFIGS]):
                with self.subTest(i=i):
                    print(f"{config=}")
                    engine = sgl.Engine(
                        **config, log_level="info", decode_log_interval=10
                    )
                    try:
                        self._test_single_generation(engine)
                        self._test_first_token_finish(engine)
                        self._test_batch_generation(engine)
                        self._test_eos_token(engine)
                        self._test_acc_length(engine)
                    finally:
                        engine.flush_cache()
                        engine.shutdown()
                    print("=" * 100)
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    def _test_single_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        params = {"temperature": 0, "max_new_tokens": 50}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)

        print(f"{engine.get_server_info()=}")

        avg_spec_accept_length = engine.get_server_info()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(
            avg_spec_accept_length, self.THRESHOLDS["batch_avg_accept_len"]
        )

    def _test_first_token_finish(self, engine):
        prompt = [
            f"There are {i} apples on the table. How to divide them equally?"
            for i in range(8)
        ]
        params = [
            {"temperature": 0, "max_new_tokens": random.randint(1, 3)} for _ in range(8)
        ]
        outputs = engine.generate(prompt, params)
        for i, output in enumerate(outputs):
            print(f"Prompt: {prompt[i]}")
            print(f"Generated: {output['text']}")
            print("-" * 40)

    def _test_eos_token(self, engine):
        prompt = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nToday is a sunny day and I like [/INST]"
        params = {
            "temperature": 0.1,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        tokenizer = get_tokenizer(DEFAULT_TARGET_MODEL_DFLASH)
        output = engine.generate(prompt, params)["text"]
        print(f"{output=}")

        tokens = tokenizer.encode(output, truncation=False)
        self.assertNotIn(tokenizer.eos_token_id, tokens)

    def _test_acc_length(self, engine):
        prompt = [
            "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
        ] * 5
        sampling_params = {"temperature": 0, "max_new_tokens": 512}
        output = engine.generate(prompt, sampling_params)
        output = output[0]

        if "spec_verify_ct" in output["meta_info"]:
            acc_length = (
                output["meta_info"]["completion_tokens"]
                / output["meta_info"]["spec_verify_ct"]
            )
        else:
            acc_length = 1.0

        speed = (
            output["meta_info"]["completion_tokens"]
            / output["meta_info"]["e2e_latency"]
        )
        print(f"{acc_length=:.4f}, {speed=}")

        self.assertGreater(acc_length, self.THRESHOLDS["accept_len"])


class TestDFlashRadixCache(CustomTestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_TARGET_MODEL_DFLASH,
        "speculative_draft_model_path": DEFAULT_DRAFT_MODEL_DFLASH,
        "speculative_algorithm": "DFLASH",
        "speculative_num_draft_tokens": 16,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "attention_backend": "flashinfer",
        "skip_server_warmup": True,
        "cuda_graph_max_bs": 5,
    }
    THRESHOLDS = {
        "batch_avg_accept_len": 4.2,
        "accept_len": 4.3,
    }

    def test_correctness(self):
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        configs = [
            # Basic config
            self.BASE_CONFIG,
            # Chunked prefill & Page Size > 1
            {**self.BASE_CONFIG, "chunked_prefill_size": 64, "page_size": 4},
            {**self.BASE_CONFIG, "page_size": 4},
            # Large page size tend to expose IMA bugs.
            {**self.BASE_CONFIG, "page_size": 256},
            {**self.BASE_CONFIG, "cuda_graph_bs": [5], "page_size": 4},
            # Disable CUDA Graph
            {
                **self.BASE_CONFIG,
                "disable_cuda_graph": True,
                "page_size": 4,
            },
        ]

        try:
            for i, config in enumerate(configs):
                with self.subTest(i=i):
                    print(f"{config=}")
                    engine = sgl.Engine(
                        **config, log_level="info", decode_log_interval=10
                    )
                    try:
                        self._test_acc_length(engine)
                        self._test_batch_generation(engine)
                    finally:
                        engine.shutdown()
                    print("=" * 100)
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    def _test_acc_length(self, engine):
        warmup_prompt = [
            "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
        ]
        sampling_params = {"temperature": 0, "max_new_tokens": 512}
        engine.generate(warmup_prompt, sampling_params)
        test_prompt = [
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGive me a fully functional FastAPI server. Show the python code.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ]
        output = engine.generate(test_prompt, sampling_params)
        output = output[0]

        if "spec_verify_ct" in output["meta_info"]:
            acc_length = (
                output["meta_info"]["completion_tokens"]
                / output["meta_info"]["spec_verify_ct"]
            )
        else:
            acc_length = 1.0

        speed = (
            output["meta_info"]["completion_tokens"]
            / output["meta_info"]["e2e_latency"]
        )

        print(f"{acc_length=:.4f}, {speed=}")
        self.assertGreater(acc_length, self.THRESHOLDS["accept_len"])

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        params = {"temperature": 0, "max_new_tokens": 50}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)

        print(f"{engine.get_server_info()=}")

        avg_spec_accept_length = engine.get_server_info()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(
            avg_spec_accept_length, self.THRESHOLDS["batch_avg_accept_len"]
        )


if __name__ == "__main__":
    unittest.main()
