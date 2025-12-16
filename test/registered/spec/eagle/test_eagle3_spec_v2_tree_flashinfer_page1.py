import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "Tengyunw/qwen3_8b_eagle3"


@unittest.skipIf(is_in_ci(), "Uses large models intended for local iteration.")
class TestEagle3SpecV2TreeFlashInferPageSize1(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "flashinfer",
            "--decode-log-interval",
            "10",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "10",
            # For num_steps=2, the v2 draft path can select at most 3 nodes (so draft_tokens <= 4 incl. root).
            "--speculative-num-draft-tokens",
            "32",
            "--page-size",
            "1",
            "--max-running-requests",
            "4",
            "--mem-fraction-static",
            "0.75",
        ]

        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate(self):
        spec_v2_env = os.environ.get("SGLANG_ENABLE_SPEC_V2", "<unset>")
        print(f"SGLANG_ENABLE_SPEC_V2={spec_v2_env}")

        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "In one sentence, explain speculative decoding.",
        ]
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "sampling_seed": 42,
                    "max_new_tokens": 64,
                },
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        outputs = resp.json()
        self.assertEqual(len(outputs), len(prompts))
        for prompt, out in zip(prompts, outputs):
            self.assertIn("text", out)
            self.assertTrue(out["text"])
            print("PROMPT:", prompt)
            print("OUTPUT:", out["text"])
            meta = out.get("meta_info", {})
            self.assertIn("spec_verify_ct", meta)
            self.assertIn("completion_tokens", meta)
            # Average accepted tokens per verify iteration (includes bonus token).
            avg_accept_len = meta["completion_tokens"] / max(meta["spec_verify_ct"], 1)
            print(
                "avg_accept_len=",
                avg_accept_len,
                "spec_verify_ct=",
                meta["spec_verify_ct"],
                "completion_tokens=",
                meta["completion_tokens"],
            )
            self.assertGreater(avg_accept_len, 1.05)
        assert self.process.poll() is None


if __name__ == "__main__":
    unittest.main()
