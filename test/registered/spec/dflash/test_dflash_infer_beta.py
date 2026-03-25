import os
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=283, suite="stage-b-test-small-1-gpu")


class TestDFlashServerBase(CustomTestCase, MatchedStopMixin):
    max_running_requests = 64
    attention_backend = "flashinfer"
    page_size = 1
    other_launch_args = []
    model = DEFAULT_TARGET_MODEL_DFLASH
    draft_model = DEFAULT_DRAFT_MODEL_DFLASH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--page-size",
            str(cls.page_size),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
                1
            ), envs.SGLANG_SPEC_NAN_DETECTION.override(
                True
            ), envs.SGLANG_SPEC_OOB_DETECTION.override(
                True
            ):
                cls.process = popen_launch_server(
                    cls.model,
                    cls.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=launch_args,
                )
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        assert self.process.poll() is None

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1000,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"TestDFlashServerBase -- {metrics=}")
        self.assertGreater(metrics["accuracy"], 0.23)
        assert self.process.poll() is None


class TestDFlashServerPage(TestDFlashServerBase):
    page_size = 64


if __name__ == "__main__":
    unittest.main()
