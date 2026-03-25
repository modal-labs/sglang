import os
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.dflash_fixture import DFlashServerBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
)

register_cuda_ci(est_time=50, suite="stage-b-test-small-1-gpu")


class TestDFlashBasic(DFlashServerBase):
    target_model = DEFAULT_TARGET_MODEL_DFLASH
    draft_model = DEFAULT_DRAFT_MODEL_DFLASH

    spec_algo = "DFLASH"
    spec_block_size = 16

    extra_args = [
        "--dtype",
        "float16",
        "--chunked-prefill-size",
        1024,
    ]

    @classmethod
    def setUpClass(cls):
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            super().setUpClass()
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.target_model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.72)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 3.15)


if __name__ == "__main__":
    unittest.main()
