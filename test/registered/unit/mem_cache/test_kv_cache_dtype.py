# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_cache_dtype import (
    configure_kv_cache_dtype,
    select_kv_cache_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSelectKVCacheDtype(unittest.TestCase):
    def test_draft_worker_uses_explicit_override(self):
        self.assertEqual(
            select_kv_cache_dtype(
                target_kv_cache_dtype="fp8_e4m3",
                draft_kv_cache_dtype="bfloat16",
                is_draft_worker=True,
            ),
            "bfloat16",
        )

    def test_target_worker_ignores_draft_override(self):
        self.assertEqual(
            select_kv_cache_dtype(
                target_kv_cache_dtype="fp8_e4m3",
                draft_kv_cache_dtype="bfloat16",
                is_draft_worker=False,
            ),
            "fp8_e4m3",
        )

    def test_draft_worker_inherits_target_without_override(self):
        self.assertEqual(
            select_kv_cache_dtype(
                target_kv_cache_dtype="fp8_e4m3",
                draft_kv_cache_dtype=None,
                is_draft_worker=True,
            ),
            "fp8_e4m3",
        )

    def test_bfloat16_draft_request_resolves_to_bfloat16_storage(self):
        requested = select_kv_cache_dtype(
            target_kv_cache_dtype="fp8_e4m3",
            draft_kv_cache_dtype="bfloat16",
            is_draft_worker=True,
        )
        resolved, dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=requested,
            model=SimpleNamespace(quant_config=None),
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=False,
            speculative_draft_attention_backend="flashinfer",
        )
        self.assertIsNone(resolved)
        self.assertIs(dtype, torch.bfloat16)

    def test_target_and_draft_runners_keep_independent_dtypes(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        server_args = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            speculative_draft_kv_cache_dtype="bfloat16",
            speculative_draft_attention_backend="flashinfer",
        )

        target_runner = object.__new__(ModelRunner)
        target_runner.server_args = server_args
        target_runner.is_draft_worker = False
        target_runner.dtype = torch.bfloat16
        target_runner.model = SimpleNamespace(quant_config=None)
        target_runner.configure_kv_cache_dtype()

        draft_runner = object.__new__(ModelRunner)
        draft_runner.server_args = server_args
        draft_runner.is_draft_worker = True
        draft_runner.dtype = torch.bfloat16
        draft_runner.model = SimpleNamespace(quant_config=None)
        draft_runner.configure_kv_cache_dtype()

        self.assertIs(target_runner.kv_cache_dtype, torch.float8_e4m3fn)
        self.assertEqual(target_runner.kv_cache_dtype_str, "fp8_e4m3")
        self.assertIs(draft_runner.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(draft_runner.kv_cache_dtype_str, "bfloat16")
        self.assertEqual(server_args.kv_cache_dtype, "fp8_e4m3")


if __name__ == "__main__":
    unittest.main()
