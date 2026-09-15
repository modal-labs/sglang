import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.layers.attention.trtllm_mla_backend import (
    _verify_fused_kv_write_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTRTLLMMLAVerifyFusedKVWrite(CustomTestCase):
    def test_gate_requires_fused_decode_env_and_static_pool(self):
        static_pool = SimpleNamespace()
        hisparse_pool = SimpleNamespace(
            translate_loc_to_hisparse_device=lambda loc: loc
        )
        cases = (
            (False, True, static_pool, False),
            (True, True, static_pool, True),
            (True, True, hisparse_pool, False),
            (True, False, static_pool, False),
        )

        for env_enabled, decode_gate, pool, expected in cases:
            with self.subTest(
                env_enabled=env_enabled,
                decode_gate=decode_gate,
                pool=pool,
            ):
                with envs.SGLANG_TRTLLM_MLA_VERIFY_FUSED_KV_WRITE.override(env_enabled):
                    result = _verify_fused_kv_write_enabled(decode_gate, pool)
                self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
