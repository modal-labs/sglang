"""Hermetic unit tests for DeepSeek MLA attention-method dispatch on ROCm.

`_dispatch_mla_subtype` picks the forward method for MLA attention. On ROCm the
fused-decode-MLA + fused-RoPE fast path (`MLA_FUSED_ROPE_ROCM`) is only correct
for the aiter attention backend; taking it under the triton backend GPU-faults
on gfx95 (MI355). This test pins the dispatch table so the triton MLA path stays
on the plain `MLA` method.

Pure Python (no GPU, no model weights): `_is_hip` is patched and `attn` /
`forward_batch` are lightweight fakes. Runs on any PR-CI lane.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _fake_forward_batch(is_decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: is_decode))


def _fake_attn(backend: str, rocm_fused_decode_mla: bool = True):
    return SimpleNamespace(
        current_attention_backend=backend,
        rocm_fused_decode_mla=rocm_fused_decode_mla,
    )


def _fake_trt_prefill_batch(prefix_len: int = 0):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: True,
        ),
        extend_prefix_lens_cpu=[prefix_len],
    )


def _fake_trt_attn():
    return SimpleNamespace(disable_chunked_prefix_cache=False)


class TestDispatchMLASubtype(CustomTestCase):
    def test_hip_aiter_decode_takes_fused_rope(self):
        # aiter + fused-decode + decode -> fused ROPE fast path (unchanged).
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA_FUSED_ROPE_ROCM)

    def test_hip_triton_decode_stays_plain_mla(self):
        # The fix: triton backend must NOT take the aiter-only fused path even
        # with rocm_fused_decode_mla set -- that path GPU-faults on gfx95.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("triton"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_hip_aiter_extend_stays_plain_mla(self):
        # Fused path is decode-only; extend/prefill uses plain MLA.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=False)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)


class TestTRTLLMPrefillDispatch(CustomTestCase):
    def _dispatch(self, *, prefix_len=0, breakable=True, tc_piecewise=False):
        with (
            mock.patch.object(
                abh, "is_in_breakable_cuda_graph", return_value=breakable
            ),
            mock.patch.object(
                abh, "is_in_tc_piecewise_cuda_graph", return_value=tc_piecewise
            ),
        ):
            return abh.handle_attention_trtllm_mla(
                _fake_trt_attn(), _fake_trt_prefill_batch(prefix_len)
            )

    def test_breakable_pure_prefill_uses_expanded_mha(self):
        self.assertEqual(
            self._dispatch(),
            AttnForwardMethod.MHA_CHUNKED_KV,
        )

    def test_breakable_cached_prefix_uses_expanded_mha(self):
        self.assertEqual(
            self._dispatch(prefix_len=8192),
            AttnForwardMethod.MHA_CHUNKED_KV,
        )

    def test_tc_piecewise_stays_absorbed_mla(self):
        self.assertEqual(
            self._dispatch(breakable=False, tc_piecewise=True),
            AttnForwardMethod.MLA,
        )

    def test_tokenspeed_breakable_stays_absorbed_mla(self):
        with (
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=True),
            mock.patch.object(abh, "is_in_tc_piecewise_cuda_graph", return_value=False),
        ):
            method = abh.handle_attention_tokenspeed_mla(
                _fake_trt_attn(), _fake_trt_prefill_batch()
            )

        self.assertEqual(method, AttnForwardMethod.MLA)


if __name__ == "__main__":
    unittest.main()
