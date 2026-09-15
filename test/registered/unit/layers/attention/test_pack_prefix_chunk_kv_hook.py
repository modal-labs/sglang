"""CPU checks for the chunked-prefix MHA ``pack_prefix_chunk_kv`` hook.

Every attention backend that defines or delegates the hook must accept
``(layer, k_nope, k_pe, v)`` -- the shared caller in ``forward_mha`` passes
the layer positionally, so a stale 3-arg override raises ``TypeError`` on the
first cached-prefix prefill.  The dispatch itself is exercised with stubs:
a packed result is used as-is, ``None`` falls back to cat + cast.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HOOK_PARAMS = ("layer", "k_nope", "k_pe", "v")

BACKENDS_WITH_HOOK = (
    TRTLLMMLABackend,
    CuteDslMLABackend,
    TokenspeedMLABackend,
    HybridAttnBackend,
    HybridLinearAttnBackend,
)


class TestPackPrefixChunkKvSignature(CustomTestCase):
    def test_every_backend_accepts_layer_k_nope_k_pe_v(self):
        for backend_cls in BACKENDS_WITH_HOOK:
            with self.subTest(backend=backend_cls.__name__):
                hook = getattr(backend_cls, "pack_prefix_chunk_kv", None)
                self.assertIsNotNone(hook, "hook missing")
                params = [
                    p.name
                    for p in inspect.signature(hook).parameters.values()
                    if p.name != "self"
                ]
                self.assertEqual(params, list(HOOK_PARAMS))

    def test_tokenspeed_overrides_trtllm_hook(self):
        self.assertIsNot(
            TokenspeedMLABackend.pack_prefix_chunk_kv,
            TRTLLMMLABackend.pack_prefix_chunk_kv,
        )
        self.assertIs(
            CuteDslMLABackend.pack_prefix_chunk_kv,
            TRTLLMMLABackend.pack_prefix_chunk_kv,
        )


def _wrapper(cls, attr, inner):
    wrapper = object.__new__(cls)
    setattr(wrapper, attr, inner)
    return wrapper


class TestHybridDelegation(CustomTestCase):
    def _check_delegation(self, cls, attr):
        layer = SimpleNamespace(name="layer")
        k_nope, k_pe, v = torch.zeros(1), torch.zeros(2), torch.zeros(3)
        calls = []

        def pack(layer_, k_nope_, k_pe_, v_):
            calls.append((layer_, k_nope_, k_pe_, v_))
            return "packed"

        with_hook = _wrapper(cls, attr, SimpleNamespace(pack_prefix_chunk_kv=pack))
        self.assertEqual(
            with_hook.pack_prefix_chunk_kv(layer, k_nope, k_pe, v), "packed"
        )
        self.assertEqual(calls, [(layer, k_nope, k_pe, v)])

        without_hook = _wrapper(cls, attr, SimpleNamespace())
        self.assertIsNone(without_hook.pack_prefix_chunk_kv(layer, k_nope, k_pe, v))

        q16 = torch.zeros(1, dtype=torch.float16)
        self.assertEqual(without_hook.prefix_chunk_kv_proj_dtype(q16), torch.float16)
        declared = _wrapper(
            cls,
            attr,
            SimpleNamespace(prefix_chunk_kv_proj_dtype=lambda q: torch.bfloat16),
        )
        self.assertEqual(declared.prefix_chunk_kv_proj_dtype(q16), torch.bfloat16)

    def test_hybrid_attn_backend_delegates_to_prefill_backend(self):
        self._check_delegation(HybridAttnBackend, "prefill_backend")

    def test_hybrid_linear_attn_backend_delegates_to_full_attn_backend(self):
        self._check_delegation(HybridLinearAttnBackend, "full_attn_backend")


class _FakeLayer:
    """Just enough of DeepseekV2AttentionMLA for ``_chunked_prefix_attn_mha``."""

    num_local_heads = 2
    qk_nope_head_dim = 4
    qk_rope_head_dim = 2
    v_head_dim = 4

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens
        self.kv_dim = self.qk_nope_head_dim + self.v_head_dim
        self.kv_a_normed = torch.randn(num_tokens, 3, dtype=torch.bfloat16)
        self.k_pe = torch.randn(
            num_tokens, 1, self.qk_rope_head_dim, dtype=torch.bfloat16
        )
        self.kv = torch.randn(
            num_tokens, self.num_local_heads * self.kv_dim, dtype=torch.bfloat16
        )
        self.fetch_dtypes = []
        self.attn_calls = []
        self.attn_mha = self._run_attn_mha

    def _get_mla_kv_buffer(self, kv_indices, dtype, forward_batch):
        self.fetch_dtypes.append(dtype)
        return self.kv_a_normed, self.k_pe

    def kv_b_proj(self, x):
        return (self.kv,)

    def _run_attn_mha(
        self, q, k, v, forward_batch, save_kv_cache, key_value_num_tokens
    ):
        self.attn_calls.append((k, v, key_value_num_tokens))
        return torch.zeros(1), torch.zeros(1)

    _chunked_prefix_attn_mha = DeepseekMHAForwardMixin._chunked_prefix_attn_mha


NUM_TOKENS = 3


def _run_chunked_prefix(backend, q_dtype=torch.bfloat16):
    layer = _FakeLayer(NUM_TOKENS)
    forward_batch = SimpleNamespace(
        num_prefix_chunks=1,
        prefix_chunk_kv_indices=[torch.arange(NUM_TOKENS)],
        prefix_chunk_seq_lens_cpu=[[NUM_TOKENS]],
        prefix_chunk_starts_cpu=[[0]],
        set_prefix_chunk_idx=lambda i: None,
    )
    q = torch.zeros(1, dtype=q_dtype)
    with (
        patch.object(forward_mha, "get_attn_backend", return_value=backend),
        patch.object(
            forward_mha,
            "all_gather_kv_cache_for_mha_chunk_extend",
            side_effect=lambda kv_a, k_pe, *_: (kv_a, k_pe),
        ),
        patch.object(forward_mha, "merge_state_v2", lambda *a: None, create=True),
    ):
        layer._chunked_prefix_attn_mha(q, torch.zeros(1), torch.zeros(1), forward_batch)
    return layer


class TestChunkedPrefixAttnMhaDispatch(CustomTestCase):
    NUM_TOKENS = NUM_TOKENS

    def _run(self, backend):
        return _run_chunked_prefix(backend)

    def _expected_unfused_k(self, layer):
        kv = layer.kv.view(-1, layer.num_local_heads, layer.kv_dim)
        return torch.cat(
            [
                kv[..., : layer.qk_nope_head_dim],
                layer.k_pe.expand(-1, layer.num_local_heads, -1),
            ],
            dim=-1,
        )

    def test_pack_fn_receives_layer_and_its_result_is_used(self):
        seen = []
        packed_k = torch.ones(self.NUM_TOKENS, 2, 6, dtype=torch.bfloat16)
        packed_v = torch.full((self.NUM_TOKENS, 2, 4), 2.0, dtype=torch.bfloat16)

        def pack(layer_, k_nope, k_pe, v):
            seen.append((layer_, k_nope.shape, k_pe.shape, v.shape))
            return packed_k, packed_v

        layer = self._run(SimpleNamespace(pack_prefix_chunk_kv=pack))

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], layer.attn_mha)
        self.assertEqual(seen[0][1], (self.NUM_TOKENS, 2, 4))
        self.assertEqual(seen[0][2], (self.NUM_TOKENS, 1, 2))
        self.assertEqual(seen[0][3], (self.NUM_TOKENS, 2, 4))
        ((k, v, n),) = layer.attn_calls
        self.assertIs(k, packed_k)
        self.assertIs(v, packed_v)
        self.assertEqual(n, self.NUM_TOKENS)

    def test_pack_fn_returning_none_falls_back_to_cat(self):
        layer = self._run(SimpleNamespace(pack_prefix_chunk_kv=lambda *a: None))
        ((k, v, _),) = layer.attn_calls
        self.assertEqual(k.dtype, torch.bfloat16)
        torch.testing.assert_close(k, self._expected_unfused_k(layer))
        kv = layer.kv.view(-1, layer.num_local_heads, layer.kv_dim)
        torch.testing.assert_close(v, kv[..., layer.qk_nope_head_dim :])

    def test_backend_without_hook_uses_unfused_path_and_q_dtype(self):
        layer = self._run(SimpleNamespace())
        ((k, _, _),) = layer.attn_calls
        torch.testing.assert_close(k, self._expected_unfused_k(layer))
        # Without a hook the KV fetch dtype stays q.dtype (bf16 here).
        self.assertEqual(layer.fetch_dtypes, [torch.bfloat16])


class TestPrefixChunkKvFetchDtype(CustomTestCase):
    """The latent fetch dtype feeding kv_b_proj is declared by the backend, not
    implied by the pack hook existing: TRT-LLM (q still in model dtype) keeps
    q.dtype so FP16 models stay FP16; TokenSpeed (q already FP8) needs BF16."""

    def _resolve(self, backend, q_dtype):
        return forward_mha._resolve_prefix_chunk_kv_proj_dtype(
            backend, torch.zeros(1, dtype=q_dtype)
        )

    def test_backend_without_declaration_uses_q_dtype(self):
        backend = SimpleNamespace(pack_prefix_chunk_kv=lambda *a: None)
        self.assertEqual(self._resolve(backend, torch.float16), torch.float16)
        self.assertEqual(
            self._resolve(SimpleNamespace(), torch.bfloat16), torch.bfloat16
        )

    def test_trtllm_keeps_model_dtype(self):
        backend = object.__new__(TRTLLMMLABackend)
        for dtype in (torch.float16, torch.bfloat16):
            self.assertEqual(self._resolve(backend, dtype), dtype)

    def test_cutedsl_inherits_trtllm_dtype(self):
        backend = object.__new__(CuteDslMLABackend)
        self.assertEqual(self._resolve(backend, torch.float16), torch.float16)

    def test_tokenspeed_requests_bf16(self):
        backend = object.__new__(TokenspeedMLABackend)
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            self.assertEqual(self._resolve(backend, dtype), torch.bfloat16)

    def test_hybrid_wrappers_delegate(self):
        for cls, attr in (
            (HybridAttnBackend, "prefill_backend"),
            (HybridLinearAttnBackend, "full_attn_backend"),
        ):
            with self.subTest(wrapper=cls.__name__):
                trtllm = _wrapper(cls, attr, object.__new__(TRTLLMMLABackend))
                self.assertEqual(self._resolve(trtllm, torch.float16), torch.float16)
                tokenspeed = _wrapper(cls, attr, object.__new__(TokenspeedMLABackend))
                self.assertEqual(
                    self._resolve(tokenspeed, torch.float8_e4m3fn), torch.bfloat16
                )

    def test_fp16_model_with_hook_returning_none_fetches_fp16(self):
        backend = SimpleNamespace(pack_prefix_chunk_kv=lambda *a: None)
        layer = _run_chunked_prefix(backend, q_dtype=torch.float16)
        self.assertEqual(layer.fetch_dtypes, [torch.float16])

    def test_declared_bf16_fetch_with_fp8_q(self):
        packed_k = torch.ones(3, 2, 6, dtype=torch.bfloat16)
        packed_v = torch.ones(3, 2, 4, dtype=torch.bfloat16)
        backend = SimpleNamespace(
            pack_prefix_chunk_kv=lambda *a: (packed_k, packed_v),
            prefix_chunk_kv_proj_dtype=lambda q: torch.bfloat16,
        )
        layer = _run_chunked_prefix(backend, q_dtype=torch.float8_e4m3fn)
        self.assertEqual(layer.fetch_dtypes, [torch.bfloat16])


if __name__ == "__main__":
    unittest.main()
