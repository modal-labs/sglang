"""The plain-TP fused MoE front must keep the router gate out of its merged
weight: the merged GEMM emits bf16 (or FP8-weight) columns, while routing
needs bf16 weights with fp32 logits from MoEGate."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_H = 64
_GATE_UP = 24
_EXPERTS = 16
_LATENT = 32


def _linear(rows: int, seed: int) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, _H, generator=gen).to(torch.bfloat16)
    return SimpleNamespace(weight=torch.nn.Parameter(weight, requires_grad=False))


def _make_owner(*, plain_front_has_router: bool) -> SimpleNamespace:
    return SimpleNamespace(
        _front_fp8=None,
        use_latent_moe=True,
        shared_experts=SimpleNamespace(gate_up_proj=_linear(_GATE_UP, 0)),
        gate=_linear(_EXPERTS, 1),
        routed_expert_down_proj=_linear(_LATENT, 2),
        _plain_front_has_router=plain_front_has_router,
        _target_fp8_front_group=None,
        _target_fp8=None,
        _front_w=None,
        _front_sizes=None,
        _front_is_ep_pair=False,
    )


def _merge(owner: SimpleNamespace, *, a2a_none: bool) -> None:
    backend = SimpleNamespace(is_none=lambda: a2a_none)
    with patch("sglang.srt.models.kimi_k3.get_moe_a2a_backend", return_value=backend):
        KimiK3MoE._merge_front_weights(owner)


class TestKimiK3FrontRouter(CustomTestCase):
    def test_plain_tp_front_excludes_router(self):
        owner = _make_owner(plain_front_has_router=False)
        gate_before = owner.gate.weight.detach().clone()
        gate_ptr = owner.gate.weight.data_ptr()
        gate_up = owner.shared_experts.gate_up_proj.weight.detach().clone()
        latent = owner.routed_expert_down_proj.weight.detach().clone()

        _merge(owner, a2a_none=True)

        self.assertEqual(owner._front_sizes, [_GATE_UP, _LATENT])
        self.assertFalse(owner._front_is_ep_pair)
        self.assertTrue(
            torch.equal(owner._front_w, torch.cat([gate_up, latent], dim=0))
        )
        # The router keeps its own bf16 storage, not a view of the front.
        self.assertEqual(owner.gate.weight.data_ptr(), gate_ptr)
        self.assertTrue(torch.equal(owner.gate.weight, gate_before))

    def test_hip_front_keeps_legacy_router_merge(self):
        owner = _make_owner(plain_front_has_router=True)

        _merge(owner, a2a_none=True)

        self.assertEqual(owner._front_sizes, [_GATE_UP, _EXPERTS, _LATENT])
        self.assertFalse(owner._front_is_ep_pair)

    def test_ep_pair_still_merges_router(self):
        owner = _make_owner(plain_front_has_router=False)

        _merge(owner, a2a_none=False)

        self.assertEqual(owner._front_sizes, [_EXPERTS, _LATENT])
        self.assertTrue(owner._front_is_ep_pair)


if __name__ == "__main__":
    unittest.main()
