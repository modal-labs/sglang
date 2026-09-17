import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

import sglang.kernels.ops.attention.fla.chunk_delta_h as chunk_delta_h
import sglang.kernels.ops.attention.fla.op as fla_op
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@triton.jit
def _exp2(x):
    return tl.math.exp2(x)


@triton.jit
def _exp(x):
    return tl.exp(x)


class TestChunkDeltaHPaddedStateIndex(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layer_count = 2
        self.pool_size = 4
        self.head_count = 2
        self.key_dim = 64
        self.value_dim = 64
        self.sequence_length = 80
        self.pool = torch.randn(
            self.layer_count,
            self.pool_size + 1,
            self.head_count,
            self.value_dim,
            self.key_dim,
            dtype=torch.float32,
        )
        self.k = torch.randn(
            1,
            2 * self.sequence_length,
            self.head_count,
            self.key_dim,
            dtype=torch.float32,
        )
        self.w = torch.randn_like(self.k)
        self.u = torch.randn(
            1,
            2 * self.sequence_length,
            self.head_count,
            self.value_dim,
            dtype=torch.float32,
        )
        self.gk = -torch.randn_like(self.k).abs()
        self.cu_seqlens = torch.tensor([0, 80, 160], dtype=torch.int64)

    def _run(self, pool, k, w, u, gk, initial_state_indices, cu_seqlens):
        with (
            patch.object(fla_op, "exp", _exp),
            patch.object(fla_op, "exp2", _exp2),
            patch.object(chunk_delta_h, "exp", _exp),
            patch.object(chunk_delta_h, "exp2", _exp2),
        ):
            return chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                gk=gk,
                initial_state=pool[1],
                initial_state_indices=initial_state_indices,
                cu_seqlens=cu_seqlens,
                use_exp2=True,
            )

    def test_padded_row_does_not_touch_previous_layer_last_row(self):
        snapshot = self.pool.clone()
        self._run(
            self.pool,
            self.k,
            self.w,
            self.u,
            self.gk,
            torch.tensor([-1, 3], dtype=torch.int64),
            self.cu_seqlens,
        )

        self.assertTrue(torch.equal(self.pool[0], snapshot[0]))
        for row in range(self.pool_size + 1):
            if row != 3:
                self.assertTrue(torch.equal(self.pool[1, row], snapshot[1, row]))

    def test_real_row_matches_single_sequence_reference(self):
        batched_pool = self.pool.clone()
        h_batched, _ = self._run(
            batched_pool,
            self.k,
            self.w,
            self.u,
            self.gk,
            torch.tensor([-1, 3], dtype=torch.int64),
            self.cu_seqlens,
        )

        reference_pool = self.pool.clone()
        h_reference, _ = self._run(
            reference_pool,
            self.k[:, self.sequence_length :],
            self.w[:, self.sequence_length :],
            self.u[:, self.sequence_length :],
            self.gk[:, self.sequence_length :],
            torch.tensor([3], dtype=torch.int64),
            torch.tensor([0, self.sequence_length], dtype=torch.int64),
        )

        self.assertTrue(
            torch.allclose(
                batched_pool[1, 3],
                reference_pool[1, 3],
                atol=1e-4,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                h_batched[0, 2:4],
                h_reference[0, 0:2],
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_padded_row_starts_from_zero_state(self):
        h, _ = self._run(
            self.pool.clone(),
            self.k,
            self.w,
            self.u,
            self.gk,
            torch.tensor([-1, 3], dtype=torch.int64),
            self.cu_seqlens,
        )

        self.assertTrue(torch.equal(h[0, 0], torch.zeros_like(h[0, 0])))


if __name__ == "__main__":
    unittest.main()
