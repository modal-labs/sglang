"""Zero-length (mamba-only) HiCache transfer ops must be safe no-ops.

A mamba-only load/backup op has zero KV pages. CUDA rejects a zero-sized grid
and the sgl_kernel C++ launcher divides by zero on empty input, so both the
JIT transfer wrappers and the host-pool entrypoints must return early.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.kvcache import hicache as hicache_ops
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _empty() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int64)


class TestZeroLengthJitTransfers(unittest.TestCase):
    def _forbid_jit_module(self):
        return mock.patch.object(
            hicache_ops,
            "_jit_hicache_module",
            side_effect=AssertionError(
                "JIT module must not be fetched for a zero-length transfer"
            ),
        )

    def test_one_layer_returns_before_jit_fetch(self):
        cache = torch.zeros(4, 128, dtype=torch.bfloat16)
        with self._forbid_jit_module():
            hicache_ops.transfer_hicache_one_layer(
                k_cache_dst=cache,
                v_cache_dst=cache,
                indices_dst=_empty(),
                k_cache_src=cache,
                v_cache_src=cache,
                indices_src=_empty(),
            )

    def test_all_layer_returns_before_jit_fetch(self):
        ptrs = torch.zeros(2, dtype=torch.uint64)
        with self._forbid_jit_module():
            hicache_ops.transfer_hicache_all_layer(
                k_ptr_dst=ptrs,
                v_ptr_dst=ptrs,
                indices_dst=_empty(),
                k_ptr_src=ptrs,
                v_ptr_src=ptrs,
                indices_src=_empty(),
                kv_cache_src_stride_bytes=256,
                kv_cache_dst_stride_bytes=256,
            )

    def test_one_layer_mla_returns_before_jit_fetch(self):
        cache = torch.zeros(4, 128, dtype=torch.bfloat16)
        with self._forbid_jit_module():
            hicache_ops.transfer_hicache_one_layer_mla(
                cache_dst=cache,
                indices_dst=_empty(),
                cache_src=cache,
                indices_src=_empty(),
            )

    def test_all_layer_mla_returns_before_jit_fetch(self):
        ptrs = torch.zeros(2, dtype=torch.uint64)
        with self._forbid_jit_module():
            hicache_ops.transfer_hicache_all_layer_mla(
                ptr_dst=ptrs,
                indices_dst=_empty(),
                ptr_src=ptrs,
                indices_src=_empty(),
                cache_src_stride_bytes=256,
                cache_dst_stride_bytes=256,
            )

    def test_nonempty_transfer_still_reaches_jit_module(self):
        # Control: proves the mock actually guards the JIT fetch.
        cache = torch.zeros(4, 128, dtype=torch.bfloat16)
        index = torch.zeros(1, dtype=torch.int64)
        with self._forbid_jit_module(), self.assertRaises(AssertionError):
            hicache_ops.transfer_hicache_one_layer(
                k_cache_dst=cache,
                v_cache_dst=cache,
                indices_dst=index,
                k_cache_src=cache,
                v_cache_src=cache,
                indices_src=index,
            )


class TestPoolHostZeroLengthGuards(unittest.TestCase):
    """Empty-op guards in the host pools also cover the sgl_kernel io-backend.

    Each host object is built via __new__ without any attributes: if the guard
    did not return first, the entrypoint would raise AttributeError.
    """

    def _load_args(self):
        return (None, _empty(), _empty(), 0, "kernel")

    def _backup_args(self):
        return (None, _empty(), _empty(), "kernel")

    def test_load_entrypoints_return_early_on_empty_indices(self):
        for cls in (
            MHATokenToKVPoolHost,
            MHATokenToKOnlyPoolHost,
            AsymmetricMHATokenToKVPoolHost,
            MLATokenToKVPoolHost,
        ):
            with self.subTest(cls=cls.__name__):
                host = cls.__new__(cls)
                self.assertIsNone(host.load_to_device_per_layer(*self._load_args()))

    def test_backup_entrypoints_return_early_on_empty_indices(self):
        for cls in (
            MHATokenToKVPoolHost,
            MHATokenToKOnlyPoolHost,
            AsymmetricMHATokenToKVPoolHost,
            MLATokenToKVPoolHost,
        ):
            with self.subTest(cls=cls.__name__):
                host = cls.__new__(cls)
                self.assertIsNone(
                    host.backup_from_device_all_layer(*self._backup_args())
                )


if __name__ == "__main__":
    unittest.main()
