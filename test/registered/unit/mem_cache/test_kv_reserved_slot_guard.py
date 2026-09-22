"""Reserved KV row 0 is the padded-token sink (padded CUDA-graph rows carry
``out_cache_loc == 0``). Writers must never store to it, and the paged
allocator must hand out zeroed page envelopes so a recycled page carries no
stale rows into page-granular attention kernels."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    resolve_reserved_skip_index,
)
from sglang.test.test_utils import CustomTestCase

PAGE = 16
SIZE = 8 * PAGE
LAYER = SimpleNamespace(layer_id=0)


def _mha_pool(page_size=PAGE, use_hnd=False):
    with envs.SGLANG_USE_HND_KVCACHE.override(use_hnd):
        return MHATokenToKVPool(
            size=SIZE,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )


def _mla_pool(page_size=PAGE, dtype=torch.bfloat16):
    return MLATokenToKVPool(
        size=SIZE,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )


def _fill_bytes(bufs, value=1):
    for buf in bufs:
        buf.view(torch.uint8).fill_(value)


def _byte_sum(buf, idx):
    return buf.view(torch.uint8)[idx].sum().item()


class ReservedSlotWriteGuardTest(CustomTestCase):
    """Pins the generic (index-assign + rezero_reserved_row) writers. The
    pools are built on CPU tensors, so the writer dispatch must not follow the
    process-level platform probe: on a host with a GPU visible it would hand
    these tensors to the CUDA-only store_cache kernel."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._platform = patch.multiple(memory_pool, _is_cuda=False, _is_hip=False)
        cls._platform.start()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_platform"):
            cls._platform.stop()
        super().tearDownClass()

    def test_resolve_reserved_skip_index_follows_flag(self):
        with envs.SGLANG_ENABLE_KV_RESERVED_SLOT_WRITE_GUARD.override(True):
            self.assertEqual(resolve_reserved_skip_index(), 0)
        with envs.SGLANG_ENABLE_KV_RESERVED_SLOT_WRITE_GUARD.override(False):
            self.assertEqual(resolve_reserved_skip_index(), -1)

    def test_mha_set_kv_buffer_skips_reserved_row(self):
        pool = _mha_pool()
        loc = torch.tensor([0, 70, 0, 71])
        k = torch.ones(4, 2, 16, dtype=torch.bfloat16)
        v = torch.full((4, 2, 16), 2.0, dtype=torch.bfloat16)
        pool.set_kv_buffer(LAYER, loc, k, v)
        for buf in (pool.k_buffer[0], pool.v_buffer[0]):
            self.assertEqual(buf[0].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(pool.k_buffer[0][70], k[1]))
        self.assertTrue(torch.equal(pool.v_buffer[0][71], v[3]))
        # Untouched layer stays zero.
        self.assertEqual(pool.k_buffer[1].abs().sum().item(), 0.0)

    def test_mla_set_kv_buffer_skips_reserved_row(self):
        pool = _mla_pool()
        loc = torch.tensor([0, 70, 71])
        k = torch.ones(3, 1, 48, dtype=torch.bfloat16)
        pool.set_kv_buffer(LAYER, loc, k, None)
        self.assertEqual(pool.kv_buffer[0][0].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(pool.kv_buffer[0][70], k[1]))
        self.assertTrue(torch.equal(pool.kv_buffer[0][71], k[2]))

    def test_guard_disabled_writes_reserved_row(self):
        with envs.SGLANG_ENABLE_KV_RESERVED_SLOT_WRITE_GUARD.override(False):
            pool = _mha_pool()
            loc = torch.tensor([0, 70])
            k = torch.ones(2, 2, 16, dtype=torch.bfloat16)
            pool.set_kv_buffer(LAYER, loc, k, k)
            self.assertTrue(torch.equal(pool.k_buffer[0][0], k[0]))


class PagedAllocatorZeroPagesTest(CustomTestCase):
    def _dirty(self, pool):
        for buf in (*pool.k_buffer, *pool.v_buffer):
            buf.fill_(1.0)

    def _pages_of(self, idx):
        return torch.unique(idx // PAGE)

    def _assert_pages_zero(self, pool, pages, zero=True):
        rows = pool.page_rows(pages)
        for buf in (*pool.k_buffer, *pool.v_buffer):
            total = buf[rows].abs().sum().item()
            if zero:
                self.assertEqual(total, 0.0)
            else:
                self.assertGreater(total, 0.0)

    def test_alloc_zeroes_handed_out_pages_only(self):
        pool = _mha_pool()
        self._dirty(pool)
        allocator = PagedTokenToKVPoolAllocator(
            SIZE, PAGE, torch.bfloat16, "cpu", pool, False, zero_pages_on_alloc=True
        )
        idx = allocator.alloc(2 * PAGE)
        handed = self._pages_of(idx)
        self.assertEqual(handed.numel(), 2)
        self._assert_pages_zero(pool, handed)
        untouched = torch.tensor(
            [p for p in range(1, SIZE // PAGE + 1) if p not in handed.tolist()]
        )
        self._assert_pages_zero(pool, untouched, zero=False)

    def test_alloc_without_flag_leaves_pages(self):
        pool = _mha_pool()
        self._dirty(pool)
        allocator = PagedTokenToKVPoolAllocator(
            SIZE, PAGE, torch.bfloat16, "cpu", pool, False, zero_pages_on_alloc=False
        )
        idx = allocator.alloc(PAGE)
        self._assert_pages_zero(pool, self._pages_of(idx), zero=False)

    def test_registered_pool_is_zeroed_too(self):
        target = _mha_pool()
        draft = _mha_pool()
        self._dirty(target)
        self._dirty(draft)
        allocator = PagedTokenToKVPoolAllocator(
            SIZE, PAGE, torch.bfloat16, "cpu", target, False, zero_pages_on_alloc=True
        )
        allocator.register_zero_pages_pool(draft)
        idx = allocator.alloc(PAGE)
        pages = self._pages_of(idx)
        self._assert_pages_zero(target, pages)
        self._assert_pages_zero(draft, pages)

    def test_register_skips_pools_without_own_zero_pages(self):
        # The base-class generic zero_pages is not a correctness guarantee for
        # exotic layouts: a pool class that does not define its own is skipped
        # and keeps its behavior, rather than being zeroed through inheritance.
        class _ExoticPool(_mha_pool().__class__):
            pass

        exotic = _ExoticPool(
            size=SIZE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self._dirty(exotic)
        allocator = PagedTokenToKVPoolAllocator(
            SIZE, PAGE, torch.bfloat16, "cpu", exotic, False, zero_pages_on_alloc=True
        )
        allocator.register_zero_pages_pool(exotic)
        idx = allocator.alloc(PAGE)
        self._assert_pages_zero(exotic, self._pages_of(idx), zero=False)

    def test_register_is_noop_when_disabled(self):
        target = _mha_pool()
        draft = _mha_pool()
        self._dirty(draft)
        allocator = PagedTokenToKVPoolAllocator(
            SIZE, PAGE, torch.bfloat16, "cpu", target, False, zero_pages_on_alloc=False
        )
        allocator.register_zero_pages_pool(draft)
        idx = allocator.alloc(PAGE)
        self._assert_pages_zero(draft, self._pages_of(idx), zero=False)

    def test_zero_pages_mla_and_hnd_layouts(self):
        mla = _mla_pool()
        for buf in mla.kv_buffer:
            buf.fill_(1.0)
        mla.zero_pages(torch.tensor([3]))
        rows = mla.page_rows(torch.tensor([3]))
        for buf in mla.kv_buffer:
            self.assertEqual(buf[rows].abs().sum().item(), 0.0)
            self.assertGreater(buf.abs().sum().item(), 0.0)

        hnd = _mha_pool(use_hnd=True)
        self.assertTrue(hnd.use_hnd)
        for buf in (*hnd.k_buffer, *hnd.v_buffer):
            buf.fill_(1.0)
        hnd.zero_pages(torch.tensor([3]))
        for buf in (*hnd.k_buffer, *hnd.v_buffer):
            self.assertEqual(buf[3].abs().sum().item(), 0.0)

    def test_zero_pages_fp8_storage(self):
        # float8 storage has no index_fill_ kernel; zero_pages must fill
        # through a byte view so recycled fp8 pages are actually cleared.
        mla = _mla_pool(dtype=torch.float8_e4m3fn)
        self.assertEqual(mla.kv_buffer[0].element_size(), 1)
        _fill_bytes(mla.kv_buffer)
        mla.zero_pages(torch.tensor([3]))
        rows = mla.page_rows(torch.tensor([3]))
        for buf in mla.kv_buffer:
            self.assertEqual(_byte_sum(buf, rows), 0)
            self.assertEqual(_byte_sum(buf, rows + PAGE), rows.numel() * buf.shape[-1])
            self.assertEqual(buf[2].abs().sum().item(), buf[2].numel())


if __name__ == "__main__":
    unittest.main()
