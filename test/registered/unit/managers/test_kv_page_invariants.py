"""Unit tests for SGLANG_CHECK_KV_PAGE_INVARIANTS: watermark + double-free checks."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 256


class _BackendPagedAllocator(PagedTokenToKVPoolAllocator):
    """A backend subtype whose allocation overrides do not maintain the bitmap."""


def _make_checker(
    page_size=_PAGE_SIZE,
    row_width=4096,
    num_reqs=8,
    free_pages=None,
    page_allocated=None,
    allocator_cls=PagedTokenToKVPoolAllocator,
):
    rtt = torch.zeros((num_reqs, row_width), dtype=torch.int32)
    rtp = SimpleNamespace(req_to_token=rtt)
    if free_pages is None:
        free_pages = torch.arange(num_reqs * 4, dtype=torch.int64)
    alloc = object.__new__(allocator_cls)
    alloc.page_size = page_size
    alloc.free_pages = free_pages
    alloc.release_pages = torch.empty(0, dtype=torch.int64)
    alloc.page_allocated = page_allocated
    alloc.double_free_page_drops_total = lambda refresh=False: 0
    tc = SimpleNamespace(slots={})
    _ps, _rtp, _alloc, _tc = page_size, rtp, alloc, tc

    class _FakeChecker:
        page_size = _ps
        req_to_token_pool = _rtp
        token_to_kv_pool_allocator = _alloc
        tree_cache = _tc
        get_last_batch = lambda self: None
        count_memory_leak_warnings = 0

        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker as _RIC,
        )

        _check_kv_page_invariants = _RIC._check_kv_page_invariants
        refresh_double_free_page_drops = _RIC.refresh_double_free_page_drops

    return _FakeChecker(), rtt, tc, alloc


class _FakeReq:
    def __init__(self, rid, rpi, committed, allocated):
        self.rid = rid
        self.req_pool_idx = rpi
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(kv_allocated_len=allocated, swa_evicted_seqlen=0)


class _FakeSlot:
    def __init__(self, rpi, committed, allocated):
        self.req_pool_idx = rpi
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(kv_allocated_len=allocated, swa_evicted_seqlen=0)
        self.is_holding_kv = True


class TestKVPageInvariants(CustomTestCase):
    def test_clean_layout_no_warning(self):
        chk, rtt, tc, alloc = _make_checker(
            free_pages=torch.arange(100, 200, dtype=torch.int64)
        )
        rtt[0, :256] = torch.arange(_PAGE_SIZE, 2 * _PAGE_SIZE)  # req 0 owns page 1
        rtt[1, :256] = torch.arange(2 * _PAGE_SIZE, 3 * _PAGE_SIZE)  # req 1 owns page 2
        chk.get_last_batch = lambda: SimpleNamespace(
            reqs=[_FakeReq("a", 0, 256, 256), _FakeReq("b", 1, 200, 256)]
        )
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)

    def test_committed_gt_allocated_raises(self):
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[_FakeReq("a", 0, 145, 144)])
        with self.assertRaises(AssertionError):
            chk._check_kv_page_invariants()

    def test_slot_committed_gt_allocated_raises(self):
        chk, rtt, tc, alloc = _make_checker()
        chk.get_last_batch = lambda: None
        tc.slots = {"s1": _FakeSlot(0, 145, 144)}
        with self.assertRaises(AssertionError):
            chk._check_kv_page_invariants()

    def test_owner_references_free_page_raises(self):
        # req 0 owns page 5, but page 5 is in the free pool -> use-after-free.
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([5, 6, 7]))
        rtt[0, :3] = torch.tensor(
            [5 * _PAGE_SIZE, 5 * _PAGE_SIZE + 1, 5 * _PAGE_SIZE + 2]
        )
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[_FakeReq("a", 0, 3, 3)])
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_owner_references_reserved_page_zero_raises(self):
        # Page 0 is never in the free pool, so Check A cannot see it; a real
        # row pointing at any location < page_size must still be flagged.
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([5, 6, 7]))
        rtt[0, :3] = torch.tensor([10 * _PAGE_SIZE, 10 * _PAGE_SIZE + 1, 37])
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[_FakeReq("a", 0, 3, 3)])
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_free_pool_duplicate_raises(self):
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([3, 3, 4]))
        rtt[0, :1] = torch.tensor([10 * _PAGE_SIZE])  # owner page 10, not in free
        chk.get_last_batch = lambda: SimpleNamespace(reqs=[_FakeReq("a", 0, 1, 1)])
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    # Check C: page_allocated bitmap == complement of the free pool.
    @staticmethod
    def _bitmap(num_pages, free_pages):
        bitmap = torch.ones(num_pages + 1, dtype=torch.bool)
        bitmap[0] = False
        bitmap[free_pages] = False
        return bitmap

    def test_bitmap_matches_free_pool_no_warning(self):
        free_pages = torch.tensor([3, 4, 7])
        chk, rtt, tc, alloc = _make_checker(
            free_pages=free_pages, page_allocated=self._bitmap(8, free_pages)
        )
        chk.get_last_batch = lambda: None
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)

    def test_bitmap_counts_release_pages_as_free(self):
        free_pages = torch.tensor([3, 4])
        chk, rtt, tc, alloc = _make_checker(
            free_pages=free_pages, page_allocated=self._bitmap(8, [3, 4, 7])
        )
        alloc.release_pages = torch.tensor([7])
        chk.get_last_batch = lambda: None
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)

    def test_bitmap_set_for_free_page_raises(self):
        # Page 4 is in the free pool but its allocated bit is still set.
        free_pages = torch.tensor([3, 4, 7])
        bitmap = self._bitmap(8, free_pages)
        bitmap[4] = True
        chk, rtt, tc, alloc = _make_checker(
            free_pages=free_pages, page_allocated=bitmap
        )
        chk.get_last_batch = lambda: None
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_bitmap_clear_for_owned_page_raises(self):
        # Page 5 is neither free nor marked allocated (a leaked bit).
        free_pages = torch.tensor([3, 4, 7])
        bitmap = self._bitmap(8, free_pages)
        bitmap[5] = False
        chk, rtt, tc, alloc = _make_checker(
            free_pages=free_pages, page_allocated=bitmap
        )
        chk.get_last_batch = lambda: None
        with self.assertRaises(ValueError):
            chk._check_kv_page_invariants()

    def test_no_bitmap_skips_check_c(self):
        chk, rtt, tc, alloc = _make_checker(free_pages=torch.tensor([3, 4]))
        chk.get_last_batch = lambda: None
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)

    def test_idle_tick_refreshes_double_free_page_drops(self):
        chk, rtt, tc, alloc = _make_checker()
        seen = []
        alloc.double_free_page_drops_total = lambda refresh=False: seen.append(refresh)
        chk.refresh_double_free_page_drops()
        self.assertEqual(seen, [True])

    def test_backend_subclass_without_bitmap_updates_skips_check_c(self):
        # A backend can hand out pages without updating the inherited bitmap.
        chk, _, _, _ = _make_checker(
            free_pages=torch.tensor([3, 4]),
            page_allocated=torch.zeros(9, dtype=torch.bool),
            allocator_cls=_BackendPagedAllocator,
        )
        chk._check_kv_page_invariants()
        self.assertEqual(chk.count_memory_leak_warnings, 0)

    def test_unsupported_allocators_skip_idle_refresh(self):
        for allocator_cls in (
            SWATokenToKVPoolAllocator,
            HiSparseTokenToKVPoolAllocator,
            _BackendPagedAllocator,
        ):
            with self.subTest(allocator_cls=allocator_cls.__name__):
                chk, _, _, _ = _make_checker()
                # Composite constructors do not initialize base counters.
                chk.token_to_kv_pool_allocator = object.__new__(allocator_cls)
                chk.refresh_double_free_page_drops()

    @staticmethod
    def _pool_stats(allocator):
        def token_info():
            return PoolStats(
                full_num_used=0,
                full_token_usage=0,
                full_available_size=8,
                full_evictable_size=0,
            )

        observer = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            is_hybrid_swa=isinstance(allocator, SWATokenToKVPoolAllocator),
            is_hybrid_ssm=False,
            enable_hisparse=isinstance(allocator, HiSparseTokenToKVPoolAllocator),
            _get_token_info=token_info,
            _get_swa_token_info=token_info,
            _get_hisparse_token_info=lambda stats: stats,
        )
        return SchedulerPoolStatsObserver.get_pool_stats(observer)

    def test_plain_paged_pool_stats_report_drop_counters(self):
        alloc = object.__new__(PagedTokenToKVPoolAllocator)
        alloc.reserved_location_drops = 3
        alloc.double_free_page_drops = 4
        alloc.out_of_pool_location_drops = 5
        stats = self._pool_stats(alloc)
        self.assertEqual(
            (
                stats.reserved_location_drops,
                stats.double_free_page_drops,
                stats.out_of_pool_location_drops,
            ),
            (3, 4, 5),
        )

    def test_unsupported_pool_stats_do_not_read_missing_counters(self):
        for allocator_cls in (
            SWATokenToKVPoolAllocator,
            HiSparseTokenToKVPoolAllocator,
            _BackendPagedAllocator,
        ):
            with self.subTest(allocator_cls=allocator_cls.__name__):
                stats = self._pool_stats(object.__new__(allocator_cls))
                self.assertEqual(
                    (
                        stats.reserved_location_drops,
                        stats.double_free_page_drops,
                        stats.out_of_pool_location_drops,
                    ),
                    (0, 0, 0),
                )


if __name__ == "__main__":
    unittest.main()
