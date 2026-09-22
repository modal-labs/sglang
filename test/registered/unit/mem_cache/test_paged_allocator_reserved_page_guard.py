"""Page 0 of the paged KV allocator is the padded-token sink and is never
in the free pool. ``free()`` must not recycle it when a caller hands back a
location below ``page_size`` (an unaligned slice, a stale ``req_to_token``
tail, or a literal 0), otherwise the next allocation hands the reserved page
to a real request."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

PAGE = 64
NUM_PAGES = 8
SIZE = NUM_PAGES * PAGE


def _allocator(need_sort=False):
    pool = MHATokenToKVPool(
        size=SIZE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return PagedTokenToKVPoolAllocator(
        SIZE, PAGE, torch.bfloat16, "cpu", pool, need_sort
    )


def _free_set(allocator):
    return set(allocator.free_pages.tolist()) | set(allocator.release_pages.tolist())


def _drain(allocator):
    """Allocate every free page so the next alloc must come from what free() adds."""
    handed = allocator.alloc(len(allocator.free_pages) * PAGE)
    assert handed is not None and handed.min().item() >= PAGE
    return handed


class PagedAllocatorReservedPageGuardTest(CustomTestCase):
    def test_fresh_pool_excludes_page_zero(self):
        allocator = _allocator()
        self.assertEqual(sorted(_free_set(allocator)), list(range(1, NUM_PAGES + 1)))

    def test_free_single_location_below_page_size_is_dropped(self):
        allocator = _allocator()
        _drain(allocator)
        allocator.free(torch.tensor([37], dtype=torch.int64))
        self.assertNotIn(0, _free_set(allocator))
        self.assertIsNone(allocator.alloc(PAGE))

    def test_free_slice_containing_zero_keeps_real_pages_only(self):
        allocator = _allocator()
        handed = _drain(allocator)
        # A slice that starts at location 0 and runs into a real page.
        slice_ = torch.cat((torch.arange(0, PAGE, dtype=torch.int64), handed[:PAGE]))
        allocator.free(slice_)
        self.assertNotIn(0, _free_set(allocator))
        self.assertEqual(_free_set(allocator), {handed[0].item() // PAGE})
        out = allocator.alloc(PAGE)
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.min().item(), PAGE)
        self.assertIsNone(allocator.alloc(PAGE))

    def test_free_only_sub_page_locations_is_a_noop(self):
        allocator = _allocator()
        before = allocator.free_pages.clone()
        allocator.free(torch.tensor([0, 1, PAGE - 1], dtype=torch.int64))
        self.assertTrue(torch.equal(allocator.free_pages, before))

    def test_need_sort_release_path_excludes_page_zero(self):
        allocator = _allocator(need_sort=True)
        handed = _drain(allocator)
        allocator.free(
            torch.cat((torch.tensor([37], dtype=torch.int64), handed[:PAGE]))
        )
        self.assertNotIn(0, allocator.release_pages.tolist())
        allocator.merge_and_sort_free()
        self.assertNotIn(0, allocator.free_pages.tolist())
        out = allocator.alloc(PAGE)
        self.assertGreaterEqual(out.min().item(), PAGE)

    def test_free_group_flush_excludes_page_zero(self):
        allocator = _allocator()
        handed = _drain(allocator)
        allocator.free_group_begin()
        allocator.free(torch.tensor([37], dtype=torch.int64))
        allocator.free(handed[:PAGE])
        allocator.free_group_end()
        self.assertEqual(_free_set(allocator), {handed[0].item() // PAGE})

    def test_debug_mode_asserts_never_see_page_zero(self):
        allocator = _allocator()
        allocator.debug_mode = True
        _drain(allocator)
        allocator.free(torch.tensor([37], dtype=torch.int64))
        allocator.free_pages = torch.tensor([0], dtype=torch.int64)
        with self.assertRaisesRegex(AssertionError, "reserved page 0"):
            allocator.free(torch.tensor([PAGE], dtype=torch.int64))


class ReservedLocationDropCounterTest(CustomTestCase):
    """Dropped reserved locations are counted and the first drop logs the
    caller stack once, so a leaking caller is visible rather than absorbed."""

    def test_counter_tallies_every_dropped_location(self):
        allocator = _allocator()
        handed = _drain(allocator)
        self.assertEqual(allocator.reserved_location_drops_total(), 0)
        allocator.free(handed[:PAGE])
        self.assertEqual(allocator.reserved_location_drops_total(), 0)
        allocator.free(torch.tensor([37], dtype=torch.int64))
        self.assertEqual(allocator.reserved_location_drops_total(), 1)
        slice_ = torch.cat(
            (torch.arange(0, PAGE, dtype=torch.int64), handed[PAGE : 2 * PAGE])
        )
        allocator.free(slice_)
        self.assertEqual(allocator.reserved_location_drops_total(), 1 + PAGE)

    def test_first_drop_logs_caller_stack_once(self):
        allocator = _allocator()
        _drain(allocator)
        with self.assertLogs(
            "sglang.srt.mem_cache.allocator.base", level="WARNING"
        ) as cm:
            allocator.free(torch.tensor([37], dtype=torch.int64))
            allocator.free(torch.tensor([3, 4], dtype=torch.int64))
        self.assertEqual(len(cm.records), 1)
        message = cm.output[0]
        self.assertIn(
            "PagedTokenToKVPoolAllocator.free() dropped 1 location(s)", message
        )
        self.assertIn("caller stack", message)
        self.assertIn("test_first_drop_logs_caller_stack_once", message)
        self.assertEqual(allocator.reserved_location_drops_total(), 3)

    def test_free_group_defers_counting_until_flush(self):
        allocator = _allocator()
        _drain(allocator)
        allocator.free_group_begin()
        allocator.free(torch.tensor([37], dtype=torch.int64))
        self.assertEqual(allocator.reserved_location_drops_total(), 0)
        allocator.free_group_end()
        self.assertEqual(allocator.reserved_location_drops_total(), 1)

    def test_pool_stats_export_reserved_location_drops(self):
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )
        from sglang.srt.observability.metrics_collector import SchedulerStats

        allocator = _allocator()
        _drain(allocator)
        allocator.free(torch.tensor([37, 38], dtype=torch.int64))
        observer = SchedulerPoolStatsObserver(
            tree_cache=SimpleNamespace(evictable_size=lambda: 0),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=None,
            session_controller=None,
            hisparse_coordinator=None,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=SIZE,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )
        stats = SchedulerStats()
        observer.get_pool_stats().update_scheduler_stats(stats)
        self.assertEqual(stats.kv_reserved_location_drops, 2)


if __name__ == "__main__":
    unittest.main()
