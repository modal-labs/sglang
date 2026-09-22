"""Zero-on-hand-out must be ordered after the in-flight forward.

Under overlap scheduling, a request that finished in batch N is still in the
already-launched batch N+1, which writes its verify KV into slots that
processing N just freed. Freed pages are prepended, so they are the next ones
handed out. The allocator must wait on N+1's completion event before zeroing
those pages, or N+1's writes land after the zero and survive in the new
owner's page."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

PAGE = 64
NUM_PAGES = 8
SIZE = NUM_PAGES * PAGE


class _RecordingPool(MHATokenToKVPool):
    def zero_pages(self, page_ids):
        self.log.append(("zero", sorted(page_ids.tolist())))
        super().zero_pages(page_ids)


class _Event:
    def __init__(self, log):
        self.log = log

    def wait(self, stream=None):
        self.log.append(("wait",))


def _allocator():
    pool = _RecordingPool(
        size=SIZE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.log = []
    allocator = PagedTokenToKVPoolAllocator(
        SIZE, PAGE, torch.bfloat16, "cpu", pool, False, zero_pages_on_alloc=True
    )
    return allocator, pool.log


class ZeroAfterForwardTest(CustomTestCase):
    def test_page_freed_after_launch_waits_before_zero(self):
        allocator, log = _allocator()
        owner = allocator.alloc(PAGE)  # finished request's last page
        allocator.note_forward_launch(_Event(log))  # batch N+1 in flight
        allocator.free(owner)  # processing N releases it
        log.clear()
        handed = allocator.alloc(PAGE)  # next owner
        self.assertEqual(handed.tolist(), owner.tolist())  # LIFO reuse
        self.assertEqual(log, [("wait",), ("zero", [owner[0].item() // PAGE])])

    def test_no_free_since_launch_skips_the_wait(self):
        allocator, log = _allocator()
        allocator.note_forward_launch(_Event(log))
        log.clear()
        handed = allocator.alloc(PAGE)
        self.assertEqual(log, [("zero", [handed[0].item() // PAGE])])

    def test_free_before_launch_skips_the_wait(self):
        allocator, log = _allocator()
        owner = allocator.alloc(PAGE)
        allocator.free(owner)  # owner was filtered out of the next batch
        allocator.note_forward_launch(_Event(log))
        log.clear()
        allocator.alloc(PAGE)
        self.assertNotIn(("wait",), log)

    def test_free_processed_ahead_of_launch_stays_fenced(self):
        # Overlap disabled for this batch: N is processed (freeing the
        # finished request's pages) after N+1 was scheduled with that request
        # and before N+1 launches, so N+1 may still write those pages.
        allocator, log = _allocator()
        owner = allocator.alloc(PAGE)
        allocator.free(owner)
        allocator.carry_frees_into_next_launch()
        allocator.note_forward_launch(_Event(log))  # N+1
        log.clear()
        handed = allocator.alloc(PAGE)
        self.assertEqual(handed.tolist(), owner.tolist())
        self.assertEqual(log, [("wait",), ("zero", [owner[0].item() // PAGE])])

    def test_carry_covers_only_the_next_launch(self):
        # Once N+1 is followed by another launch, the scheduler has processed
        # N+1 before scheduling again, so the carried frees are settled.
        allocator, log = _allocator()
        owner = allocator.alloc(PAGE)
        allocator.free(owner)
        allocator.carry_frees_into_next_launch()
        allocator.note_forward_launch(_Event(log))
        allocator.note_forward_launch(_Event(log))
        log.clear()
        allocator.alloc(PAGE)
        self.assertNotIn(("wait",), log)

    def test_carry_without_frees_adds_no_wait(self):
        allocator, log = _allocator()
        allocator.carry_frees_into_next_launch()
        allocator.note_forward_launch(_Event(log))
        log.clear()
        allocator.alloc(PAGE)
        self.assertNotIn(("wait",), log)

    def test_grouped_free_sets_the_hazard(self):
        allocator, log = _allocator()
        owner = allocator.alloc(PAGE)
        allocator.note_forward_launch(_Event(log))
        allocator.free_group_begin()
        allocator.free(owner)
        allocator.free_group_end()
        log.clear()
        allocator.alloc(PAGE)
        self.assertEqual(log[0], ("wait",))

    def test_every_hand_out_before_next_launch_is_fenced(self):
        # Hand-outs may run on the schedule stream or the overlap plan stream,
        # so each one waits on the event from its current stream.
        allocator, log = _allocator()
        pages = allocator.alloc(2 * PAGE)
        allocator.note_forward_launch(_Event(log))
        allocator.free(pages)
        log.clear()
        allocator.alloc(PAGE)
        allocator.alloc(PAGE)
        self.assertEqual([entry[0] for entry in log], ["wait", "zero", "wait", "zero"])

    def test_without_launch_event_zero_still_runs(self):
        # Non-overlap scheduling never hands over an event: nothing is in flight.
        allocator, log = _allocator()
        owner = allocator.alloc(PAGE)
        allocator.free(owner)
        log.clear()
        allocator.alloc(PAGE)
        self.assertEqual(log, [("zero", [owner[0].item() // PAGE])])

    def test_zeroed_bytes_are_cleared(self):
        allocator, _ = _allocator()
        pool = allocator.get_kvcache()
        owner = allocator.alloc(PAGE)
        pool.k_buffer[0][owner] = 1
        allocator.note_forward_launch(_Event([]))
        allocator.free(owner)
        handed = allocator.alloc(PAGE)
        self.assertEqual(int(pool.k_buffer[0][handed].abs().sum()), 0)

    def test_composite_allocators_accept_the_launch_hooks(self):
        # The scheduler calls note_forward_launch() after every overlap forward
        # on whatever allocator it holds; these composites do not run the base
        # __init__, so the hooks must work without its instance state.
        for cls in (
            SWATokenToKVPoolAllocator,
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ):
            with self.subTest(cls=cls.__name__):
                allocator = cls.__new__(cls)
                allocator.carry_frees_into_next_launch()
                allocator.note_forward_launch(_Event([]))
                allocator.note_forward_launch(_Event([]))
                self.assertFalse(allocator._freed_since_forward_launch)


if __name__ == "__main__":
    unittest.main()
