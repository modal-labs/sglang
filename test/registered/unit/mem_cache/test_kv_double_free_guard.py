"""The paged allocator's per-page allocated bitmap: a page that reaches free() while
not allocated (a cross-call double free) is dropped and counted instead of
being put in the free pool twice, and a correct alloc/free sequence sees
exactly the free pool the unguarded implementation produced.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python python -m pytest \
        test/registered/unit/mem_cache/test_kv_double_free_guard.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import random
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator import paged as paged_module
from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 4
NUM_PAGES = 16
DEVICE = "cpu"


def _make_kv_pool(size, page_size, device=DEVICE):
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
    )


def _make_paged(need_sort=False, num_pages=NUM_PAGES, page_size=PAGE_SIZE):
    size = num_pages * page_size
    return PagedTokenToKVPoolAllocator(
        size,
        page_size,
        torch.bfloat16,
        DEVICE,
        _make_kv_pool(size, page_size),
        need_sort,
    )


def _t(values):
    return torch.tensor(values, dtype=torch.int64, device=DEVICE)


def _pages(indices, page_size=PAGE_SIZE):
    return sorted(set((indices // page_size).tolist()))


def _drops(allocator):
    return allocator.double_free_page_drops_total(refresh=True)


def _out_of_pool(allocator):
    return allocator.out_of_pool_location_drops_total(refresh=True)


def _unguarded_paged_free(free_pages, free_index, page_size):
    """`PagedTokenToKVPoolAllocator.free` before the bitmap guard."""
    free_index = free_index[free_index >= page_size]
    return torch.cat((torch.unique(free_index // page_size), free_pages))


class _NaiveKernel:
    """Stands in for a Triton `kernel[(grid,)](...)` launch on CPU."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def _naive_alloc_extend(
    prefix_lens, seq_lens, last_loc, free_pages, out_indices, bs_bound, page_size
):
    alloc_extend_naive(
        prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, DEVICE
    )


def _naive_alloc_decode(seq_lens, last_loc, free_pages, out_indices, bs_bound, ps):
    new_page = 0
    for i in range(len(seq_lens)):
        if int(seq_lens[i]) % ps == 1:
            out_indices[i] = free_pages[new_page] * ps
            new_page += 1
        else:
            out_indices[i] = last_loc[i] + 1


def _patch_naive_kernels():
    return (
        mock.patch.object(
            paged_module, "alloc_extend_kernel", _NaiveKernel(_naive_alloc_extend)
        ),
        mock.patch.object(
            paged_module, "alloc_decode_kernel", _NaiveKernel(_naive_alloc_decode)
        ),
    )


class PagedDoubleFreeGuardTest(CustomTestCase):
    def _assert_pool_consistent(self, allocator):
        free = torch.cat((allocator.free_pages, allocator.release_pages))
        self.assertEqual(len(torch.unique(free)), len(free))
        expected = torch.ones_like(allocator.page_allocated)
        expected[0] = False
        expected[free] = False
        self.assertTrue(torch.equal(allocator.page_allocated, expected))

    def test_alloc_sets_bitmap_and_free_clears_it(self):
        allocator = _make_paged()
        self.assertFalse(allocator.page_allocated.any())
        out = allocator.alloc(2 * PAGE_SIZE)
        pages = _pages(out)
        self.assertEqual(allocator.page_allocated.nonzero().flatten().tolist(), pages)
        allocator.free(out)
        self.assertFalse(allocator.page_allocated.any())
        self._assert_pool_consistent(allocator)

    def test_cross_call_double_free_is_dropped_once(self):
        allocator = _make_paged()
        out = allocator.alloc(3 * PAGE_SIZE)
        page = _pages(out[:PAGE_SIZE])[0]

        allocator.free(out[:PAGE_SIZE])
        self.assertEqual(_drops(allocator), 0)
        allocator.free(out[:PAGE_SIZE])
        self.assertEqual(_drops(allocator), 1)
        self.assertEqual(int((allocator.free_pages == page).sum()), 1)
        self.assertEqual(allocator.available_size(), (NUM_PAGES - 2) * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

        # The next two hand-outs are distinct pages (no aliasing).
        first = allocator.alloc(PAGE_SIZE)
        second = allocator.alloc(PAGE_SIZE)
        self.assertNotEqual(_pages(first), _pages(second))
        self.assertEqual(_pages(first), [page])

    def test_same_call_duplicates_are_deduplicated_not_counted(self):
        allocator = _make_paged()
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(torch.cat((out, out)))
        self.assertEqual(_drops(allocator), 0)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

    def test_repeated_double_freed_page_is_counted_once_per_call(self):
        allocator = _make_paged()
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(out[:PAGE_SIZE])
        # Two copies of the freed page plus one live page in one call.
        allocator.free(torch.cat((out[:PAGE_SIZE], out[:PAGE_SIZE], out[PAGE_SIZE:])))
        self.assertEqual(_drops(allocator), 1)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

    def test_partial_page_free_and_double_free_mix(self):
        allocator = _make_paged()
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(out[:PAGE_SIZE])
        allocator.free(torch.cat((out[:1], out[PAGE_SIZE : PAGE_SIZE + 1])))
        self.assertEqual(_drops(allocator), 1)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

    def test_locations_past_the_pool_are_dropped_and_counted_apart(self):
        allocator = _make_paged()
        out = allocator.alloc(PAGE_SIZE)
        # Pages run 1..NUM_PAGES, so the first location past the pool is
        # (NUM_PAGES + 1) * PAGE_SIZE.
        beyond = _t([(NUM_PAGES + 1) * PAGE_SIZE, (NUM_PAGES + 4) * PAGE_SIZE])
        allocator.free(torch.cat((beyond, out)))
        self.assertEqual(_out_of_pool(allocator), 2)
        self.assertEqual(_drops(allocator), 0)
        self.assertEqual(allocator.reserved_location_drops_total(), 0)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

    def test_three_drop_classes_are_counted_separately(self):
        allocator = _make_paged()
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(out[:PAGE_SIZE])
        # One call carrying a reserved location, a cross-call double free of
        # the first page, three past-the-end locations and one valid page.
        allocator.free(
            torch.cat(
                (
                    _t([PAGE_SIZE - 1]),
                    out[:PAGE_SIZE],
                    _t([(NUM_PAGES + 1) * PAGE_SIZE + i for i in range(3)]),
                    out[PAGE_SIZE:],
                )
            )
        )
        self.assertEqual(allocator.reserved_location_drops_total(), 1)
        self.assertEqual(_drops(allocator), 1)
        self.assertEqual(_out_of_pool(allocator), 3)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

    def test_three_drop_classes_are_counted_separately(self):
        allocator = _make_paged()
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(out[:PAGE_SIZE])
        # One call carrying a reserved location, a cross-call double free of
        # the first page, three past-the-end locations and one valid page.
        allocator.free(
            torch.cat(
                (
                    _t([PAGE_SIZE - 1]),
                    out[:PAGE_SIZE],
                    _t([(NUM_PAGES + 1) * PAGE_SIZE + i for i in range(3)]),
                    out[PAGE_SIZE:],
                )
            )
        )
        self.assertEqual(allocator.reserved_location_drops_total(), 1)
        self.assertEqual(_drops(allocator), 1)
        self.assertEqual(_out_of_pool(allocator), 3)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

    def test_reserved_locations_are_attributed_to_reserved_counter(self):
        allocator = _make_paged()
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(torch.cat((_t([0, 1]), out)))
        self.assertEqual(allocator.reserved_location_drops_total(), 2)
        self.assertEqual(_drops(allocator), 0)
        self.assertFalse((allocator.free_pages == 0).any())
        self._assert_pool_consistent(allocator)

    def test_reserved_locations_are_attributed_to_reserved_counter(self):
        allocator = _make_paged()
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(torch.cat((_t([0, 1]), out)))
        self.assertEqual(allocator.reserved_location_drops_total(), 2)
        self.assertEqual(_drops(allocator), 0)
        self.assertFalse((allocator.free_pages == 0).any())
        self._assert_pool_consistent(allocator)

    def test_counter_is_device_side_until_refresh(self):
        allocator = _make_paged()
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(out)
        allocator.free(out)
        self.assertEqual(allocator.double_free_page_drops_total(), 0)
        self.assertEqual(int(allocator._double_free_page_drops_dev), 1)
        self.assertEqual(allocator.refresh_double_free_page_drops(), 1)
        self.assertEqual(allocator.double_free_page_drops_total(), 1)

    def test_double_free_logs_once(self):
        allocator = _make_paged()
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(out)
        with self.assertLogs("sglang.srt.mem_cache.allocator.base", "WARNING") as cm:
            allocator.free(out[:PAGE_SIZE])
            allocator.refresh_double_free_page_drops()
            allocator.free(out[PAGE_SIZE:])
            allocator.refresh_double_free_page_drops()
        self.assertEqual(len(cm.output), 1)
        self.assertIn("cross-call double free", cm.output[0])
        self.assertEqual(_drops(allocator), 2)

    def test_debug_mode_logs_caller_stack(self):
        allocator = _make_paged()
        allocator.debug_mode = True
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(out)
        with self.assertLogs("sglang.srt.mem_cache.allocator.base", "WARNING") as cm:
            allocator.free(out)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("caller stack", cm.output[0])
        self.assertIn("test_debug_mode_logs_caller_stack", cm.output[0])

    def test_alloc_extend_naive_sets_bitmap(self):
        allocator = _make_paged()
        extend_patch, decode_patch = _patch_naive_kernels()
        with extend_patch, decode_patch:
            # Two requests: one fresh (0 -> 6 tokens), one extending a page
            # that is partially filled (2 -> 9 tokens).
            first = allocator.alloc(PAGE_SIZE)
            prefix = _t([0, 2])
            seq = _t([6, 9])
            last_loc = _t([-1, int(first[1])])
            out = allocator.alloc_extend(
                prefix,
                prefix.cpu(),
                seq,
                seq.cpu(),
                last_loc,
                int((seq - prefix).sum()),
            )
        self.assertIsNotNone(out)
        touched = _pages(torch.cat((first, out)))
        self.assertEqual(allocator.page_allocated.nonzero().flatten().tolist(), touched)
        self.assertEqual(len(torch.unique(out)), len(out))
        self._assert_pool_consistent(allocator)
        allocator.free(torch.cat((first, out)))
        self.assertFalse(allocator.page_allocated.any())
        self.assertEqual(_drops(allocator), 0)

    def test_alloc_decode_naive_sets_bitmap(self):
        allocator = _make_paged()
        extend_patch, decode_patch = _patch_naive_kernels()
        with extend_patch, decode_patch:
            base = allocator.alloc(2 * PAGE_SIZE)
            # Request 0 is at a page boundary (needs a new page); request 1
            # still has room in its current page.
            seq = _t([PAGE_SIZE + 1, PAGE_SIZE + 2])
            last_loc = _t([int(base[PAGE_SIZE - 1]), int(base[PAGE_SIZE])])
            out = allocator.alloc_decode(seq, seq.cpu(), last_loc)
        self.assertIsNotNone(out)
        new_page = int(out[0]) // PAGE_SIZE
        self.assertNotIn(new_page, _pages(base))
        self.assertTrue(bool(allocator.page_allocated[new_page]))
        self.assertEqual(int(out[1]), int(last_loc[1]) + 1)
        self._assert_pool_consistent(allocator)

    def test_backup_restore_round_trip(self):
        allocator = _make_paged()
        kept = allocator.alloc(PAGE_SIZE)
        state = allocator.backup_state()
        snapshot = allocator.page_allocated.clone()

        speculative = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(kept)
        self.assertFalse(torch.equal(allocator.page_allocated, snapshot))

        allocator.restore_state(state)
        self.assertTrue(torch.equal(allocator.page_allocated, snapshot))
        self._assert_pool_consistent(allocator)
        # After rollback the speculative pages are free again and `kept` is
        # still live, so freeing each exactly once is clean.
        allocator.free(kept)
        self.assertEqual(_drops(allocator), 0)
        # ... and the rolled-back speculative pages are not live.
        allocator.free(speculative)
        self.assertEqual(_drops(allocator), 2)
        self._assert_pool_consistent(allocator)

    def test_backup_snapshot_is_not_aliased(self):
        allocator = _make_paged()
        state = allocator.backup_state()
        allocator.alloc(PAGE_SIZE)
        self.assertFalse(state[2].any())

    def test_free_group_buffers_and_drops_on_end(self):
        allocator = _make_paged()
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free_group_begin()
        allocator.free(out[:PAGE_SIZE])
        allocator.free(out[:PAGE_SIZE])
        allocator.free(out[PAGE_SIZE:])
        self.assertTrue(allocator.page_allocated.any())
        self.assertEqual(_drops(allocator), 0)
        allocator.free_group_end()
        # Buffered frees are concatenated into one call, so the repeated
        # page is a same-call duplicate and is deduplicated, not dropped.
        self.assertEqual(_drops(allocator), 0)
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self._assert_pool_consistent(allocator)

        # A page freed before the group and again inside it is a double free.
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(out)
        allocator.free_group_begin()
        allocator.free(out)
        allocator.free_group_end()
        self.assertEqual(_drops(allocator), 1)
        self._assert_pool_consistent(allocator)

    def test_need_sort_release_pages_path(self):
        allocator = _make_paged(need_sort=True)
        out = allocator.alloc(2 * PAGE_SIZE)
        allocator.free(out[:PAGE_SIZE])
        allocator.free(out[:PAGE_SIZE])
        page = _pages(out[:PAGE_SIZE])[0]
        self.assertEqual(int((allocator.release_pages == page).sum()), 1)
        self.assertEqual(_drops(allocator), 1)
        self._assert_pool_consistent(allocator)
        allocator.merge_and_sort_free()
        self.assertEqual(int((allocator.free_pages == page).sum()), 1)
        self.assertEqual(allocator.release_pages.numel(), 0)
        self._assert_pool_consistent(allocator)
        # A page sitting in release_pages is not allocated either.
        allocator.free(out[PAGE_SIZE:])
        allocator.free(out[PAGE_SIZE:])
        self.assertEqual(_drops(allocator), 2)
        self._assert_pool_consistent(allocator)

    def test_clear_resets_bitmap_and_keeps_cumulative_counter(self):
        allocator = _make_paged()
        out = allocator.alloc(PAGE_SIZE)
        allocator.free(out)
        allocator.free(out)
        self.assertEqual(_drops(allocator), 1)
        allocator.alloc(PAGE_SIZE)
        allocator.clear()
        self.assertFalse(allocator.page_allocated.any())
        self.assertEqual(allocator.page_allocated.shape, (NUM_PAGES + 1,))
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self.assertEqual(_drops(allocator), 1)
        self._assert_pool_consistent(allocator)

    def test_bitmap_lives_on_allocator_device(self):
        allocator = _make_paged()
        self.assertEqual(allocator.page_allocated.device.type, DEVICE)
        self.assertEqual(allocator.page_allocated.dtype, torch.bool)
        self.assertEqual(allocator._double_free_page_drops_dev.device.type, DEVICE)


class UnguardedParityFuzzTest(CustomTestCase):
    """A correct alloc/free sequence must leave the guard invisible: the free
    pool is bitwise what the unguarded free() produced, and the bitmap is the
    complement of the pool at every step."""

    def _run_paged(self, need_sort, seed, ops=10_000):
        rng = random.Random(seed)
        allocator = _make_paged(need_sort=need_sort, num_pages=32)
        reference = allocator.free_pages.clone()
        reference_release = allocator.release_pages.clone()
        live = []  # per-page token index tensors owned by the "requests"
        for _ in range(ops):
            if live and rng.random() < 0.5:
                # Free a random subset of live pages, some partially (an
                # unaligned tail), sometimes as one batched free.
                k = rng.randint(1, min(3, len(live)))
                chunk = [live.pop(rng.randrange(len(live))) for _ in range(k)]
                parts = []
                for page in chunk:
                    cut = rng.randint(1, PAGE_SIZE)
                    parts.append(page[:cut])
                free_index = torch.cat(parts)
                allocator.free(free_index)
                if allocator.need_sort:
                    reference_release = torch.cat(
                        (torch.unique(free_index // PAGE_SIZE), reference_release)
                    )
                else:
                    reference = _unguarded_paged_free(reference, free_index, PAGE_SIZE)
            else:
                n = rng.randint(1, 4)
                if allocator.need_sort and n > len(reference):
                    reference = torch.sort(torch.cat((reference, reference_release)))[0]
                    reference_release = reference_release[:0]
                if n > len(reference):
                    self.assertIsNone(allocator.alloc(n * PAGE_SIZE))
                    continue
                out = allocator.alloc(n * PAGE_SIZE)
                self.assertIsNotNone(out)
                self.assertTrue(
                    torch.equal(
                        out // PAGE_SIZE, reference[:n].repeat_interleave(PAGE_SIZE)
                    )
                )
                reference = reference[n:]
                live.extend(out.split(PAGE_SIZE))
            self.assertTrue(torch.equal(allocator.free_pages, reference))
            self.assertTrue(torch.equal(allocator.release_pages, reference_release))
        expected = torch.ones_like(allocator.page_allocated)
        expected[0] = False
        expected[torch.cat((allocator.free_pages, allocator.release_pages))] = False
        self.assertTrue(torch.equal(allocator.page_allocated, expected))
        self.assertEqual(_drops(allocator), 0)
        self.assertEqual(allocator.reserved_location_drops_total(), 0)

    def test_paged_lifo_matches_unguarded(self):
        self._run_paged(need_sort=False, seed=0xD0)

    def test_paged_need_sort_matches_unguarded(self):
        self._run_paged(need_sort=True, seed=0xD1)


@unittest.skipUnless(torch.cuda.is_available(), "Triton alloc kernels need a GPU")
class TritonAllocPathsSetBitmapTest(CustomTestCase):
    def _make(self):
        size = NUM_PAGES * PAGE_SIZE
        return PagedTokenToKVPoolAllocator(
            size,
            PAGE_SIZE,
            torch.bfloat16,
            "cuda",
            _make_kv_pool(size, PAGE_SIZE, "cuda"),
            False,
        )

    def test_alloc_extend_and_decode_kernels_mark_pages(self):
        allocator = self._make()
        dev = allocator.device
        first = allocator.alloc(PAGE_SIZE)
        prefix = torch.tensor([0, 2], dtype=torch.int64, device=dev)
        seq = torch.tensor([6, 9], dtype=torch.int64, device=dev)
        last_loc = torch.tensor([-1, int(first[1])], dtype=torch.int64, device=dev)
        out = allocator.alloc_extend(
            prefix, prefix.cpu(), seq, seq.cpu(), last_loc, int((seq - prefix).sum())
        )
        self.assertIsNotNone(out)
        touched = _pages(torch.cat((first, out)).cpu())
        self.assertEqual(
            allocator.page_allocated.nonzero().flatten().cpu().tolist(), touched
        )

        seq = torch.tensor([2 * PAGE_SIZE + 1], dtype=torch.int64, device=dev)
        last_loc = torch.tensor([int(out[5])], dtype=torch.int64, device=dev)
        dec = allocator.alloc_decode(seq, seq.cpu(), last_loc)
        self.assertIsNotNone(dec)
        new_page = int(dec[0]) // PAGE_SIZE
        self.assertNotIn(new_page, touched)
        self.assertTrue(bool(allocator.page_allocated[new_page]))

        allocator.free(torch.cat((first, out, dec)))
        self.assertFalse(allocator.page_allocated.any())
        allocator.free(dec)
        self.assertEqual(_drops(allocator), 1)


if __name__ == "__main__":
    unittest.main()
