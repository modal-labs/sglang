# CPU unit tests for the DFLASH draft-KV ring (--speculative-dflash-draft-ring):
# geometry sizing and the static request->slot map's addressing / aliasing /
# collision-freedom guarantees. No GPU, no server.

import unittest

from sglang.srt.speculative.dflash_utils import compute_dflash_draft_ring_geometry

WINDOW = 4096
BLOCK = 16
PAGE = 64


def ring_slot(pos: int, ring_tokens: int, page_offset: int, row: int) -> int:
    """The static map installed by _install_draft_ring_map / _ring_locs_flat:
    row r, position p -> page_offset + r * ring_tokens + p % ring_tokens."""
    return page_offset + row * ring_tokens + pos % ring_tokens


class TestDraftRingGeometry(unittest.TestCase):
    def test_page_aligned_and_covers_live_span(self):
        for page in (1, 16, 64, 128):
            for block in (1, 4, 16, 63):
                ring, keep = compute_dflash_draft_ring_geometry(
                    draft_window_size=WINDOW, block_size=block, page_size=page
                )
                self.assertEqual(ring % max(page, 1), 0)
                # Collision-freedom invariant: ctx keep + one draft block must
                # fit the ring with no aliasing.
                self.assertGreaterEqual(ring, keep + block)
                # Coverage: every future read window (window back from the
                # prefix, plus paged left-alignment slack) fits inside keep.
                self.assertGreaterEqual(keep, WINDOW)

    def test_rejects_degenerate_inputs(self):
        with self.assertRaises(ValueError):
            compute_dflash_draft_ring_geometry(
                draft_window_size=0, block_size=BLOCK, page_size=PAGE
            )
        with self.assertRaises(ValueError):
            compute_dflash_draft_ring_geometry(
                draft_window_size=WINDOW, block_size=0, page_size=PAGE
            )


class TestDraftRingMap(unittest.TestCase):
    def setUp(self):
        self.ring, self.keep = compute_dflash_draft_ring_geometry(
            draft_window_size=WINDOW, block_size=BLOCK, page_size=PAGE
        )
        self.page_offset = PAGE  # pool slot [0, page) is the reserved padding hole

    def test_rows_never_alias(self):
        a = {ring_slot(p, self.ring, self.page_offset, row=3) for p in range(self.ring)}
        b = {ring_slot(p, self.ring, self.page_offset, row=4) for p in range(self.ring)}
        self.assertEqual(len(a & b), 0)

    def test_padding_hole_never_addressed(self):
        for row in (0, 1, 17):
            for p in (0, 1, WINDOW, 10 * self.ring + 5):
                self.assertGreaterEqual(
                    ring_slot(p, self.ring, self.page_offset, row), self.page_offset
                )

    def test_live_span_is_collision_free(self):
        # At any prefix end E, the live positions are the kept ctx window
        # [E - keep, E) plus the draft block [E, E + BLOCK). All must map to
        # distinct ring slots.
        for end in (self.keep, WINDOW + 7, 123_457, 10 * self.ring + 1):
            live = list(range(max(0, end - self.keep), end + BLOCK))
            slots = {ring_slot(p, self.ring, self.page_offset, row=2) for p in live}
            self.assertEqual(len(slots), len(live), f"aliasing at end={end}")

    def test_aliasing_is_exactly_ring_period(self):
        p = WINDOW + 3
        self.assertEqual(
            ring_slot(p, self.ring, self.page_offset, 0),
            ring_slot(p + self.ring, self.ring, self.page_offset, 0),
        )
        self.assertNotEqual(
            ring_slot(p, self.ring, self.page_offset, 0),
            ring_slot(p + self.ring - 1, self.ring, self.page_offset, 0),
        )

    def test_ring_preserves_local_page_structure(self):
        # All positions of one logical page land in one ring page: required by
        # paged backends deriving page tables with pos // page_size math.
        for page_start in (0, PAGE, 5 * PAGE, (self.ring // PAGE + 2) * PAGE):
            ring_pages = {
                (ring_slot(p, self.ring, self.page_offset, row=1) - self.page_offset)
                // PAGE
                for p in range(page_start, page_start + PAGE)
            }
            self.assertEqual(len(ring_pages), 1, f"page split at {page_start}")

    def test_chunk_clamp_makes_long_chunks_safe(self):
        # A prefill chunk longer than the ring would self-collide; clamping to
        # the last `keep` positions (what the worker writes) never collides.
        end = 3 * self.ring + 11
        full_chunk = range(0, end)
        clamped = [p for p in full_chunk if p >= end - self.keep]
        slots = {ring_slot(p, self.ring, self.page_offset, row=0) for p in clamped}
        self.assertEqual(len(slots), len(clamped))


if __name__ == "__main__":
    unittest.main()
