"""CPU regression tests for DFlash ring and HiCache registration."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheDFlashRingRegistration(unittest.TestCase):
    def test_ring_skips_draft_mirroring_without_touching_target_cache(self):
        components = {"full": object(), "swa": object(), "mamba": object()}
        controller = SimpleNamespace(set_draft_kv_pool=mock.Mock())
        tree_cache = SimpleNamespace(
            cache_controller=controller,
            components=components,
            dflash_draft_ring_reprefill_tail_tokens=4_224,
        )

        maybe_register_hicache_draft(
            tree_cache=tree_cache,
            draft_worker=SimpleNamespace(use_draft_ring=True),
            spec_algorithm=object(),
            server_args=object(),
            enable_hierarchical_cache=True,
            page_size=64,
        )

        controller.set_draft_kv_pool.assert_not_called()
        self.assertIs(tree_cache.components, components)
        self.assertEqual(
            tree_cache.dflash_draft_ring_reprefill_tail_tokens,
            4_224,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
