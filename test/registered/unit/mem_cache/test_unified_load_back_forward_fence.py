"""The HiCache load-back H2D must be ordered behind the in-flight forward.

`ready_to_load_host_cache` records the load `start_event` on the schedule
stream. Under the overlap scheduler the destination slots of a load-back can
be slots freed (at result processing of batch N-1) by a request that batch N
is still writing, so the start_event must be recorded only after
`fence_state_read` has ordered the schedule stream behind batch N.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnifiedLoadBackForwardFence(CustomTestCase):
    def _make_cache(self, calls, queued=True):
        controller = SimpleNamespace(
            load_queue=[object()] if queued else [],
            start_loading=lambda: calls.append("start_loading") or 7,
        )
        return SimpleNamespace(
            cache_controller=controller,
            fence_state_read=lambda: calls.append("fence_state_read"),
        )

    def test_fence_precedes_start_loading(self):
        calls = []
        cache = self._make_cache(calls)
        producer = UnifiedRadixCache.ready_to_load_host_cache(cache)
        self.assertEqual(producer, 7)
        self.assertEqual(calls, ["fence_state_read", "start_loading"])

    def test_empty_queue_no_fence(self):
        calls = []
        cache = self._make_cache(calls, queued=False)
        self.assertEqual(UnifiedRadixCache.ready_to_load_host_cache(cache), -1)
        self.assertEqual(calls, [])

    def test_no_controller_no_fence(self):
        calls = []
        cache = self._make_cache(calls)
        cache.cache_controller = None
        self.assertEqual(UnifiedRadixCache.ready_to_load_host_cache(cache), 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
