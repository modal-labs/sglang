"""A latched batch_is_full must not skip the prefill pass while requests wait."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_pass(reevaluate: bool):
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock(**{"has_waiting_grammars.return_value": False})
    s.enable_hierarchical_cache = False
    s.server_args = SimpleNamespace(enable_flexkv=False)
    s.enable_priority_preemption = s.is_hybrid_swa = False
    s.reevaluate_batch_full = reevaluate
    s.chunked_req = None
    s.waiting_queue = [MagicMock()]
    # First gate after the early return; reaching it proves the pass ran.
    s.min_free_slots_delayer = MagicMock(**{"should_delay.return_value": True})
    s.get_num_allocatable_reqs = MagicMock(return_value=1)
    # Admission-block attribution state that Scheduler.__init__ sets.
    s._last_admission_block_cause = None
    s.metrics_reporter = MagicMock()
    running = SimpleNamespace(batch_is_full=True, reqs=[])
    s._get_new_batch_prefill_raw(
        prefill_delayer_single_pass=None, running_batch=running
    )
    return running.batch_is_full, s.min_free_slots_delayer.should_delay.called


class TestBatchIsFullReevaluate(CustomTestCase):
    def test_latched_flag_is_cleared_and_pass_runs(self):
        self.assertEqual(_run_pass(reevaluate=True), (False, True))

    def test_kill_switch_keeps_latch(self):
        self.assertEqual(_run_pass(reevaluate=False), (True, False))


if __name__ == "__main__":
    unittest.main()
