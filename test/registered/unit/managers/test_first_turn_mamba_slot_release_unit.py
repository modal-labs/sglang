import time
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _waiting_req(rid, mamba_pool_idx, req_pool_idx=None):
    class _Req:  # hashable: _abort_on_waiting_timeout dedupes via set()
        pass

    req = _Req()
    req.rid = rid
    req.session = SimpleNamespace(
        abort_req=Mock(), release_dropped_turn_mm_inputs=Mock()
    )
    req.multimodal_inputs = None
    req.req_pool_idx = req_pool_idx
    req.mamba_pool_idx = mamba_pool_idx
    req.kv = None
    req.priority = 5
    req.time_stats = SimpleNamespace(
        wait_queue_entry_time=time.perf_counter() - 100,
        trace_ctx=SimpleNamespace(abort=Mock()),
    )
    return req


def _scheduler_stub(waiting_queue):
    mamba_allocator = Mock()
    stub = SimpleNamespace(
        waiting_queue=waiting_queue,
        enable_hicache_storage=False,
        enable_hierarchical_cache=False,
        disaggregation_mode=DisaggregationMode.NULL,
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        ),
        tree_cache=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_allocator),
            supports_mamba=lambda: True,
        ),
    )
    stub._release_dropped_waiting_req_mm_inputs = (
        Scheduler._release_dropped_waiting_req_mm_inputs.__get__(stub)
    )
    stub._release_dropped_waiting_req_mamba_slot = (
        Scheduler._release_dropped_waiting_req_mamba_slot.__get__(stub)
    )
    return stub, mamba_allocator


class TestFirstTurnMambaSlotRelease(CustomTestCase):
    def test_waiting_timeout_releases_first_turn_mamba_slot(self):
        req = _waiting_req("turn-1", mamba_pool_idx=torch.tensor([3]))
        stub, mamba_allocator = _scheduler_stub([req])
        send_output = stub.ipc_channels.send_to_tokenizer.send_output

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(50.0):
            Scheduler._abort_on_waiting_timeout(stub)

        self.assertEqual(stub.waiting_queue, [])
        send_output.assert_called_once()
        mamba_allocator.free.assert_called_once()
        self.assertEqual(mamba_allocator.free.call_args[0][0].flatten().tolist(), [3])
        self.assertIsNone(req.mamba_pool_idx)

    def test_priority_eviction_releases_first_turn_mamba_slot(self):
        candidate = _waiting_req("turn-1", mamba_pool_idx=torch.tensor([3]))
        stub, mamba_allocator = _scheduler_stub([candidate])
        stub.max_queued_requests = 1
        stub.enable_priority_scheduling = True
        stub.schedule_low_priority_values_first = True
        recv_req = SimpleNamespace(rid="new-req", priority=0)

        aborted = Scheduler._abort_on_queued_limit(stub, recv_req)

        self.assertFalse(aborted)
        self.assertEqual(stub.waiting_queue, [])
        mamba_allocator.free.assert_called_once()
        self.assertEqual(mamba_allocator.free.call_args[0][0].flatten().tolist(), [3])
        self.assertIsNone(candidate.mamba_pool_idx)

    def test_waiting_timeout_keeps_slot_owned_by_session(self):
        req = _waiting_req("turn-2", mamba_pool_idx=torch.tensor([3]), req_pool_idx=7)
        stub, mamba_allocator = _scheduler_stub([req])

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(50.0):
            Scheduler._abort_on_waiting_timeout(stub)

        self.assertEqual(stub.waiting_queue, [])
        mamba_allocator.free.assert_not_called()
        self.assertEqual(req.mamba_pool_idx.tolist(), [3])


if __name__ == "__main__":
    import unittest

    unittest.main()
