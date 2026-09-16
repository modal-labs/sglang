"""PD offload must save the same boundary in CPU metadata and GPU state."""

import ast
import logging
import os
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace


def load_update_running_batch():
    # Run the actual scheduler method without importing CUDA worker dependencies.
    source = Path(
        os.environ.get(
            "SCHEDULER_SOURCE",
            Path(__file__).parents[2] / "python/sglang/srt/managers/scheduler.py",
        )
    ).read_text()
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_running_batch"
    )
    ns = {
        "DisaggregationMode": SimpleNamespace(DECODE="decode"),
        "TEST_RETRACT": True,
        "TEST_RETRACT_INTERVAL": 16,
        "logger": logging.getLogger(__name__),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "scheduler.py", "exec"), ns)
    return ns["update_running_batch"], ns


class PendingBlock:
    def __init__(self):
        self.output_ids = list(range(100))
        self.origin_input_ids = [1] * 20
        self.is_retracted = False
        self.finished = False
        self.gpu_committed = 104


class Batch:
    def __init__(self, events, capacity):
        self.events = events
        self.capacity = capacity
        self.reqs = [PendingBlock(), PendingBlock()]
        self.batch_is_full = True
        self.snapshots = []

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self):
        self.reqs = [req for req in self.reqs if not req.finished]

    def is_empty(self):
        return not self.reqs

    def check_decode_mem(self):
        return self.capacity["enough"]

    def retract_decode(self, args):
        req = self.reqs.pop()
        self.snapshots.append((len(req.output_ids), req.gpu_committed))
        self.events.append("offload")
        req.is_retracted = True
        return [req], 0.5, []

    def prepare_for_decode(self):
        self.events.append("prepare")


class RetractionOverlapBoundaryTest(unittest.TestCase):
    def setup_case(
        self,
        *,
        forced=True,
        enough=True,
        mode="decode",
        overlap=True,
        complete=False,
        free_on_result=False,
    ):
        method, ns = load_update_running_batch()
        ns["TEST_RETRACT"] = forced
        events = []
        capacity = {"enough": enough}
        batch = Batch(events, capacity)
        pending = list(batch.reqs)
        queue = deque([(pending, [100, 101, 102, 103])])

        def process_batch_result(reqs, accepted):
            events.append("settle")
            for index, req in enumerate(reqs):
                # Matches output processing's intentional retracted-request guard.
                if not req.is_retracted:
                    req.output_ids.extend(accepted)
                    req.finished = complete is True or (
                        complete == "first" and index == 0
                    )
            if free_on_result:
                capacity["enough"] = True

        scheduler = SimpleNamespace(
            enable_hierarchical_cache=False,
            enable_overlap=overlap,
            disaggregation_mode=mode,
            forward_ct=16,
            result_queue=queue,
            last_batch=pending,
            process_batch_result=process_batch_result,
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 1024),
            new_token_ratio_tracker=SimpleNamespace(
                current=0.5, decay_step=lambda: events.append("decay")
            ),
            tree_cache=SimpleNamespace(req_to_token_pool=SimpleNamespace()),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            server_args=SimpleNamespace(),
            _add_request_to_queue=lambda req, is_retracted: events.append("enqueue"),
        )
        return method, scheduler, batch, events

    def test_pending_accepted_block_is_settled_before_pd_snapshot(self):
        for forced, enough in ((True, True), (False, False)):
            with self.subTest(forced=forced):
                method, scheduler, batch, events = self.setup_case(
                    forced=forced, enough=enough
                )
                method(scheduler, batch)
                self.assertEqual(batch.snapshots, [(104, 104)])
                self.assertLess(events.index("settle"), events.index("offload"))
                self.assertFalse(scheduler.result_queue)
                self.assertIsNone(scheduler.last_batch)

    def test_settling_finished_requests_can_avoid_retraction(self):
        method, scheduler, batch, events = self.setup_case(
            forced=False, enough=False, free_on_result=True
        )
        method(scheduler, batch)
        self.assertEqual(batch.snapshots, [])
        self.assertEqual(events, ["settle", "decay", "prepare"])
        self.assertIsNone(scheduler.last_batch)

    def test_all_finished_during_settlement_returns_empty_batch(self):
        method, scheduler, batch, events = self.setup_case(
            forced=False, enough=False, complete=True
        )
        self.assertIs(method(scheduler, batch), batch)
        self.assertTrue(batch.is_empty())
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(events, ["settle"])

    def test_partial_completion_is_filtered_before_memory_recheck(self):
        method, scheduler, batch, events = self.setup_case(
            forced=False, enough=False, complete="first", free_on_result=True
        )
        method(scheduler, batch)
        self.assertEqual(batch.batch_size(), 1)
        self.assertEqual(len(batch.reqs[0].output_ids), 104)
        self.assertEqual(batch.snapshots, [])
        self.assertIsNone(scheduler.last_batch)

    def test_healthy_pd_steps_keep_overlap(self):
        method, scheduler, batch, events = self.setup_case(forced=False)
        method(scheduler, batch)
        self.assertEqual(events, ["decay", "prepare"])
        self.assertEqual(len(scheduler.result_queue), 1)
        self.assertIsNotNone(scheduler.last_batch)

    def test_other_scheduler_modes_do_not_drain_here(self):
        for mode, overlap in (("null", True), ("decode", False)):
            with self.subTest(mode=mode, overlap=overlap):
                method, scheduler, batch, events = self.setup_case(
                    mode=mode, overlap=overlap
                )
                method(scheduler, batch)
                self.assertNotIn("settle", events)
                self.assertEqual(len(scheduler.result_queue), 1)


if __name__ == "__main__":
    unittest.main()
