import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "backfill", ROOT / "python/sglang/srt/disaggregation/decode_backfill.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Budget = module.DecodeBackfillBudget


class BudgetTests(unittest.TestCase):
    def test_credit_persists_and_exhausts(self):
        b = Budget()
        b.begin("long")
        self.assertTrue(b.blocked("long", 100))
        for _ in range(4):
            b.begin("long")
            self.assertTrue(b.allows("short", 25))
            b.admitted("short", 25)
        self.assertFalse(b.blocked("long", 100))
        self.assertFalse(b.allows("another", 1))
        self.assertTrue(b.allows("long", 100))
        self.assertEqual(b.total_tokens, 100)

    def test_no_equal_or_larger_bypass(self):
        b = Budget()
        b.blocked("a", 100)
        self.assertFalse(b.allows("b", 100))
        self.assertFalse(b.allows("b", 101))

    def test_abort_or_head_completion_resets(self):
        b = Budget()
        b.blocked("a", 100)
        b.admitted("b", 90)
        b.begin("c")
        self.assertIsNone(b.head)
        b.blocked("c", 50)
        self.assertEqual(b.remaining, 50)
        b.admitted("c", 50)
        self.assertIsNone(b.head)

    def test_credit_not_renewed_by_another_failed_candidate(self):
        b = Budget()
        b.blocked("a", 100)
        b.admitted("short", 70)
        b.blocked("other", 500)
        self.assertEqual((b.head, b.remaining), ("a", 30))


class ReachedAllocation(Exception):
    pass


class DecodePathTests(unittest.TestCase):
    """Execute the actual admission method up to allocation, without CUDA."""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(
            (ROOT / "python/sglang/srt/disaggregation/decode.py").read_text()
        )
        c = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DecodePreallocQueue"
        )
        fn = next(
            n
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "pop_preallocated"
        )
        mod = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                fn,
            ],
            type_ignores=[],
        )
        ns = {"FINISH_ABORT": type("Abort", (), {}), "CLIP_MAX_NEW_TOKEN": 4096}
        exec(
            compile(ast.fix_missing_locations(mod), "<real decode admission>", "exec"),
            ns,
        )
        cls.pop = staticmethod(ns["pop_preallocated"])

    def make_queue(self, enabled=True, free=80000, slots=10, lengths=(123313, 472)):
        def entry(i, n):
            return NS(
                req=NS(
                    rid=str(i),
                    origin_input_ids=range(n),
                    finished_reason=None,
                    sampling_params=NS(max_new_tokens=32),
                ),
                waiting_for_input=True,
                is_rebootstrap=False,
            )

        q = NS(
            queue=[entry(i, n) for i, n in enumerate(lengths)],
            _backfill=Budget() if enabled else None,
        )
        q.scheduler = NS(
            running_batch=NS(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            server_args=NS(disaggregation_decode_enable_radix_cache=False),
        )
        q._resolve_pending_reqs = lambda: None
        q._update_handshake_waiters = lambda _: None
        q._uses_swa_tail_prealloc = lambda: False
        q._allocatable_token_budgets = lambda **_: free
        q._hicache_pending_restore_tokens = lambda: 0
        q.req_to_token_pool = NS(available_size=lambda: slots)
        q.req_to_metadata_buffer_idx_allocator = NS(available_size=lambda: 10)
        q._rebootstrap_prefill_len = lambda req: len(req.origin_input_ids)
        q._pre_alloc_fill_len = lambda req: (
            ((len(req.origin_input_ids) + 511) // 512) * 512
        )
        q.num_reserved_decode_tokens = 512

        def allocate(req, *args):
            raise ReachedAllocation(req.rid)

        q._pre_alloc = allocate
        return q

    def test_fifo_baseline_blocks(self):
        q = self.make_queue(enabled=False)
        self.assertEqual(self.pop(q), ([], []))
        self.assertEqual(len(q.queue), 2)

    def test_backfill_reaches_short_request(self):
        with self.assertRaisesRegex(ReachedAllocation, "^1$"):
            self.pop(self.make_queue())

    def test_head_admitted_first_if_it_fits(self):
        with self.assertRaisesRegex(ReachedAllocation, "^0$"):
            self.pop(self.make_queue(free=200000))

    def test_no_bypass_when_request_slots_exhausted(self):
        self.assertEqual(self.pop(self.make_queue(slots=0)), ([], []))

    def test_decode_safety_budget_still_enforced(self):
        self.assertEqual(self.pop(self.make_queue(free=100)), ([], []))

    def test_exhausted_credit_stops_bypass(self):
        q = self.make_queue()
        q._backfill.blocked("0", 123904)
        q._backfill.remaining = 0
        self.assertEqual(self.pop(q), ([], []))

    def test_aborted_head_does_not_leak_credit(self):
        q = self.make_queue()
        q._backfill.blocked("old", 99999)
        with self.assertRaisesRegex(ReachedAllocation, "^1$"):
            self.pop(q)
        self.assertEqual(q._backfill.head, "0")
        self.assertEqual(q._backfill.remaining, 123904)


if __name__ == "__main__":
    unittest.main()
