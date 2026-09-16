"""Exercise the actual admission method without loading CUDA modules."""

import ast
import contextlib
import enum
import types
import unittest
from pathlib import Path


class Result(enum.Enum):
    OTHER = 1
    NO_TOKEN = 2
    CONTINUE = 3


def admission_method():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/managers/schedule_policy.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrefillAdder"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "add_one_req"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    scope = {
        "contextlib": contextlib,
        "AddReqResult": Result,
        "CLIP_MAX_NEW_TOKENS": 4096,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["add_one_req"]


def make_adder():
    a = types.SimpleNamespace(
        dsa_prefill_cp_in_seq_split=False,
        prefill_max_requests=None,
        can_run_list=[object()],
        tree_cache=types.SimpleNamespace(disable=False),
        page_size=512,
        rem_total_tokens=1000000,
        rem_chunk_tokens=2048,
        is_hybrid_swa=False,
        rem_input_tokens=16384,
        prefill_delayer_single_pass=None,
        dllm_config=None,
        new_chunked_req=None,
    )
    a._mamba_gap_budget_for_req = lambda r: 0
    a.ceil_paged_tokens = lambda n: (n + 511) // 512 * 512
    a._lock_node = lambda n: contextlib.nullcontext()
    a._req_inc_lock_ref = lambda r: None

    def budget(prefix, n, *args, **kwargs):
        a.rem_chunk_tokens -= n + 512

    a._update_prefill_budget = budget
    a.budget_state = lambda: Result.CONTINUE
    a.add_one_req = types.MethodType(admission_method(), a)
    return a


def request(n):
    r = types.SimpleNamespace(
        sampling_params=types.SimpleNamespace(ignore_eos=False, max_new_tokens=64),
        output_ids=[],
        full_untruncated_fill_ids=range(n),
        prefix_indices=[],
        host_hit_length=0,
        last_node=object(),
        retracted_stain=False,
    )
    r.needs_host_load_back = lambda: False
    r.set_extend_range = lambda a, b: None
    return r


class SingleContinuingPrefillTest(unittest.TestCase):
    def test_short_request_fits_but_second_continuation_is_rejected(self):
        a = make_adder()
        short = request(500)
        self.assertEqual(a.add_one_req(short, True, None), Result.CONTINUE)
        self.assertIn(short, a.can_run_list)
        self.assertEqual(a.add_one_req(request(5000), True, None), Result.OTHER)
        self.assertIsNone(a.new_chunked_req)
        self.assertGreaterEqual(a.rem_chunk_tokens, 0)

    def test_first_continuation_remains_allowed(self):
        a = make_adder()
        self.assertEqual(a.add_one_req(request(5000), False, None), Result.CONTINUE)
        self.assertIsNotNone(a.new_chunked_req)

    def test_ignore_eos_cannot_create_second_continuation(self):
        a = make_adder()
        a.tree_cache.disable = True
        r = request(5000)
        r.sampling_params.ignore_eos = True
        self.assertEqual(a.add_one_req(r, True, None), Result.OTHER)
        self.assertIsNone(a.new_chunked_req)


if __name__ == "__main__":
    unittest.main()
