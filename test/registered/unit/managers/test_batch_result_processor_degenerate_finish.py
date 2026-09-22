"""A spec-v2 verify whose target row had no positive finite mass must end the
request at that step: none of the verify's tokens (the run ends on the
reject-sampler sentinel, [PAD] on Kimi-K3) are committed or emitted, and the
request finishes with an engine-fault FINISH_ABORT whatever its EOS set or
ignore_eos. With the overlap scheduler, the in-flight step that still carries
the finished request emits nothing more and releases nothing twice."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    reject_sampling_sentinel_token,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase

VOCAB = 64
SENTINEL = reject_sampling_sentinel_token(VOCAB)
EOS_TOKEN_ID = 2
STRIDE = 4


class _FakeTokenizer:
    eos_token_id = EOS_TOKEN_ID
    additional_stop_token_ids = None


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False


class _FakeForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


class _SentinelRejectingGrammar:
    """Grammar that raises on the sentinel, as a real grammar does for a token
    outside its language."""

    def __init__(self):
        self.accepted = []
        self.finished = False

    def accept_token(self, token_id: int):
        if token_id == SENTINEL:
            raise ValueError("token not allowed by grammar")
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = _FakeForwardMode()
        self.return_logprob = False
        self.spec_algorithm = _FakeSpecAlgorithm()

    def batch_size(self) -> int:
        return len(self.reqs)


class _Streamer:
    def __init__(self):
        self.calls = []

    def stream_output(self, reqs, return_logprob):
        # Snapshot what a detokenizer would see for each finished req.
        self.calls.append(
            [
                (req.rid, list(req.output_ids_through_stop), req.finished_reason)
                for req in reqs
            ]
        )


def _make_processor(enable_overlap: bool, streamer: _Streamer, observed=None):
    observed = [] if observed is None else observed
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=enable_overlap,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(
            enable_metrics=False,
            disaggregation_decode_enable_offload_kvcache=False,
            enable_hisparse=False,
        ),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=lambda: None, free_group_end=lambda: None
        ),
        tree_cache=mock.sentinel.tree_cache,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(
            num_generated_tokens=0,
            forward_ct_decode=0,
            update_spec_metrics=lambda *a, **k: observed.append(("metrics", a, k)),
            report_decode_stats=lambda *a, **k: observed.append(("decode_stats", a, k)),
        ),
        draft_worker=None,
        model_worker=SimpleNamespace(
            on_verify_complete_cpu=lambda *a, **k: observed.append(("adaptive", a, k))
        ),
        logprob_result_processor=None,
        output_streamer=streamer,
        abort_request=lambda *a, **k: None,
    )


def _make_req(rid, eos_token_ids, ignore_eos=False, max_new_tokens=1_000):
    sp = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
    sp.normalize(tokenizer=_FakeTokenizer())
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1, 3, 5]),
        sampling_params=sp,
        eos_token_ids=set(eos_token_ids),
        vocab_size=VOCAB,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", [7])
    req.kv_committed_len = len(req.origin_input_ids) + 1
    return req


def _make_result(rows, target_degenerate):
    """rows: per-req accepted run (<= STRIDE tokens), padded to STRIDE."""
    flat, accept_lens = [], []
    for run in rows:
        flat.extend(run + [0] * (STRIDE - len(run)))
        accept_lens.append(len(run))
    return SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(hidden_states=None, customized_info=None),
        next_token_ids=torch.tensor(flat, dtype=torch.long),
        can_run_cuda_graph=True,
        accept_lens=torch.tensor(accept_lens, dtype=torch.int32),
        speculative_num_draft_tokens=STRIDE,
        num_correct_drafts=sum(max(n - 1, 0) for n in accept_lens),
        num_correct_drafts_per_req_cpu=[max(n - 1, 0) for n in accept_lens],
        num_block_accept_tokens=0,
        num_cap_tokens=0,
        block_accept_lens=None,
        cap_lens=None,
        target_degenerate=torch.tensor(target_degenerate, dtype=torch.bool),
        grammar_advanced=False,
    )


class TestDegenerateVerifyEndsRequest(CustomTestCase):
    def setUp(self):
        released = self.released = []
        patches = [
            mock.patch.object(
                batch_result_processor,
                "release_kv_cache",
                lambda req, tree_cache, is_insert=True: released.append(
                    (req.rid, req.spec_target_degenerate, req.kv_committed_len)
                ),
            ),
            mock.patch.object(
                batch_result_processor,
                "get_server_args",
                lambda: SimpleNamespace(enable_mamba_extra_buffer_lazy=lambda: False),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _decode(self, proc, reqs, rows, degenerate):
        proc.process_batch_result_decode(
            _FakeBatch(reqs), _make_result(rows, degenerate)
        )

    def _assert_engine_fault(self, req):
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        reason = req.finished_reason.to_json()
        self.assertEqual(reason["type"], "abort")
        self.assertEqual(reason["status_code"], HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertIn("engine fault", reason["message"])

    def _run_step(self, **req_kwargs):
        streamer = _Streamer()
        proc = _make_processor(enable_overlap=False, streamer=streamer)
        bad = _make_req("bad", **req_kwargs)
        good = _make_req("good", eos_token_ids={EOS_TOKEN_ID})
        committed_before = bad.kv_committed_len
        # bad: two accepted drafts then the sentinel bonus from its degenerate row.
        self._decode(proc, [bad, good], [[11, 13, SENTINEL], [21, 22]], [True, False])
        return streamer, bad, good, committed_before

    def test_sentinel_outside_eos_set_still_ends_request_without_pad(self):
        # The production recipe has no sentinel EOS override: before the fix
        # this request kept decoding and emitted [PAD] until max_new_tokens.
        streamer, bad, good, committed_before = self._run_step(
            eos_token_ids={EOS_TOKEN_ID}
        )

        self._assert_engine_fault(bad)
        self.assertEqual(list(bad.output_ids), [7])
        self.assertNotIn(SENTINEL, bad.output_ids)
        self.assertEqual(bad.kv_committed_len, committed_before)
        # The streamed final chunk carries no token from the degenerate verify.
        (step,) = streamer.calls
        self.assertEqual(step[0][0], "bad")
        self.assertEqual(step[0][1], [7])
        self.assertIsInstance(step[0][2], FINISH_ABORT)
        # Released once, still flagged so the radix / HiCache insert is skipped.
        self.assertEqual(self.released, [("bad", True, committed_before)])
        # Its batch neighbour is unaffected.
        self.assertFalse(good.finished())
        self.assertEqual(list(good.output_ids), [7, 21, 22])
        self.assertFalse(good.spec_target_degenerate)

    def test_ignore_eos_does_not_keep_degenerate_request_alive(self):
        _, bad, _, _ = self._run_step(
            eos_token_ids={EOS_TOKEN_ID, SENTINEL}, ignore_eos=True
        )
        self._assert_engine_fault(bad)
        self.assertEqual(list(bad.output_ids), [7])

    def test_sentinel_in_eos_set_reports_fault_not_natural_stop(self):
        _, bad, _, _ = self._run_step(eos_token_ids={EOS_TOKEN_ID, SENTINEL})
        self._assert_engine_fault(bad)
        self.assertNotIn(SENTINEL, bad.output_ids)

    def test_degenerate_on_last_allowed_step_reports_fault_not_length(self):
        _, bad, _, _ = self._run_step(eos_token_ids={EOS_TOKEN_ID}, max_new_tokens=2)
        self._assert_engine_fault(bad)

    def test_grammar_request_reports_engine_fault_and_skips_the_run(self):
        streamer = _Streamer()
        proc = _make_processor(enable_overlap=False, streamer=streamer)
        bad = _make_req("bad", eos_token_ids={EOS_TOKEN_ID})
        bad.grammar = _SentinelRejectingGrammar()
        good = _make_req("good", eos_token_ids={EOS_TOKEN_ID})
        good.grammar = _SentinelRejectingGrammar()

        self._decode(proc, [bad, good], [[11, 13, SENTINEL], [21, 22]], [True, False])

        # The dropped run never reaches the grammar, so its rejection of the
        # sentinel cannot replace the engine fault with a status-less abort.
        self.assertEqual(bad.grammar.accepted, [])
        self._assert_engine_fault(bad)
        self.assertEqual(list(bad.output_ids), [7])
        self.assertEqual(good.grammar.accepted, [21, 22])
        self.assertEqual(list(good.output_ids), [7, 21, 22])
        self.assertFalse(good.finished())

    def test_dropped_run_is_excluded_from_accept_accounting(self):
        observed = []
        proc = _make_processor(
            enable_overlap=False, streamer=_Streamer(), observed=observed
        )
        bad = _make_req("bad", eos_token_ids={EOS_TOKEN_ID})
        good = _make_req("good", eos_token_ids={EOS_TOKEN_ID})
        result = _make_result([[11, 13, SENTINEL], [21, 22]], [True, False])
        result.block_accept_lens = torch.tensor([3, 2], dtype=torch.int32)
        result.cap_lens = torch.tensor([1, 1], dtype=torch.int32)

        proc.process_batch_result_decode(_FakeBatch([bad, good]), result)

        # Only the healthy request's run is committed: one accepted draft.
        self.assertEqual(result.num_correct_drafts, 1)
        self.assertEqual(result.num_correct_drafts_per_req_cpu, [0, 1])
        self.assertEqual(result.num_block_accept_tokens, 2)
        self.assertEqual(result.num_cap_tokens, 1)
        adaptive = [o for o in observed if o[0] == "adaptive"]
        # The healthy sample goes to the slot of the batch size that ran it.
        self.assertEqual(adaptive, [("adaptive", ([1],), {"batch_size": 2})])
        (metrics,) = [o for o in observed if o[0] == "metrics"]
        # One emitting row: the dropped run is neither a bonus token nor a
        # verify observation.
        self.assertEqual(metrics[1][0], 1)
        self.assertEqual(metrics[1][1], 1)
        self.assertEqual(metrics[2]["num_block_accept_tokens"], 2)
        self.assertEqual(metrics[2]["num_cap_tokens"], 1)
        self.assertEqual(proc.metrics_reporter.num_generated_tokens, 1)
        (stats,) = [o for o in observed if o[0] == "decode_stats"]
        self.assertEqual(stats[2]["num_correct_drafts"], 1)
        self.assertEqual(stats[2]["num_emitting_rows"], 1)
        self.assertEqual(bad.spec_verify_ct, 0)
        self.assertEqual(good.spec_num_correct_drafts, 1)

    def test_realtime_decode_tokens_count_only_emitting_rows(self):
        reporter = object.__new__(SchedulerMetricsReporter)
        increments = []
        reporter.current_scheduler_metrics_enabled = True
        reporter.enable_mfu_metrics = False
        reporter.scheduler_status_logger = None
        reporter.metrics_collector = SimpleNamespace(
            increment_realtime_tokens=lambda **k: increments.append(k["decode_tokens"])
        )
        reporter.forward_ct_decode = 1
        reporter.decode_log_interval = 40
        batch = SimpleNamespace(batch_size=lambda: 2, dp_cooperation_info=None)

        reporter.report_decode_stats(
            True, running_batch=batch, num_correct_drafts=1, num_emitting_rows=1
        )
        reporter.report_decode_stats(True, running_batch=batch, num_correct_drafts=1)

        # One emitting row + one accepted draft; the default keeps batch size.
        self.assertEqual(increments, [2, 3])

    def test_overlap_in_flight_step_after_finish_emits_nothing(self):
        streamer = _Streamer()
        proc = _make_processor(enable_overlap=True, streamer=streamer)
        bad = _make_req("bad", eos_token_ids={EOS_TOKEN_ID})
        self._decode(proc, [bad], [[11, SENTINEL]], [True])
        self._assert_engine_fault(bad)
        committed = bad.kv_committed_len

        # The overlap scheduler already launched the next verify with this
        # request; its (again degenerate) result arrives after the finish.
        self._decode(proc, [bad], [[SENTINEL, SENTINEL, SENTINEL]], [True])

        self.assertEqual(list(bad.output_ids), [7])
        self.assertEqual(bad.kv_committed_len, committed)
        self._assert_engine_fault(bad)
        self.assertEqual(len(self.released), 1)


if __name__ == "__main__":
    unittest.main()
