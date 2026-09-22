"""A DFlash verify target row without positive finite mass (all-NaN, Inf, or
zero) must not reach the reject sampler as-is; the request is marked sticky
(one warning) and a marked request finishes with an engine-fault FINISH_ABORT
whatever its EOS set or ignore_eos. An unmarked request that samples the
reject-sampler sentinel ``vocab_size - 1`` still follows the normal EOS rules."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from http import HTTPStatus
from unittest import mock

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    reject_sampling_sentinel_token,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_MATCHED_TOKEN, Req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import (
    greedy_target_predict,
    greedy_target_row_degenerate,
    sanitize_dflash_verify_target_probs,
    sanitize_greedy_target_logits,
)
from sglang.test.test_utils import CustomTestCase

VOCAB = 64
SENTINEL = reject_sampling_sentinel_token(VOCAB)
EOS_TOKEN_ID = 2


class _FakeTokenizer:
    eos_token_id = EOS_TOKEN_ID
    additional_stop_token_ids = None


def _make_req(output_ids, eos_token_ids=frozenset({EOS_TOKEN_ID, SENTINEL})):
    sampling_params = SamplingParams(max_new_tokens=1_000)
    sampling_params.normalize(tokenizer=_FakeTokenizer())
    req = Req(
        rid="target-degenerate",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sampling_params,
        eos_token_ids=set(eos_token_ids),
        vocab_size=VOCAB,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


class TestSanitizeVerifyTargetProbs(CustomTestCase):
    def _probs(self, bs=3, rows=4):
        logits = torch.randn(bs, rows, VOCAB)
        return torch.softmax(logits, dim=-1)

    def test_valid_rows_are_untouched(self):
        probs = self._probs()
        expected = probs.clone()
        out, degenerate = sanitize_dflash_verify_target_probs(probs)
        self.assertIs(out, probs)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertFalse(degenerate.any().item())
        self.assertEqual(tuple(degenerate.shape), (3,))
        self.assertEqual(degenerate.dtype, torch.bool)

    def test_nan_inf_zero_rows_collapse_onto_sentinel(self):
        probs = self._probs()
        expected = probs.clone()
        probs[0, 1] = float("nan")
        probs[2, 0, 5] = float("inf")
        probs[2, 3] = 0.0
        out, degenerate = sanitize_dflash_verify_target_probs(probs)

        self.assertEqual(degenerate.tolist(), [True, False, True])
        one_hot = torch.zeros(VOCAB)
        one_hot[SENTINEL] = 1.0
        for b, r in ((0, 1), (2, 0), (2, 3)):
            torch.testing.assert_close(out[b, r], one_hot, rtol=0, atol=0)
            expected[b, r] = one_hot
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_partial_nan_row_is_degenerate(self):
        probs = self._probs(bs=1, rows=1)
        probs[0, 0, 7] = float("nan")
        _, degenerate = sanitize_dflash_verify_target_probs(probs)
        self.assertEqual(degenerate.tolist(), [True])


class TestGreedyTargetRowDegenerate(CustomTestCase):
    def test_neg_inf_masks_over_finite_rows_are_valid(self):
        logits = torch.randn(3, VOCAB)
        logits[0, 1:] = float("-inf")
        logits[1, ::2] = float("-inf")
        self.assertEqual(
            greedy_target_row_degenerate(logits).tolist(), [False, False, False]
        )

    def test_nan_pos_inf_or_no_finite_entry_is_degenerate(self):
        logits = torch.randn(4, VOCAB)
        logits[0, 3] = float("nan")
        logits[1, 5] = float("inf")
        logits[2] = float("-inf")
        self.assertEqual(
            greedy_target_row_degenerate(logits).tolist(), [True, True, True, False]
        )

    def test_greedy_predict_collapses_degenerate_rows_onto_sentinel(self):
        logits = torch.randn(4, VOCAB)
        logits[0] = float("nan")
        logits[1, 5] = float("inf")
        logits[2, 1:] = float("-inf")
        expected = torch.argmax(logits, dim=-1)
        predict, degenerate = greedy_target_predict(logits)
        self.assertEqual(degenerate.tolist(), [True, True, False, False])
        self.assertEqual(predict[:2].tolist(), [SENTINEL, SENTINEL])
        self.assertEqual(predict[2:].tolist(), expected[2:].tolist())
        self.assertEqual(predict.dtype, expected.dtype)

    def test_sanitized_greedy_logits_argmax_to_sentinel(self):
        logits = torch.randn(3, VOCAB)
        logits[1, 3] = float("nan")
        expected = logits.clone()
        out, degenerate = sanitize_greedy_target_logits(logits)
        self.assertEqual(degenerate.tolist(), [False, True, False])
        torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(out[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
        self.assertEqual(int(torch.argmax(out[1])), SENTINEL)
        self.assertEqual(float(out[1, SENTINEL]), 0.0)
        self.assertTrue(torch.isneginf(out[1, :SENTINEL]).all().item())


class TestDegenerateReqFinishState(CustomTestCase):
    def _assert_engine_fault(self, req):
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(
            req.finished_reason.status_code, HTTPStatus.INTERNAL_SERVER_ERROR
        )
        self.assertTrue(req.spec_target_degenerate)

    def test_marked_req_aborts_with_sentinel_outside_eos_set(self):
        # The served recipe has no sentinel EOS override.
        req = _make_req([11, 13], eos_token_ids={EOS_TOKEN_ID})
        req.mark_spec_target_degenerate()

        req.update_finish_state(new_accepted_len=0)

        self._assert_engine_fault(req)
        self.assertEqual(list(req.output_ids_through_stop), [11, 13])

    def test_marked_req_aborts_under_ignore_eos(self):
        req = _make_req([11, 13])
        req.sampling_params.ignore_eos = True
        req.mark_spec_target_degenerate()

        req.update_finish_state(new_accepted_len=0)

        self._assert_engine_fault(req)

    def test_marked_req_abort_beats_same_step_eos(self):
        req = _make_req([11, EOS_TOKEN_ID])
        req.mark_spec_target_degenerate()

        req.update_finish_state(new_accepted_len=2)

        self._assert_engine_fault(req)

    def test_marked_req_abort_beats_length(self):
        req = _make_req([11, 13])
        req.sampling_params.max_new_tokens = 2
        req.mark_spec_target_degenerate()

        req.update_finish_state(new_accepted_len=0)

        self._assert_engine_fault(req)

    def test_pending_to_finish_is_kept(self):
        req = _make_req([11])
        client_abort = FINISH_ABORT("client abort")
        req.to_finish = client_abort
        req.mark_spec_target_degenerate()

        req.update_finish_state(new_accepted_len=0)

        self.assertIs(req.finished_reason, client_abort)

    def test_unmarked_req_sampling_sentinel_also_stops(self):
        req = _make_req([11, 13, SENTINEL, 21])

        req.update_finish_state(new_accepted_len=4)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, SENTINEL)
        self.assertEqual(req.finished_len, 3)
        self.assertFalse(req.spec_target_degenerate)

    def test_unmarked_sentinel_outside_eos_set_is_a_normal_token(self):
        req = _make_req([11, 13, SENTINEL, 21], eos_token_ids={EOS_TOKEN_ID})

        req.update_finish_state(new_accepted_len=4)

        self.assertFalse(req.finished())
        self.assertFalse(req.spec_target_degenerate)

    def test_mark_is_sticky_and_logs_once(self):
        req = _make_req([11])
        with self.assertLogs(
            "sglang.srt.managers.schedule_batch", level="WARNING"
        ) as cm:
            req.mark_spec_target_degenerate()
            req.mark_spec_target_degenerate()
        self.assertEqual(len(cm.output), 1)
        self.assertTrue(req.spec_target_degenerate)


class TestBatchResultCarriesTargetDegenerate(CustomTestCase):
    def test_copy_to_cpu_moves_mask(self):
        result = GenerationBatchResult(
            next_token_ids=torch.zeros(4, dtype=torch.int64),
            accept_lens=torch.ones(1, dtype=torch.int32),
            target_degenerate=torch.tensor([True]),
            copy_done=mock.Mock(),
        )
        result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
        self.assertTrue(result.target_degenerate.is_cpu)
        self.assertEqual(result.target_degenerate.tolist(), [True])


if __name__ == "__main__":
    unittest.main()
