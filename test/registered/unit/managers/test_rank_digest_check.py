"""CPU tests for the per-rank extend-batch digest check
(SGLANG_K3_RANK_DIGEST_CHECK, managers/scheduler_components/rank_digest_check.py).

The check exists because a TP rank whose HiCache V3 / radix state drifted
computes different extend_prefix_lens, hence a different chunked-prefix plan
and a different number of DCP KV all-gathers, and the gang then wedges in
NCCL instead of faulting. These tests drive RankDigestChecker with a fake
all-gather that injects the peers' encoded digests.
"""

import unittest

import torch

from sglang.srt.managers.scheduler_components.rank_digest_check import (
    AttnDecisionConfig,
    DigestAttnMethod,
    RankDigestChecker,
    build_digest,
    chunked_prefix_plan,
    compare_digests,
    decide_extend_attn_forward_method,
    encode_digest,
    encoded_length,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAX_BATCH = 8
EXTEND = 1  # any stable int: the digest compares the value, not the enum

CFG = AttnDecisionConfig(
    prefill_backend="trtllm_mla",
    disable_chunked_prefix_cache=False,
    chunked_prefix_cache_threshold=8192,
    flashinfer_mla_disable_ragged=False,
    enable_deterministic_inference=False,
    mla_prefill_cp=False,
    chunk_capacity=1024,
)


def _batch(**overrides):
    kwargs = dict(
        forward_mode=EXTEND,
        is_extend_without_speculative=True,
        rids=["req-a", "req-b", "req-c"],
        extend_lens=[100, 200, 300],
        extend_prefix_lens=[256, 0, 1024],
        seq_lens=[356, 200, 1324],
    )
    kwargs.update(overrides)
    return kwargs


class _FakeAllGather:
    """Returns this rank's tensor in slot 0 and the peers' pre-encoded digests
    in the remaining slots; records what was sent so the test can assert the
    collective shape is batch-independent."""

    def __init__(self, peers):
        self.peers = peers
        self.sent = []

    def __call__(self, tensor):
        self.sent.append(tensor.clone())
        for peer in self.peers:
            assert peer.shape == tensor.shape, "collective shape must be fixed"
        return [tensor] + list(self.peers)


def _check(checker, **overrides):
    """Drive check() the way run_batch does: seq_lens arrive as the
    ScheduleBatch's CPU tensor."""
    kwargs = _batch(**overrides)
    kwargs["seq_lens_cpu"] = torch.tensor(kwargs.pop("seq_lens"), dtype=torch.int64)
    return checker.check(**kwargs)


def _peer_tensor(forward_ct, **overrides):
    digest = build_digest(forward_ct=forward_ct, cfg=CFG, **_batch(**overrides))
    return encode_digest(digest, max_batch=MAX_BATCH)


class TestChunkedPrefixPlan(CustomTestCase):
    def test_matches_forward_batch_mixin_derivation(self):
        """Derived property: the pure-Python plan must equal what
        ForwardBatchDeepSeekMHAMixin.prepare_chunked_prefix_cache_info computes
        on the GPU path (its docstring example: prefix 1024, capacity 1024,
        bs 4 -> chunk len 256, 4 chunks, lower-triangular lens). The plan sizes
        the DCP all-gathers, so a drift here would make the digest agree
        across ranks while the forward does not."""
        chunk_len, n_chunks, rows = chunked_prefix_plan(
            extend_prefix_lens=[256, 512, 768, 1024], chunk_capacity=1024
        )
        self.assertEqual((chunk_len, n_chunks), (256, 4))
        self.assertEqual(
            rows,
            (
                (256, 256, 256, 256),
                (0, 256, 256, 256),
                (0, 0, 256, 256),
                (0, 0, 0, 256),
            ),
        )
        self.assertEqual(
            chunked_prefix_plan(extend_prefix_lens=[0, 0], chunk_capacity=1024),
            (0, 0, ()),
        )


class TestAttnMethodMirror(CustomTestCase):
    def test_generic_backend_prefix_threshold_and_one_shot(self):
        """Derived property of attention_backend_handler._handle_attention_backend:
        below the chunked-prefix threshold a prefixed batch is absorbed MLA;
        at/above it, MHA one-shot when the whole batch fits one chunk, else
        chunked KV; trtllm_mla ignores the threshold."""
        fa3 = AttnDecisionConfig(
            prefill_backend="fa3",
            disable_chunked_prefix_cache=False,
            chunked_prefix_cache_threshold=8192,
            flashinfer_mla_disable_ragged=False,
            enable_deterministic_inference=False,
            mla_prefill_cp=False,
            chunk_capacity=16384,
        )

        def decide(cfg, prefix, seq):
            return decide_extend_attn_forward_method(
                cfg=cfg,
                is_extend_without_speculative=True,
                sum_extend_prefix_lens=prefix,
                sum_seq_lens=seq,
            )

        self.assertEqual(decide(fa3, 100, 1000), DigestAttnMethod.MLA)
        self.assertEqual(decide(fa3, 8192, 9000), DigestAttnMethod.MHA_ONE_SHOT)
        self.assertEqual(decide(fa3, 8192, 20000), DigestAttnMethod.MHA_CHUNKED_KV)
        self.assertEqual(decide(CFG, 100, 1000), DigestAttnMethod.MHA_CHUNKED_KV)


class TestRankDigestChecker(CustomTestCase):
    def _checker(self, peers):
        gather = _FakeAllGather(peers)
        checker = RankDigestChecker(
            rank=0,
            world_size=1 + len(peers),
            max_batch=MAX_BATCH,
            cfg=CFG,
            all_gather=gather,
        )
        return checker, gather

    def test_identical_ranks_pass_and_count(self):
        """Negative-branch contract: agreeing ranks never trip the fail-stop,
        and the per-rank forward counter advances so a later skipped forward
        on one rank shows up as a counter mismatch."""
        checker, gather = self._checker([_peer_tensor(1), _peer_tensor(1)])
        digest = _check(checker)
        self.assertEqual(digest.forward_ct, 1)
        self.assertEqual(digest.attn_forward_method, DigestAttnMethod.MHA_CHUNKED_KV)
        self.assertEqual(digest.num_prefix_chunks, 4)  # ceil(1024 / (1024 // 3))
        self.assertEqual(gather.sent[0].shape, (encoded_length(MAX_BATCH),))

    def test_prefix_len_divergence_fail_stops_with_per_rank_values(self):
        """Bug the check is for: one rank sees a different cached-prefix
        length for the same request. The gather must still complete (fixed
        shape), the ERROR line must name the differing fields with every
        rank's values, and the process must fail-stop via RuntimeError."""
        bad_peer = _peer_tensor(
            1, extend_prefix_lens=[256, 0, 512], seq_lens=[356, 200, 812]
        )
        checker, _ = self._checker([_peer_tensor(1), bad_peer])
        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.rank_digest_check", level="ERROR"
        ) as logs:
            with self.assertRaises(RuntimeError) as ctx:
                _check(checker)
        line = "\n".join(logs.output)
        self.assertIn(
            "extend_prefix_lens: r0=[256,0,1024] r1=[256,0,1024] r2=[256,0,512]", line
        )
        self.assertIn("num_prefix_chunks: r0=4 r1=4 r2=2", line)
        self.assertIn("prefix_chunk_seq_lens_hash", line)
        self.assertNotIn("rid_hashes", line)
        self.assertIn("extend_prefix_lens", str(ctx.exception))
        self.assertIn("fail-stop", str(ctx.exception))

    def test_batch_membership_divergence_is_reported_not_crashed(self):
        """A peer running a different batch (one request fewer, different
        rids) must decode cleanly from the fixed-size encoding and be
        reported as batch_size + rid + lens mismatches."""
        peer = _peer_tensor(
            1,
            rids=["req-a", "req-b"],
            extend_lens=[100, 200],
            extend_prefix_lens=[256, 0],
            seq_lens=[356, 200],
        )
        checker, _ = self._checker([peer])
        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.rank_digest_check", level="ERROR"
        ) as logs:
            with self.assertRaises(RuntimeError):
                _check(checker)
        line = "\n".join(logs.output)
        self.assertIn("batch_size: r0=3 r1=2", line)
        self.assertIn("rid_hashes:", line)
        self.assertIn("local=rids=['req-a', 'req-b', 'req-c']", line)

    def test_forward_counter_divergence(self):
        """A rank that skipped (or doubled) an extend forward differs only in
        the counter; that must be enough to fail-stop."""
        checker, _ = self._checker([_peer_tensor(2)])
        with self.assertRaises(RuntimeError) as ctx:
            _check(checker)
        self.assertIn("forward_ct", str(ctx.exception))

    def test_gather_timeout_fail_stops_with_local_digest(self):
        """A peer that never reaches the gather (gloo wait timeout) must
        surface as a fail-stop that carries the local digest, not as a bare
        collective error."""

        def timed_out(_tensor):
            raise RuntimeError("Operation timed out!")

        checker = RankDigestChecker(
            rank=3, world_size=8, max_batch=MAX_BATCH, cfg=CFG, all_gather=timed_out
        )
        with self.assertRaises(RuntimeError) as ctx:
            _check(checker)
        msg = str(ctx.exception)
        self.assertIn("rank 3", msg)
        self.assertIn("did not complete", msg)
        self.assertIn("rids=['req-a', 'req-b', 'req-c']", msg)


class TestEncoding(CustomTestCase):
    def test_hash_covers_fields_the_vectors_truncate(self):
        """Critical-path bookkeeping: the digest hash must still catch a
        divergence in request slots beyond max_batch (vectors are truncated to
        the fixed length) so an oversized batch cannot silently pass."""
        small = 2
        a = build_digest(forward_ct=1, cfg=CFG, **_batch())
        b = build_digest(
            forward_ct=1,
            cfg=CFG,
            **_batch(rids=["req-a", "req-b", "req-zzz"]),
        )
        ta = encode_digest(a, max_batch=small)
        tb = encode_digest(b, max_batch=small)
        self.assertTrue(torch.equal(ta[1:], tb[1:]))  # only the hash differs
        mismatches = compare_digests([ta, tb], max_batch=small)
        self.assertEqual([name for name, _ in mismatches], ["digest_hash"])


if __name__ == "__main__":
    unittest.main()
