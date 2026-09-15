import unittest
from array import array

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingCache:
    dflash_draft_ring_reprefill_tail_tokens = 4_224
    reprefill_tail_tokens = BasePrefixCache.reprefill_tail_tokens

    def __init__(self):
        self.last_key = None

    def swa_reprefill_tail_tokens(self) -> int:
        return 0

    def supports_mamba(self) -> bool:
        return False

    def match_prefix(self, params):
        self.last_key = params.key
        node = object()
        return MatchResult(
            device_indices=torch.arange(len(params.key), dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )


class TestDFlashDraftRingPrefixCap(unittest.TestCase):
    def setUp(self):
        self.input_ids = array("q", range(5_000))
        self.req = Req(
            rid="ring-prefix-cap",
            origin_input_text="",
            origin_input_ids=self.input_ids,
            sampling_params=SamplingParams(),
        )
        self.cache = _RecordingCache()

    def test_schedule_policy_holds_back_ring_tail(self):
        match_prefix_for_req(self.cache, self.req)

        self.assertEqual(len(self.cache.last_key), 776)
        self.assertEqual(len(self.req.prefix_indices), 776)

    def test_canonical_req_rematch_preserves_ring_tail(self):
        self.req.init_next_round_input(self.cache)

        self.assertEqual(len(self.cache.last_key), 776)
        self.assertEqual(len(self.req.prefix_indices), 776)


if __name__ == "__main__":
    unittest.main()
