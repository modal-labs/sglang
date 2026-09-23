from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_policy import (
    get_mamba_cache_miss_tokens,
    match_prefix_for_req,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAMBA_CHUNK = 64


class _RecordingMetric:
    def __init__(self):
        self.values = []

    def labels(self, **_labels):
        return self

    def inc(self, value):
        self.values.append(value)


def _match_result(*, branching_seqlen=None, full_kv_hit_length=0, device_len=0):
    return MatchResult(
        device_indices=torch.empty((device_len,), dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
        mamba_branching_seqlen=branching_seqlen,
        full_kv_hit_length=full_kv_hit_length,
    )


def test_mamba_cache_miss_tokens_uses_reusable_boundary():
    result = _match_result(
        branching_seqlen=8192,
        full_kv_hit_length=10000,
        device_len=2048,
    )

    assert get_mamba_cache_miss_tokens(result) == 6144


def test_mamba_cache_miss_tokens_ignores_non_mamba_misses():
    assert get_mamba_cache_miss_tokens(_match_result(full_kv_hit_length=10000)) == 0
    assert (
        get_mamba_cache_miss_tokens(
            _match_result(branching_seqlen=8192, full_kv_hit_length=0)
        )
        == 0
    )


def test_match_prefix_records_mamba_cache_miss_on_request():
    result = _match_result(
        branching_seqlen=8192,
        full_kv_hit_length=10000,
        device_len=2048,
    )

    class _TreeCache:
        def reprefill_tail_tokens(self):
            return 0

        def match_prefix(self, _params):
            return result

    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        kv=SimpleNamespace(cache_protected_len=0),
        _compute_max_prefix_len=lambda input_len: input_len,
    )

    match_prefix_for_req(_TreeCache(), req)

    assert req.mamba_cache_miss_tokens == 6144


def test_prefill_stats_reports_each_request_once():
    miss_req = SimpleNamespace(
        mamba_cache_miss_tokens=6144,
        _mamba_cache_miss_reported=False,
    )
    clean_req = SimpleNamespace(
        mamba_cache_miss_tokens=0,
        _mamba_cache_miss_reported=False,
    )
    adder = SimpleNamespace(
        can_run_list=[miss_req, clean_req],
        log_input_tokens=1,
        log_hit_tokens=2,
        reprocessed_log_input_tokens=0,
        reprocessed_log_hit_tokens=0,
        log_device_hit_tokens=2,
        log_host_hit_tokens=0,
        log_storage_hit_tokens=0,
        new_token_ratio=1.0,
    )

    first = PrefillStats.from_adder(adder, [])
    second = PrefillStats.from_adder(adder, [])

    assert (first.mamba_cache_miss_requests, first.mamba_cache_miss_tokens) == (1, 6144)
    assert (second.mamba_cache_miss_requests, second.mamba_cache_miss_tokens) == (0, 0)


def test_scheduler_metrics_collector_increments_mamba_cache_miss_counters():
    collector = object.__new__(SchedulerMetricsCollector)
    collector.labels = {"model_name": "test"}
    collector.mamba_cache_miss_requests_total = _RecordingMetric()
    collector.mamba_cache_miss_tokens_total = _RecordingMetric()

    collector.increment_mamba_cache_miss(num_requests=2, num_tokens=12288)

    assert collector.mamba_cache_miss_requests_total.values == [2]
    assert collector.mamba_cache_miss_tokens_total.values == [12288]


def test_init_next_round_input_records_mamba_cache_miss():
    """Admission matches through Req.init_next_round_input (FCFS on the unified
    cache never runs the policy-time match), so the miss must be recorded there."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    result = _match_result(
        branching_seqlen=8192, full_kv_hit_length=10000, device_len=2048
    )

    class _TreeCache:
        def reprefill_tail_tokens(self):
            return 0

        def supports_mamba(self):
            return True

        def match_prefix(self, _params):
            return result

    req = Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", range(16)),
        sampling_params=SamplingParams(),
    )
    req.init_next_round_input(_TreeCache())

    assert req.mamba_cache_miss_tokens == 6144
    assert req.mamba_branching_seqlen == 8192


class _NoopLRU:
    def reset_node_and_parents_mru(self, node, root):
        pass

    def reset_node_mru(self, node):
        pass


def _legacy_mamba_tree(cls, spans):
    """A cls tree holding one root-to-leaf chain, one node per
    (length, has_mamba_state, on_device) span; off-device nodes are host-only.
    Returns the tree and the chain's token ids."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = MAMBA_CHUNK
    set_global_server_args_for_scheduler(server_args)
    tree = cls.__new__(cls)
    tree.page_size = 1
    tree.disable = False
    tree.device = "cpu"
    tree.mamba_cache_chunk_size = MAMBA_CHUNK
    tree.full_lru_list = _NoopLRU()
    tree.mamba_lru_list = _NoopLRU()
    tree.root_node = TreeNode()
    tree.root_node.key = RadixKey(array("q"))
    tree.root_node.value = torch.empty((0,), dtype=torch.int64)
    node, start = tree.root_node, 1
    for length, has_mamba_state, on_device in spans:
        child = TreeNode()
        child.parent = node
        child.key = RadixKey(array("q", range(start, start + length)))
        kv = torch.arange(start, start + length, dtype=torch.int64)
        state = torch.tensor([0]) if has_mamba_state else None
        child.value, child.host_value = (kv, None) if on_device else (None, kv)
        child.mamba_value, child.mamba_host_value = (
            (state, None) if on_device else (None, state)
        )
        node.children[child.key.child_key(1)] = child
        node, start = child, start + length
    return tree, list(range(1, start))


def _match(tree, tokens):
    return tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))


def test_mamba_radix_cache_reports_full_kv_past_the_mamba_state():
    # 192 tokens of device KV, but only the first 64 have a mamba state.
    tree, tokens = _legacy_mamba_tree(
        MambaRadixCache, [(64, True, True), (128, False, True)]
    )

    result = _match(tree, tokens)

    assert len(result.device_indices) == 64
    assert result.full_kv_hit_length == 192
    assert get_mamba_cache_miss_tokens(result) == 128


def test_hi_mamba_radix_cache_counts_host_nodes_in_full_kv():
    # Device KV for 128 tokens (mamba state at 64), then 64 host-only tokens.
    # The branching point comes from the device part, so the miss stops at 128.
    tree, tokens = _legacy_mamba_tree(
        HiMambaRadixCache,
        [(64, True, True), (64, False, True), (64, False, False)],
    )

    result = _match(tree, tokens)

    assert len(result.device_indices) == 64
    assert result.full_kv_hit_length == 192
    assert get_mamba_cache_miss_tokens(result) == 64
