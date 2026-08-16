"""The cache_unfinished_req re-match must survive an in-flight mamba load-back.

Regression test for the production scheduler crash
``AssertionError: new_prefix_len=64, len(new_indices)=0``: the post-insert
re-match in ``cache_unfinished_req`` routes through
``MambaComponent.finalize_match_result``, whose in-flight-load-back guard
dropped the whole match whenever the terminal node had a pending mamba H2D
(commit->ack window). The guard protects CoW and load-back construction —
neither of which the re-match does — while insert had already counted the
full overlap and freed the request's own KV slots for it, so the dropped
re-match violated ``new_prefix_len <= len(new_indices)`` and killed the rank.

``MatchPrefixParams.repoint_only`` (set only by the re-match call site) now
bypasses the drop; every admission-context match keeps the protection.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.test.test_utils import CustomTestCase

_EMPTY = object()  # stands in for cache._empty_match_result


def _component_with_inflight_load(node_id=7):
    component = object.__new__(MambaComponent)
    component.component_type = "mamba"
    component.mamba_cache_chunk_size = 64
    component.cache = SimpleNamespace(
        ongoing_load_back={
            node_id: SimpleNamespace(pinned_mamba_slots=torch.tensor([3]))
        },
        _empty_match_result=_EMPTY,
        root_node=object(),
    )
    return component


def _matched_result(node):
    return MatchResult(
        device_indices=torch.arange(320),
        last_device_node=node,
        last_host_node=node,
        best_match_node=node,
        full_kv_hit_length=320,
    )


def _node(node_id=7, mamba_value=True):
    return SimpleNamespace(
        id=node_id,
        component_data={
            "mamba": SimpleNamespace(
                value=torch.tensor([5]) if mamba_value else None,
                host_value=None,
            )
        },
    )


class TestRepointMatchSurvivesInflightLoad(CustomTestCase):
    def test_admission_match_still_dropped(self):
        component = _component_with_inflight_load()
        node = _node()
        result = component.finalize_match_result(
            result=_matched_result(node),
            params=MatchPrefixParams(key=None),
            value_chunks=[],
            best_value_len=320,
        )
        self.assertIs(result, _EMPTY)

    def test_repoint_match_is_not_dropped(self):
        component = _component_with_inflight_load()
        node = _node()
        result = component.finalize_match_result(
            result=_matched_result(node),
            params=MatchPrefixParams(key=None, repoint_only=True),
            value_chunks=[],
            best_value_len=320,
        )
        self.assertIsNot(result, _EMPTY)
        # The invariant cache_unfinished_req asserts on: the full device match
        # survives, so new_prefix_len (<= key length) <= len(device_indices).
        self.assertEqual(len(result.device_indices), 320)

    def test_repoint_match_without_inflight_load_unchanged(self):
        component = _component_with_inflight_load(node_id=99)  # different node
        node = _node(node_id=7)
        result = component.finalize_match_result(
            result=_matched_result(node),
            params=MatchPrefixParams(key=None, repoint_only=True),
            value_chunks=[],
            best_value_len=320,
        )
        self.assertEqual(len(result.device_indices), 320)


if __name__ == "__main__":
    unittest.main()
