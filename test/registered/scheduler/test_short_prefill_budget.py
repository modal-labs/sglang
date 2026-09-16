from types import SimpleNamespace

import pytest

import importlib.util
from pathlib import Path

path = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/managers/short_prefill.py"
)
spec = importlib.util.spec_from_file_location("short_prefill", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CHUNK, TOTAL, SMALL = 4096, 6144, 2048


def add(adder, chunk, queue, threshold=SMALL):
    return module.add_chunk_with_short_prefill_budget(
        adder, chunk, queue, threshold=threshold, chunk_size=CHUNK, batch_size=TOTAL
    )


class Adder:
    dllm_config = None

    def __init__(self, budget=16384, remaining=100000, fail=False):
        self.rem_chunk_tokens = budget
        self.remaining = remaining
        self.fail = fail
        self.calls = 0

    def add_chunked_req(self, req):
        self.calls += 1
        if self.fail:
            raise RuntimeError("allocation failed")
        if self.rem_chunk_tokens is None:
            return req
        consumed = min(self.remaining, self.rem_chunk_tokens)
        self.rem_chunk_tokens -= consumed
        self.remaining -= consumed
        return req if self.remaining else None


def head(prompt, cached=0):
    return SimpleNamespace(
        origin_input_ids=range(prompt), num_matched_prefix_tokens=cached
    )


def test_short_waiter_caps_the_continuing_chunk_and_the_step():
    a = Adder()
    chunk = object()
    continuation = add(a, chunk, [head(500)])
    assert continuation is chunk and a.calls == 1
    assert 100000 - a.remaining == CHUNK  # the chunk took exactly the cap
    assert a.rem_chunk_tokens == TOTAL - CHUNK  # what waiters may still take


@pytest.mark.parametrize("waiting", [[], [head(10000)], [head(500, 500)]])
def test_no_eligible_waiter_preserves_full_chunk(waiting):
    a = Adder()
    add(a, object(), waiting)
    assert a.rem_chunk_tokens == 0 and a.remaining == 83616


def test_finishing_chunk_leaves_capped_step_budget():
    a = Adder(remaining=1000)
    continuation = add(a, object(), [head(500)])
    assert continuation is None
    assert a.rem_chunk_tokens == min(16384 - 1000, TOTAL - 1000)


def test_small_remaining_budget_is_not_capped_further():
    a = Adder(budget=CHUNK)  # already at or below the cap: untouched
    add(a, object(), [head(500)])
    assert a.rem_chunk_tokens == 0


def test_exception_restores_original_budget():
    a = Adder(fail=True)
    with pytest.raises(RuntimeError):
        add(a, object(), [head(500)])
    assert a.rem_chunk_tokens == 16384


def test_disabled_chunking_is_unchanged():
    a = Adder(budget=None)
    req = object()
    assert add(a, req, [head(500)]) is req


def test_cached_long_prompt_can_qualify():
    a = Adder()
    add(a, object(), [head(100000, 99500)])
    assert a.remaining == 100000 - CHUNK


def test_threshold_is_inclusive():
    a = Adder()
    add(a, object(), [head(SMALL)])
    assert a.remaining == 100000 - CHUNK
    b = Adder()
    add(b, object(), [head(SMALL + 1)])
    assert b.remaining == 83616


def test_disabled_protection_keeps_original_budget():
    a = Adder()
    add(a, object(), [head(500)], threshold=0)
    assert a.remaining == 83616
    assert a.rem_chunk_tokens == 0


def test_only_queue_head_controls_budget():
    a = Adder()
    add(a, object(), [head(10000), head(500)])
    assert a.remaining == 83616
    assert a.rem_chunk_tokens == 0


def test_scan_promotes_short_waiters_beyond_ineligible_head():
    a = Adder()
    long, short, medium, short2 = head(10000), head(500), head(3000), head(1000)
    queue = [long, short, medium, short2]
    module.add_chunk_with_short_prefill_budget(
        a,
        object(),
        queue,
        threshold=SMALL,
        chunk_size=CHUNK,
        batch_size=TOTAL,
        scan_waiting=True,
    )
    assert queue == [short, short2, long, medium]
    assert a.remaining == 100000 - CHUNK
    assert a.rem_chunk_tokens == SMALL


def test_scan_keeps_long_request_progress_until_completion():
    a = Adder(remaining=CHUNK * 3)
    continuing = object()
    for index in range(3):
        continuing = module.add_chunk_with_short_prefill_budget(
            a,
            continuing,
            [head(10000), head(500)],
            threshold=SMALL,
            chunk_size=CHUNK,
            batch_size=TOTAL,
            scan_waiting=True,
        )
        assert a.remaining == CHUNK * (2 - index)
        a.rem_chunk_tokens = 16384
    assert continuing is None


def test_scan_does_not_mutate_queue_on_allocation_failure():
    a = Adder(fail=True)
    queue = [head(10000), head(500)]
    original = list(queue)
    with pytest.raises(RuntimeError):
        module.add_chunk_with_short_prefill_budget(
            a,
            object(),
            queue,
            threshold=SMALL,
            chunk_size=CHUNK,
            batch_size=TOTAL,
            scan_waiting=True,
        )
    assert queue == original
    assert a.rem_chunk_tokens == 16384


def test_scan_without_eligible_requests_preserves_queue_and_full_chunk():
    a = Adder()
    queue = [head(10000), head(3000)]
    original = list(queue)
    module.add_chunk_with_short_prefill_budget(
        a,
        object(),
        queue,
        threshold=SMALL,
        chunk_size=CHUNK,
        batch_size=TOTAL,
        scan_waiting=True,
    )
    assert queue == original
    assert a.remaining == 83616
