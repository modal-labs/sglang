from types import SimpleNamespace

import pytest

import importlib.util
from pathlib import Path

_source = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/short_prefill.py"
)
_spec = importlib.util.spec_from_file_location("short_prefill_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
add_chunk_with_short_prefill_budget = _module.add_chunk_with_short_prefill_budget


class Adder:
    def __init__(self, budget=16384, finish_after=None, fail=False):
        self.rem_chunk_tokens = budget
        self.page_size = 256
        self.dllm_config = None
        self.finish_after = finish_after
        self.fail = fail
        self.offered = None

    def add_chunked_req(self, req):
        self.offered = self.rem_chunk_tokens
        if self.fail:
            raise RuntimeError("allocation failed")
        used = min(self.offered, self.finish_after or self.offered)
        self.rem_chunk_tokens -= used
        return None if self.finish_after and used == self.finish_after else req


def waiter(n):
    return SimpleNamespace(origin_input_ids=range(n), num_matched_prefix_tokens=0)


def run(adder, queue):
    return add_chunk_with_short_prefill_budget(
        adder,
        object(),
        queue,
        threshold=8192,
        chunk_size=8192,
        batch_size=16384,
        scan_waiting=True,
    )


@pytest.mark.parametrize(
    "sizes,offered,residual",
    [
        ([], 16384, 0),
        ([2048], 14336, 2048),
        ([2048, 4096], 10240, 6144),
        ([8192], 8192, 8192),
        ([9000], 16384, 0),
        ([1024, 512], 14848, 1536),
        ([2049], 14080, 2304),
    ],
)
def test_reserve_only_fitting_whole_work(sizes, offered, residual):
    adder = Adder()
    run(adder, [waiter(n) for n in sizes])
    assert adder.offered == offered
    assert adder.rem_chunk_tokens == residual
    assert adder.offered + adder.rem_chunk_tokens == 16384


def test_fit_scan_preserves_relative_order_and_all_requests():
    queue = [waiter(n) for n in [9000, 6144, 4096, 2048]]
    original = list(queue)
    adder = Adder()
    run(adder, queue)
    assert queue == [original[1], original[3], original[0], original[2]]
    assert adder.offered == 8192


def test_continuation_finishes_early_returns_space():
    adder = Adder(finish_after=1024)
    assert run(adder, [waiter(2048)]) is None
    assert adder.rem_chunk_tokens == 15360


def test_exception_restores_budget_and_queue():
    adder = Adder(fail=True)
    queue = [waiter(9000), waiter(2048)]
    before = list(queue)
    with pytest.raises(RuntimeError):
        run(adder, queue)
    assert adder.rem_chunk_tokens == 16384
    assert queue == before


@pytest.mark.parametrize("budget", [4096, 8192, 12288])
def test_respects_existing_smaller_budget(budget):
    adder = Adder(budget)
    run(adder, [waiter(8192)])
    assert adder.offered == budget
