import socket
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.speculative.dflash_worker_v2 import (
    _is_all_greedy,
    _sync_dflash_selector_draft,
)
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTpSync:
    def __init__(self, enabled_sites, canonical=None):
        self._sites = frozenset(enabled_sites)
        self.calls = []
        self.canonical = canonical

    def enabled(self, site):
        return site in self._sites

    def sync(self, site, values):
        self.calls.append((site, values))
        if self.canonical is not None:
            values.copy_(self.canonical)
        return values


def _pack_selector_draft(draft_next, candidate_ids, q_rows):
    bs, gamma, top_k = candidate_ids.shape
    packed = torch.empty(
        (bs, gamma, 1 + 2 * top_k), dtype=torch.int32, device=draft_next.device
    )
    packed[:, :, 0].copy_(draft_next)
    packed[:, :, 1 : 1 + top_k].copy_(candidate_ids)
    packed[:, :, 1 + top_k :].copy_(q_rows.view(torch.int32))
    return packed


def test_selector_draft_sync_broadcasts_one_packed_buffer():
    bs, gamma, top_k = 2, 7, 4
    draft_next = torch.zeros((bs, gamma), dtype=torch.int64)
    candidate_ids = torch.zeros((bs, gamma, top_k), dtype=torch.int64)
    q_rows = torch.zeros((bs, gamma, top_k), dtype=torch.float32)
    canonical_draft = torch.arange(bs * gamma, dtype=torch.int64).view(bs, gamma)
    canonical_candidates = torch.arange(bs * gamma * top_k, dtype=torch.int64).view(
        bs, gamma, top_k
    )
    canonical_q = torch.tensor(
        [-1.25, 1.40129846e-45, 0.5, -0.0] * (bs * gamma),
        dtype=torch.float32,
    ).view(bs, gamma, top_k)
    canonical = _pack_selector_draft(canonical_draft, canonical_candidates, canonical_q)
    pack_buffer = torch.empty_like(canonical)
    tp_sync = _FakeTpSync({SpecTpSyncSite.DFLASH_DRAFT_SAMPLE}, canonical=canonical)

    _sync_dflash_selector_draft(
        draft_next,
        candidate_ids,
        q_rows,
        tp_sync=tp_sync,
        pack_buffer=pack_buffer,
    )

    assert len(tp_sync.calls) == 1
    site, values = tp_sync.calls[0]
    assert site == SpecTpSyncSite.DFLASH_DRAFT_SAMPLE
    assert values.data_ptr() == pack_buffer.data_ptr()
    assert values.shape == pack_buffer.shape
    assert torch.equal(draft_next, canonical_draft)
    assert torch.equal(candidate_ids, canonical_candidates)
    assert torch.equal(q_rows, canonical_q)


def test_selector_draft_sync_disabled_is_noop():
    draft_next = torch.arange(14, dtype=torch.int64).view(2, 7)
    candidate_ids = torch.arange(56, dtype=torch.int64).view(2, 7, 4)
    q_rows = torch.arange(56, dtype=torch.float32).view(2, 7, 4)
    original = (draft_next.clone(), candidate_ids.clone(), q_rows.clone())
    pack_buffer = torch.empty((2, 7, 9), dtype=torch.int32)
    tp_sync = _FakeTpSync(frozenset())

    _sync_dflash_selector_draft(
        draft_next,
        candidate_ids,
        q_rows,
        tp_sync=tp_sync,
        pack_buffer=pack_buffer,
    )

    assert tp_sync.calls == []
    assert torch.equal(draft_next, original[0])
    assert torch.equal(candidate_ids, original[1])
    assert torch.equal(q_rows, original[2])


def test_selector_draft_sync_rejects_non_float_q_rows():
    candidate_ids = torch.zeros((2, 7, 4), dtype=torch.int64)
    draft_next = torch.zeros((2, 7), dtype=torch.int64)
    q_rows = torch.zeros((2, 7, 4), dtype=torch.float64)
    pack_buffer = torch.empty((2, 7, 9), dtype=torch.int32)
    tp_sync = _FakeTpSync({SpecTpSyncSite.DFLASH_DRAFT_SAMPLE})

    with pytest.raises(ValueError, match="q rows must be float32"):
        _sync_dflash_selector_draft(
            draft_next,
            candidate_ids,
            q_rows,
            tp_sync=tp_sync,
            pack_buffer=pack_buffer,
        )


@pytest.mark.parametrize(
    "pack_buffer",
    [
        torch.empty((2, 7, 9), dtype=torch.int64),
        torch.empty((2, 7, 8), dtype=torch.int32),
    ],
)
def test_selector_draft_sync_rejects_invalid_pack_buffer(pack_buffer):
    draft_next = torch.zeros((2, 7), dtype=torch.int64)
    candidate_ids = torch.zeros((2, 7, 4), dtype=torch.int64)
    q_rows = torch.zeros((2, 7, 4), dtype=torch.float32)
    tp_sync = _FakeTpSync({SpecTpSyncSite.DFLASH_DRAFT_SAMPLE})

    with pytest.raises(ValueError, match="pack buffer"):
        _sync_dflash_selector_draft(
            draft_next,
            candidate_ids,
            q_rows,
            tp_sync=tp_sync,
            pack_buffer=pack_buffer,
        )


def test_draft_sample_site_is_rng_and_in_all():
    assert SpecTpSyncSite.DFLASH_DRAFT_SAMPLE in parse_spec_tp_sync("all")
    assert SpecTpSyncSite.DFLASH_DRAFT_SAMPLE in parse_spec_tp_sync("rng")
    assert SpecTpSyncSite.DFLASH_DRAFT_SAMPLE not in parse_spec_tp_sync("init")
    assert parse_spec_tp_sync("dflash-draft-sample") == {
        SpecTpSyncSite.DFLASH_DRAFT_SAMPLE
    }


def test_selector_draft_sync_respects_greedy_gate():
    assert _is_all_greedy(None) is True
    assert _is_all_greedy(types.SimpleNamespace(is_all_greedy=False)) is False


class _GlooGroup:
    def __init__(self, rank):
        self.world_size = 2
        self.rank_in_group = rank

    def broadcast(self, tensor, src):
        dist.broadcast(tensor, src=src)


def _run_selector_draft_gloo(rank, port):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        bs, gamma, top_k = 1, 2, 2
        if rank == 0:
            draft_next = torch.tensor([[11, 12]], dtype=torch.int64)
            candidate_ids = torch.tensor([[[21, 22], [23, 24]]], dtype=torch.int64)
            q_rows = torch.tensor(
                [[[-1.25, 1.40129846e-45], [0.5, -0.0]]],
                dtype=torch.float32,
            )
        else:
            draft_next = torch.tensor([[31, 32]], dtype=torch.int64)
            candidate_ids = torch.tensor([[[41, 42], [43, 44]]], dtype=torch.int64)
            q_rows = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], dtype=torch.float32)
        pack_buffer = torch.empty((bs, gamma, 1 + 2 * top_k), dtype=torch.int32)
        tp_sync = SpecTpSync(_GlooGroup(rank))

        _sync_dflash_selector_draft(
            draft_next,
            candidate_ids,
            q_rows,
            tp_sync=tp_sync,
            pack_buffer=pack_buffer,
        )

        assert torch.equal(draft_next, torch.tensor([[11, 12]], dtype=torch.int64))
        assert torch.equal(
            candidate_ids, torch.tensor([[[21, 22], [23, 24]]], dtype=torch.int64)
        )
        assert torch.equal(
            q_rows,
            torch.tensor(
                [[[-1.25, 1.40129846e-45], [0.5, -0.0]]],
                dtype=torch.float32,
            ),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed gloo is unavailable",
)
def test_selector_draft_sync_real_gloo_two_processes():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_run_selector_draft_gloo, args=(port,), nprocs=2, join=True)
