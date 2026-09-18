"""CPU tests for rank-consistent multimodal encode decisions.

The mm embedding cache is per scheduler process, so attention-TP ranks can
disagree on which items need the ViT; the DP-sharded encoder ends in a
collective, so a hit-on-one-rank / miss-on-another split deadlocks the
group. These tests drive `_batch_encode_per_image_misses` with a fake group
whose all_reduce injects the peers' miss flags.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_utils
from sglang.srt.managers.mm_utils import PerImageRequestInfo
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeAttnTpGroup:
    """all_reduce(SUM) over this rank's flags plus a preset peer vector."""

    def __init__(self, peer_flags):
        self.peer_flags = peer_flags
        self.calls = []

    def all_reduce(self, input_):
        self.calls.append(input_.clone())
        peer = torch.tensor(self.peer_flags, dtype=input_.dtype, device=input_.device)
        assert peer.shape == input_.shape
        return input_ + peer


def _item(hash_value, tokens):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_value,
        offsets=[(0, tokens - 1)],
        feature=torch.zeros(tokens, 2),
    )
    return item


@pytest.fixture(autouse=True)
def _fresh_cache():
    mm_utils.init_mm_embedding_cache(1 << 20)
    yield
    mm_utils.init_mm_embedding_cache(1 << 20)


def _requests(items):
    offsets = []
    cursor = 0
    for item in items:
        n = item.offsets[0][1] - item.offsets[0][0] + 1
        offsets.append((cursor, cursor + n - 1))
        cursor += n
    return [
        PerImageRequestInfo(
            req_idx=0,
            items=list(items),
            items_offset=offsets,
            extend_prefix_len=0,
            extend_seq_len=cursor,
        )
    ]


def _run(items, group, encode):
    with patch.object(mm_utils, "_mm_encode_sync_group", return_value=group):
        return mm_utils._batch_encode_per_image_misses(
            encode, _requests(items), torch.device("cpu")
        )


def test_local_hit_peer_miss_forces_reencode():
    item = _item(11, 4)
    cached = torch.ones(4, 2)
    mm_utils.embedding_cache.set(11, mm_utils.EmbeddingResult(embedding=cached))
    group = _FakeAttnTpGroup(peer_flags=[1])
    encode = Mock(return_value=torch.full((4, 2), 2.0))

    out = _run([item], group, encode)

    encode.assert_called_once()
    assert encode.call_args.args[0] == [item]
    assert group.calls[0].tolist() == [0]
    assert torch.equal(out[11], torch.full((4, 2), 2.0))


def test_all_ranks_hit_skips_encoder_and_still_syncs():
    item = _item(12, 3)
    mm_utils.embedding_cache.set(
        12, mm_utils.EmbeddingResult(embedding=torch.ones(3, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[0])
    encode = Mock()

    out = _run([item], group, encode)

    encode.assert_not_called()
    assert len(group.calls) == 1
    assert torch.equal(out[12], torch.ones(3, 2))


def test_encode_order_follows_batch_order_not_local_miss_order():
    a, b = _item(21, 2), _item(22, 3)
    # This rank hit `a` and missed `b`; a peer missed `a`.
    mm_utils.embedding_cache.set(
        21, mm_utils.EmbeddingResult(embedding=torch.ones(2, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[1, 0])
    encode = Mock(return_value=torch.arange(10.0).reshape(5, 2))

    out = _run([a, b], group, encode)

    assert encode.call_args.args[0] == [a, b]
    assert torch.equal(out[21], torch.arange(4.0).reshape(2, 2))
    assert torch.equal(out[22], torch.arange(4.0, 10.0).reshape(3, 2))


def test_flag_off_or_single_rank_does_not_sync():
    item = _item(31, 2)
    mm_utils.embedding_cache.set(
        31, mm_utils.EmbeddingResult(embedding=torch.ones(2, 2))
    )
    encode = Mock()

    with envs.SGLANG_MM_RANK_CONSISTENT_ENCODE.override(False):
        assert mm_utils._mm_encode_sync_group() is None

    out = _run([item], None, encode)
    encode.assert_not_called()
    assert torch.equal(out[31], torch.ones(2, 2))


def test_sync_group_none_without_distributed_init():
    with (
        envs.SGLANG_MM_RANK_CONSISTENT_ENCODE.override(True),
        patch.object(mm_utils.torch.distributed, "is_initialized", return_value=False),
    ):
        assert mm_utils._mm_encode_sync_group() is None


def test_rank_consistent_miss_hashes_is_union():
    group = _FakeAttnTpGroup(peer_flags=[0, 1, 0])
    got = mm_utils._rank_consistent_miss_hashes(
        [1, 2, 3], {1}, group, torch.device("cpu")
    )
    assert got == {1, 2}
