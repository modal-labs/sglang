"""CPU tests for multimodal embedding-cache hygiene.

Covers three invariants of the image-request cache path in mm_utils /
multimodal_cache (ports of upstream sgl-project/sglang#36595, #39120, #34995):

* a cache hit whose row count does not match the request's placeholder span is
  discarded and re-encoded instead of being silently cropped;
* admitted cache entries own their storage and the cache accounts bytes by
  ``untyped_storage().nbytes()`` so the cap bounds real device memory;
* placeholder counts are derived from host offsets instead of a
  ``mask.sum().item()`` device sync.

No engine, no GPU: drives the helpers directly with mock encoders.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers import mm_utils
from sglang.srt.managers.mm_utils import PerImageRequestInfo
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

HIDDEN = 4
_CPU = torch.device("cpu")


def _item(hash_value, start, end):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_value,
        offsets=[(start, end)],
        feature=torch.zeros(1),
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    mm_utils.init_mm_embedding_cache(1 << 20)
    yield
    mm_utils.init_mm_embedding_cache(1 << 20)


def _per_image_request(items, offsets, prefix_len=0, seq_len=None):
    if seq_len is None:
        seq_len = max(end for _, end in offsets) + 1
    return PerImageRequestInfo(
        req_idx=0,
        items=list(items),
        items_offset=list(offsets),
        extend_prefix_len=prefix_len,
        extend_seq_len=seq_len,
    )


# --------------------------------------------------------------------------- #
# (b) #36595: wrong-length cache hits are discarded and re-encoded
# --------------------------------------------------------------------------- #


def test_batched_path_discards_wrong_length_hit_and_reencodes():
    item = _item(7, 0, 3)  # 4 placeholder tokens
    stale = torch.ones(6, HIDDEN)
    mm_utils.embedding_cache.set(7, EmbeddingResult(embedding=stale))
    fresh = torch.full((4, HIDDEN), 2.0)
    encode = Mock(return_value=fresh)

    with patch.object(mm_utils, "_mm_encode_sync_group", return_value=None):
        out = mm_utils._batch_encode_per_image_misses(
            encode, [_per_image_request([item], [(0, 3)])], _CPU
        )

    encode.assert_called_once()
    assert encode.call_args.args[0] == [item]
    assert torch.equal(out[(7, 4)], fresh)
    cached = mm_utils.embedding_cache.get_single(7).embedding
    assert cached.shape[0] == 4
    assert torch.equal(cached, fresh)


def test_batched_path_matching_hit_is_not_reencoded():
    item = _item(8, 2, 4)  # 3 tokens
    good = torch.ones(3, HIDDEN)
    mm_utils.embedding_cache.set(8, EmbeddingResult(embedding=good))
    encode = Mock()

    with patch.object(mm_utils, "_mm_encode_sync_group", return_value=None):
        out = mm_utils._batch_encode_per_image_misses(
            encode, [_per_image_request([item], [(2, 4)])], _CPU
        )

    encode.assert_not_called()
    assert torch.equal(out[(8, 3)], good)


def test_batched_path_colliding_hash_different_span_not_deduplicated():
    # Same compact hash, different placeholder spans in one batch: both must be
    # encoded, keyed separately, and assembled with their own row counts.
    a = _item(9, 0, 1)  # 2 tokens
    b = _item(9, 2, 4)  # 3 tokens
    encode = Mock(return_value=torch.arange(5.0 * HIDDEN).reshape(5, HIDDEN))

    with patch.object(mm_utils, "_mm_encode_sync_group", return_value=None):
        out = mm_utils._batch_encode_per_image_misses(
            encode, [_per_image_request([a, b], [(0, 1), (2, 4)])], _CPU
        )

    assert encode.call_args.args[0] == [a, b]
    assert out[(9, 2)].shape[0] == 2
    assert out[(9, 3)].shape[0] == 3


def test_per_item_path_discards_wrong_length_hit_and_reencodes():
    item = _item(10, 0, 3)
    mm_utils.embedding_cache.set(10, EmbeddingResult(embedding=torch.ones(9, HIDDEN)))
    fresh = torch.full((4, HIDDEN), 5.0)
    encode = Mock(return_value=fresh)

    chunk = mm_utils._get_chunked_embedding_by_item(
        encode, [item], [(0, 3)], 0, 4, _CPU
    )

    encode.assert_called_once()
    assert torch.equal(chunk, fresh)
    assert mm_utils.embedding_cache.get_single(10).embedding.shape[0] == 4


def test_combined_path_discards_wrong_length_hit_and_reencodes():
    items = [_item(11, 0, 1), _item(12, 2, 3)]
    item_hashes = [item.hash for item in items]
    combined = MultiModalStaticCache.combine_hashes(item_hashes)
    mm_utils.embedding_cache.set(
        combined, EmbeddingResult(embedding=torch.ones(7, HIDDEN))
    )
    fresh = torch.full((4, HIDDEN), 3.0)
    encode = Mock(return_value=fresh)
    input_ids = torch.zeros(4, dtype=torch.long)

    chunk, _ = mm_utils._get_chunked_embedding_full(
        encode, items, [(0, 1), (2, 3)], 0, 4, input_ids, _CPU
    )

    encode.assert_called_once()
    assert torch.equal(chunk, fresh)
    assert mm_utils.embedding_cache.get(item_hashes).embedding.shape[0] == 4


def test_adjust_embedding_length_rejects_overlong_embedding():
    # The keep-last-N-rows compromise is gone: any mismatch is an error.
    server_args = SimpleNamespace(chunked_prefill_size=-1)
    with patch.object(mm_utils, "get_server_args", return_value=server_args):
        with pytest.raises(RuntimeError, match="does not match"):
            mm_utils._adjust_embedding_length(
                torch.zeros(6, HIDDEN), 4, mm_utils.logger
            )
        with pytest.raises(RuntimeError, match="does not match"):
            mm_utils._adjust_embedding_length(
                torch.zeros(2, HIDDEN), 4, mm_utils.logger
            )
    exact = torch.zeros(4, HIDDEN)
    assert mm_utils._adjust_embedding_length(exact, 4, mm_utils.logger) is exact


# --------------------------------------------------------------------------- #
# (c) #39120: clone admitted slices, account by storage bytes
# --------------------------------------------------------------------------- #


def test_cache_set_clones_view_and_accounts_storage_bytes():
    cache = MultiModalStaticCache(1 << 20)
    batch = torch.zeros(10, HIDDEN)
    a, b = torch.split(batch, [4, 6], dim=0)

    assert cache.set(1, EmbeddingResult(embedding=a))
    cached = cache.get_single(1).embedding
    own_bytes = 4 * HIDDEN * batch.element_size()
    assert cached.untyped_storage().nbytes() == own_bytes
    assert cached.untyped_storage().data_ptr() != batch.untyped_storage().data_ptr()
    assert torch.equal(cached, a)
    assert cache.current_size == own_bytes

    assert cache.set(2, EmbeddingResult(embedding=b))
    assert cache.current_size == own_bytes + 6 * HIDDEN * batch.element_size()


def test_cache_set_keeps_owning_tensor_without_copy():
    cache = MultiModalStaticCache(1 << 20)
    emb = torch.zeros(5, HIDDEN)
    assert cache.set(3, EmbeddingResult(embedding=emb))
    assert cache.get_single(3).embedding is emb
    assert cache.current_size == emb.untyped_storage().nbytes()


def test_cache_cap_bounds_storage_bytes_and_evicts_lru():
    row_bytes = HIDDEN * torch.zeros(1).element_size()
    cache = MultiModalStaticCache(max_size=8 * row_bytes)
    batch = torch.zeros(12, HIDDEN)
    a, b, c = torch.split(batch, [4, 4, 4], dim=0)

    assert cache.set(1, EmbeddingResult(embedding=a))
    assert cache.set(2, EmbeddingResult(embedding=b))
    assert cache.current_size == 8 * row_bytes
    assert cache.set(3, EmbeddingResult(embedding=c))
    assert not cache.has(1)
    assert cache.has(2) and cache.has(3)
    assert cache.current_size == 8 * row_bytes

    cache.free(2, None)
    assert cache.current_size == 4 * row_bytes
    assert cache.set(4, EmbeddingResult(embedding=torch.zeros(4, HIDDEN)))
    assert cache.current_size == 8 * row_bytes

    # Oversize entries are rejected even against an empty cache.
    cache.clear()
    assert cache.current_size == 0
    assert not cache.set(5, EmbeddingResult(embedding=torch.zeros(9, HIDDEN)))


def test_batched_encode_cache_entries_do_not_pin_batch():
    items = [_item(21, 0, 3), _item(22, 4, 9)]
    batch = torch.arange(10.0 * HIDDEN).reshape(10, HIDDEN)
    encode = Mock(return_value=batch)

    with patch.object(mm_utils, "_mm_encode_sync_group", return_value=None):
        out = mm_utils._batch_encode_per_image_misses(
            encode, [_per_image_request(items, [(0, 3), (4, 9)])], _CPU
        )

    for item, tokens in ((items[0], 4), (items[1], 6)):
        cached = mm_utils.embedding_cache.get_single(item.hash).embedding
        assert (
            cached.untyped_storage().nbytes() == tokens * HIDDEN * batch.element_size()
        )
        assert torch.equal(cached, out[(item.hash, tokens)])
    assert mm_utils.embedding_cache.current_size == 10 * HIDDEN * batch.element_size()


# --------------------------------------------------------------------------- #
# (d) #34995: placeholder counts from offsets, no mask.sum().item()
# --------------------------------------------------------------------------- #


def test_count_mm_tokens_in_extend_clips_to_chunk_window():
    offsets = [[(2, 5), (9, 14)], [(0, 3)]]
    # Request 0 extends [4, 12): item 0 contributes 4,5 -> 2; item 1 contributes 9..11 -> 3.
    # Request 1 extends [0, 2): contributes 0,1 -> 2.
    assert mm_utils._count_mm_tokens_in_extend([4, 0], [8, 2], offsets) == 7
    # Whole sequence.
    assert mm_utils._count_mm_tokens_in_extend([0, 0], [20, 4], offsets) == 14
    # No overlap.
    assert mm_utils._count_mm_tokens_in_extend([15, 4], [5, 1], offsets) == 0
    # Missing extend length counts as zero.
    assert mm_utils._count_mm_tokens_in_extend([0, 0], [20], offsets) == 10


class _SyncTrap(torch.Tensor):
    """Bool mask whose .item()/.sum().item() would be a device sync."""

    @classmethod
    def wrap(cls, t):
        return t.as_subclass(cls)

    def item(self):
        raise AssertionError("mask.sum().item() must not be called")

    def sum(self, *args, **kwargs):
        return _SyncTrap.wrap(super().sum(*args, **kwargs))


def _run_get_embedding_and_mask(input_ids, items, offsets, prefix, extend, encoder):
    with patch.object(
        mm_utils,
        "_get_multimodal_mask",
        side_effect=lambda ids, ph: _SyncTrap.wrap(torch.isin(ids, ph).unsqueeze(-1)),
    ):
        return mm_utils.get_embedding_and_mask(
            data_embedding_func=encoder,
            embedding_items=items,
            placeholder_tensor=torch.tensor([99], dtype=torch.long),
            input_ids=input_ids,
            items_size=[0, len(items)],
            prefix_length=[prefix],
            extend_length=[extend],
            items_offset_list=[offsets],
        )


def test_get_embedding_and_mask_derives_count_from_offsets_without_sync():
    input_ids = torch.tensor([1, 1, 99, 99, 99, 99, 1, 1], dtype=torch.long)
    items = [_item(31, 2, 5)]
    encoder = Mock(return_value=torch.ones(4, HIDDEN))

    emb, mask, out_ids = _run_get_embedding_and_mask(
        input_ids, items, [(2, 5)], 0, 8, encoder
    )

    assert emb.shape == (4, HIDDEN)
    assert int(torch.Tensor.sum(mask)) == 4
    assert out_ids is input_ids


def test_get_embedding_and_mask_raises_on_span_mismatch_instead_of_cropping():
    # The chunk assembler hands back 6 rows for a 4-token span: the old code
    # kept the last 4 rows silently; now it is an error before anything is
    # scattered.
    input_ids = torch.tensor([1, 1, 99, 99, 99, 99, 1, 1], dtype=torch.long)
    items = [_item(32, 2, 5)]
    server_args = SimpleNamespace(chunked_prefill_size=-1)

    with (
        patch.object(mm_utils, "get_server_args", return_value=server_args),
        patch.object(
            mm_utils,
            "_get_chunked_prefill_embedding",
            return_value=(torch.ones(6, HIDDEN), input_ids),
        ),
        pytest.raises(RuntimeError, match="does not match"),
    ):
        _run_get_embedding_and_mask(input_ids, items, [(2, 5)], 0, 8, Mock())


def test_get_embedding_and_mask_async_assert_checks_offset_count():
    input_ids = torch.tensor([1, 1, 99, 99, 99, 99, 1, 1], dtype=torch.long)
    items = [_item(33, 2, 5)]
    encoder = Mock(return_value=torch.ones(4, HIDDEN))

    with patch.object(mm_utils, "maybe_assert_sum") as assert_sum:
        _run_get_embedding_and_mask(input_ids, items, [(2, 5)], 0, 8, encoder)

    assert_sum.assert_called_once()
    mask_arg, expected = assert_sum.call_args.args[:2]
    assert expected == 4
    assert int(torch.Tensor.sum(mask_arg)) == 4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
