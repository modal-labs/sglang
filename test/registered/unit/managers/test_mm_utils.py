import unittest

import torch

import sglang.srt.managers.mm_utils as mm_utils
from sglang.srt.managers.mm_utils import (
    _can_use_per_item_image_cache_fallback,
    _get_per_item_image_embeddings_with_fallback,
    count_image_patches,
    init_mm_embedding_cache,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_image_item(hash_value: int, start: int, end: int) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_value,
        offsets=[(start, end)],
        feature=torch.zeros((end - start + 1, 2), dtype=torch.float32),
    )


def _make_embedding(item: MultimodalDataItem, value: float) -> torch.Tensor:
    start, end = item.offsets[0]
    token_length = end - start + 1
    return torch.full((token_length, 2), value, dtype=torch.float32)


class TestMmUtilsPerItemEmbeddingFallback(unittest.TestCase):
    def setUp(self):
        init_mm_embedding_cache(1024 * 1024)
        self.embedder_calls = []

    def _fake_embedder(self, items):
        self.embedder_calls.append([item.hash for item in items])
        embeddings = [
            _make_embedding(item, float(item.hash % 1000)) for item in items
        ]
        return torch.cat(embeddings, dim=0)

    def test_can_use_per_item_fallback_for_split_images(self):
        items = [
            _make_image_item(101, 0, 1),
            _make_image_item(102, 2, 4),
        ]
        items_offset = [(0, 1), (2, 4)]

        self.assertTrue(
            _can_use_per_item_image_cache_fallback(items, items_offset)
        )

    def test_reuses_cached_images_and_only_encodes_misses(self):
        item_a = _make_image_item(101, 0, 1)
        item_b = _make_image_item(102, 2, 4)
        item_c = _make_image_item(103, 5, 5)

        cache_a = _make_embedding(item_a, 11.0)
        cache_c = _make_embedding(item_c, 33.0)
        mm_utils.embedding_cache.set(
            MultiModalStaticCache.combine_hashes([item_a.hash]),
            EmbeddingResult(embedding=cache_a),
        )
        mm_utils.embedding_cache.set(
            MultiModalStaticCache.combine_hashes([item_c.hash]),
            EmbeddingResult(embedding=cache_c),
        )

        (
            embedding_result,
            cache_hit_count,
            vit_encode_count,
            cache_hit_patch_count,
            vit_encode_patch_count,
        ) = (
            _get_per_item_image_embeddings_with_fallback(
                self._fake_embedder, [item_a, item_b, item_c]
            )
        )

        self.assertIsNotNone(embedding_result)
        self.assertEqual(cache_hit_count, 2)
        self.assertEqual(vit_encode_count, 1)
        self.assertEqual(
            cache_hit_patch_count, count_image_patches([item_a, item_c])
        )
        self.assertEqual(vit_encode_patch_count, count_image_patches([item_b]))
        self.assertEqual(self.embedder_calls, [[item_b.hash]])

        expected = torch.cat(
            [
                cache_a,
                _make_embedding(item_b, float(item_b.hash % 1000)),
                cache_c,
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(embedding_result.embedding, expected))

        cached_b = mm_utils.embedding_cache.get([item_b.hash])
        self.assertIsNotNone(cached_b)
        self.assertTrue(
            torch.equal(
                cached_b.embedding,
                _make_embedding(item_b, float(item_b.hash % 1000)),
            )
        )


if __name__ == "__main__":
    unittest.main()
