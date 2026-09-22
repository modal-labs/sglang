"""Kimi image identity must include the patch grid, not only the patch bytes.

The per-image hash derives the pad value (and so the radix-cache key) and keys
the multimodal embedding cache. A solid-colour image and its transpose, at
sizes that need no padding, produce byte-identical patch tensors with the same
token count but different grids, and the vision tower encodes them differently.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.managers.mm_utils import hash_feature, hash_feature_with_grid
from sglang.srt.multimodal.processors import kimi_k3
from sglang.srt.multimodal.processors.kimi_k25 import (
    MMFeatureStreamSink,
    _gpu_preprocess_images,
    navit_resize_config,
)
from sglang.srt.utils.cuda_ipc_transport_utils import PRECOMPUTED_FEATURE_HASHES_KEY
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PATCH_SIZE, MERGE, IN_PATCH_LIMIT, SIDE_LIMIT = 14, 2, 16384, 512


def _transposed_solid_images():
    # 448 and 896 are multiples of MERGE * PATCH_SIZE, so neither needs padding
    # and every patch of each image is identical.
    return [
        torch.full((3, 448, 896), 200, dtype=torch.uint8),
        torch.full((3, 896, 448), 200, dtype=torch.uint8),
    ]


def _norm_tensors():
    return torch.full((1, 3, 1, 1), 1.0 / 127.5), torch.full((1, 3, 1, 1), -1.0)


def _sink():
    return MMFeatureStreamSink(types.SimpleNamespace(use_cuda_ipc=False))


class TestKimiImageHashIncludesGrid(CustomTestCase):
    def test_sink_hashes_separate_identical_patches_on_different_grids(self):
        images = _transposed_solid_images()
        configs = [
            navit_resize_config(
                img.shape[2],
                img.shape[1],
                PATCH_SIZE,
                MERGE,
                IN_PATCH_LIMIT,
                SIDE_LIMIT,
            )
            for img in images
        ]
        scale, bias = _norm_tensors()
        sink = _sink()
        patches, grids = _gpu_preprocess_images(
            images,
            configs,
            scale,
            bias,
            PATCH_SIZE,
            to_chw=lambda image: image,
            per_image_sink=sink,
        )

        # Same patch bytes and token count, different grids.
        self.assertTrue(torch.equal(patches[0], patches[1]))
        self.assertEqual(configs[0]["num_tokens"], configs[1]["num_tokens"])
        self.assertEqual(grids.tolist(), [[1, 32, 64], [1, 64, 32]])

        hashes = sink.hash_list(len(images), grids)
        self.assertNotEqual(hashes[0], hashes[1])
        for i, patch in enumerate(patches):
            self.assertEqual(
                hashes[i],
                hash_feature_with_grid(hash_feature(patch), grids[i].tolist()),
            )

    def test_k3_processor_publishes_grid_aware_hashes(self):
        # Drive the K3 GPU entry point on CPU: only the CUDA conversion, the
        # cached normalization constants and the prompt expansion are stubbed.
        wrapper = object.__new__(kimi_k3.KimiK3GPUProcessorWrapper)
        wrapper._patch_size = PATCH_SIZE
        wrapper._merge_kernel_size = MERGE
        wrapper._in_patch_limit = IN_PATCH_LIMIT
        wrapper._patch_limit_on_one_side = SIDE_LIMIT
        wrapper._fixed_output_tokens = None
        wrapper._transparent_bg_config = None
        wrapper._gpu_norm_tensors = _norm_tensors()
        wrapper._prepare_input_ids = lambda *args: torch.tensor([[1, 2, 3]])

        with mock.patch.object(kimi_k3, "_k3_to_cuda_chw", lambda image: image):
            ret = wrapper._gpu_call(
                "<image><image>", _transposed_solid_images(), feature_sink=_sink()
            )

        self.assertTrue(torch.equal(ret["pixel_values"][0], ret["pixel_values"][1]))
        self.assertEqual(ret["image_grid_thw"].tolist(), [[1, 32, 64], [1, 64, 32]])
        hashes = ret[PRECOMPUTED_FEATURE_HASHES_KEY]
        self.assertEqual(len(hashes), 2)
        self.assertNotEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
