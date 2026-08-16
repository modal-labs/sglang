"""Bounded GPU image preprocessing: chunking, per-image streaming, parity.

Regression tests for the 2026-08-15 Kimi-K3 production incident: the GPU
preprocessing pipeline kept every image's fp32 patch tensor alive in a list
and finished with a request-wide ``torch.cat``, so peak tokenizer-side GPU
memory was >= 2x the request's total patch bytes — unbounded in the image
count. ``_gpu_preprocess_images`` now processes size-groups in bounded
sub-batches and hands each image's patches to a sink as they are produced,
so a sink that moves tensors off-GPU keeps the peak at one sub-batch plus
one image regardless of how many images a request carries.

These tests pin down:
- bit-exact parity of the per-image outputs with the legacy concat pipeline,
  across size-groups, sub-batch chunking, and the K3 RGBA compositing hook;
- output ordering and grid metadata;
- ownership: returned tensors must not be views pinning sub-batch storage;
- the memory bound itself, measured against the legacy pipeline's peak.
"""

import types
from collections import defaultdict

import pytest
import torch
from PIL import Image

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.multimodal.processors.kimi_k3 import (
    _fill_transparent_bg,
    _k3_to_cuda_chw,
)
from sglang.srt.multimodal.processors.kimi_k25 import (
    MMFeatureStreamSink,
    _default_to_cuda_chw,
    _gpu_preprocess_images,
    _grid_thw_from_resize_config,
    _process_single_image,
    _resize_images_by_source_shape,
    navit_resize_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

PATCH_SIZE = 14
MERGE = 2
IN_PATCH_LIMIT = 65536
SIDE_LIMIT = 512


def _norm_tensors():
    scale = torch.full((3, 1, 1), 1.0 / 127.5, device="cuda")
    bias = torch.full((3, 1, 1), -1.0, device="cuda")
    return scale, bias


def _image(width: int, height: int, seed: int, mode: str = "RGB") -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    channels = len(mode)
    arr = torch.randint(
        0, 256, (height, width, channels), generator=generator, dtype=torch.uint8
    )
    return Image.fromarray(arr.numpy(), mode=mode)


def _configs(images):
    return [
        navit_resize_config(
            *img.size, PATCH_SIZE, MERGE, IN_PATCH_LIMIT, SIDE_LIMIT, None
        )
        for img in images
    ]


def _legacy_reference(images, resize_configs, scale, bias, to_chw, post_resize=None):
    """The pre-change pipeline: whole-group batches, request-wide concat."""
    from sglang.kernels.ops.mm.process import normalize_and_patchify

    n = len(images)
    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, resize_configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        key = (config["new_height"], config["new_width"], padded_h, padded_w)
        groups[key].append((idx, image, config))

    all_patches = [None] * n
    all_grids = [None] * n
    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        if len(group) == 1:
            idx, image, config = group[0]
            all_patches[idx] = _process_single_image(
                image,
                config,
                scale,
                bias,
                PATCH_SIZE,
                to_chw=to_chw,
                post_resize=post_resize,
            )
            all_grids[idx] = _grid_thw_from_resize_config(config, PATCH_SIZE)
            continue
        indexed = [(idx, to_chw(image)) for idx, image, _ in group]
        resized = _resize_images_by_source_shape(indexed, target_h, target_w)
        if post_resize is not None:
            resized = [post_resize(part) for part in resized]
        batch = torch.cat(resized, dim=0)
        batch = normalize_and_patchify(
            batch, scale, bias, PATCH_SIZE, padded_h, padded_w
        )
        grid = (1, padded_h // PATCH_SIZE, padded_w // PATCH_SIZE)
        for i, (idx, _, _) in enumerate(group):
            all_patches[idx] = batch[i]
            all_grids[idx] = grid
    return torch.cat(all_patches, dim=0), torch.tensor(all_grids, dtype=torch.int64)


def _run_new(images, configs, scale, bias, to_chw, post_resize=None, **kwargs):
    entries, grids = _gpu_preprocess_images(
        images,
        configs,
        scale,
        bias,
        PATCH_SIZE,
        to_chw=to_chw,
        post_resize=post_resize,
        **kwargs,
    )
    return entries, grids


def test_parity_mixed_size_groups():
    """Multiple size-groups incl. a singleton: concat of per-image outputs
    must be bit-exact with the legacy request-wide pipeline."""
    images = [
        _image(640, 480, seed=0),
        _image(640, 480, seed=1),
        _image(1024, 768, seed=2),
        _image(333, 517, seed=3),
        _image(640, 480, seed=4),
    ]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    reference, ref_grids = _legacy_reference(
        images, configs, scale, bias, _default_to_cuda_chw
    )
    entries, grids = _run_new(images, configs, scale, bias, _default_to_cuda_chw)

    assert len(entries) == len(images)
    assert torch.equal(torch.cat(entries, dim=0), reference)
    assert torch.equal(grids, ref_grids)


def test_parity_single_group_chunked():
    """Same-size images forced through multiple sub-batches (the incident
    shape): chunking must not change a single output bit."""
    images = [_image(896, 896, seed=i) for i in range(7)]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    reference, ref_grids = _legacy_reference(
        images, configs, scale, bias, _default_to_cuda_chw
    )
    per_image_bytes = (
        (configs[0]["new_height"] + configs[0]["pad_height"])
        * (configs[0]["new_width"] + configs[0]["pad_width"])
        * 3
        * 4
    )
    # Force sub-batches of exactly 2 images.
    entries, grids = _run_new(
        images,
        configs,
        scale,
        bias,
        _default_to_cuda_chw,
        chunk_bytes=2 * per_image_bytes,
    )

    assert torch.equal(torch.cat(entries, dim=0), reference)
    assert torch.equal(grids, ref_grids)


def test_parity_k3_rgba_compositing():
    """The K3 hooks (RGBA-aware to_chw + transparent-background compositing)
    must survive chunking unchanged."""
    bg_config = {
        "pattern": "chessboard",
        "chessboard_square_size": 8,
        "chessboard_square_on_top_left": True,
        "chessboard_white_value": 255,
        "chessboard_gray_value": 180,
    }
    post = lambda x: _fill_transparent_bg(x, bg_config)  # noqa: E731
    images = [
        _image(512, 512, seed=10, mode="RGBA"),
        _image(512, 512, seed=11),
        _image(512, 512, seed=12, mode="RGBA"),
    ]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    reference, ref_grids = _legacy_reference(
        images, configs, scale, bias, _k3_to_cuda_chw, post_resize=post
    )
    entries, grids = _run_new(
        images,
        configs,
        scale,
        bias,
        _k3_to_cuda_chw,
        post_resize=post,
        chunk_bytes=1,  # one image per sub-batch: the strictest chunking
    )

    assert torch.equal(torch.cat(entries, dim=0), reference)
    assert torch.equal(grids, ref_grids)


def test_entries_own_their_storage():
    """Returned tensors must be independent copies, not views pinning the
    sub-batch: a view would keep the whole batch resident for as long as any
    single image is alive, defeating the bound."""
    images = [_image(448, 448, seed=i) for i in range(4)]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    entries, _ = _run_new(images, configs, scale, bias, _default_to_cuda_chw)
    for entry in entries:
        assert entry.untyped_storage().nbytes() == entry.nbytes


def test_sink_receives_every_image_in_order():
    images = [_image(448, 448, seed=i) for i in range(5)] + [_image(640, 480, seed=9)]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    seen = []

    def sink(index, patches):
        seen.append(index)
        return patches.cpu()

    entries, _ = _run_new(
        images, configs, scale, bias, _default_to_cuda_chw, per_image_sink=sink
    )
    assert sorted(seen) == list(range(len(images)))
    assert all(not entry.is_cuda for entry in entries)

    # Sink outputs are stored by original index: parity per image.
    reference, _ = _legacy_reference(images, configs, scale, bias, _default_to_cuda_chw)
    assert torch.equal(torch.cat([e.cuda() for e in entries], dim=0), reference)


def test_stream_sink_hashes_match_legacy_split():
    """MMFeatureStreamSink's per-image hashes must equal what the legacy
    path computed via set_pad_value on the post-split slices, so radix-cache
    keys are unchanged across the upgrade."""
    images = [_image(560, 420, seed=i) for i in range(3)]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    sink = MMFeatureStreamSink(types.SimpleNamespace(use_cuda_ipc=False))
    entries, grids = _run_new(
        images, configs, scale, bias, _default_to_cuda_chw, per_image_sink=sink
    )
    hashes = sink.hash_list(len(images))
    assert hashes is not None and all(not e.is_cuda for e in entries)

    reference, _ = _legacy_reference(images, configs, scale, bias, _default_to_cuda_chw)
    patches_per_image = [int(torch.prod(g).item()) for g in grids]
    start = 0
    for i, count in enumerate(patches_per_image):
        legacy_slice = reference[start : start + count]
        assert hashes[i] == hash_feature(legacy_slice)
        start += count


def test_peak_memory_bounded_vs_legacy():
    """The incident shape: many same-size images in one size-group. With an
    off-GPU sink and chunking, peak allocation must stay near one sub-batch,
    while the legacy pipeline's peak scales with (2x) the whole request."""
    images = [_image(1344, 1344, seed=i) for i in range(8)]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    padded_h = configs[0]["new_height"] + configs[0]["pad_height"]
    padded_w = configs[0]["new_width"] + configs[0]["pad_width"]
    per_image_bytes = padded_h * padded_w * 3 * 4
    total_bytes = per_image_bytes * len(images)
    chunk_bytes = 2 * per_image_bytes

    def measure(fn):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del result
        return peak

    legacy_peak = measure(
        lambda: _legacy_reference(images, configs, scale, bias, _default_to_cuda_chw)
    )
    bounded_peak = measure(
        lambda: _run_new(
            images,
            configs,
            scale,
            bias,
            _default_to_cuda_chw,
            per_image_sink=lambda index, patches: patches.cpu(),
            chunk_bytes=chunk_bytes,
        )
    )

    # Legacy: all patches live + request-wide concat >= 2x total.
    assert legacy_peak >= 2 * total_bytes, (legacy_peak, total_bytes)
    # Bounded: a sub-batch plus transient copies, independent of image count.
    # Allow 4x chunk for resize/patchify intermediates and the per-image clone.
    assert bounded_peak <= 4 * chunk_bytes, (bounded_peak, chunk_bytes)
    assert bounded_peak * 4 < legacy_peak, (bounded_peak, legacy_peak)


def test_empty_images_returns_empty():
    scale, bias = _norm_tensors()
    entries, grids = _gpu_preprocess_images([], [], scale, bias, PATCH_SIZE)
    assert entries == []
    assert grids.shape[0] == 0
