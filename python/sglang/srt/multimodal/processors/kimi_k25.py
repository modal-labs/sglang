import math
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.kernels.ops.mm.process import normalize_and_patchify
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    PRECOMPUTED_FEATURE_HASHES_KEY,
)

# ---------------------------------------------------------------------------
# GPU image preprocessing utilities (resize, pad, normalize, patchify on CUDA)
# ---------------------------------------------------------------------------


def navit_resize_config(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None = None,
) -> dict:
    """Compute NaViT resize target dimensions and token count.

    Pure math -- no image data needed, only (width, height).
    """
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:
    """Get (width, height) from a CUDA tensor or PIL Image."""
    if isinstance(image, torch.Tensor):
        # nvJPEG returns (C, H, W) uint8
        return image.shape[2], image.shape[1]
    return image.size  # PIL returns (width, height)


def _expand_image_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
) -> torch.Tensor:
    """Expand one placeholder per image without tokenizing the media string again."""
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_mask = input_ids == image_token_id
    placeholder_count = np.count_nonzero(placeholder_mask)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )

    repeats = np.ones(input_ids.shape, dtype=np.int64)
    repeats[placeholder_mask] = image_token_counts
    return torch.from_numpy(np.repeat(input_ids, repeats)).unsqueeze(0)


def _pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to (C, H, W) uint8 CUDA tensor."""
    arr = np.asarray(image.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def _ensure_chw_rgb(image: torch.Tensor) -> torch.Tensor:
    """Coerce an already-decoded (C, H, W) image tensor to 3-channel RGB.

    PIL inputs are RGB-normalized by _pil_to_cuda_chw, but pre-decoded
    tensor inputs (e.g. nvJPEG / cached CUDA tensors) keep their native
    channel count. Grayscale (1ch) or RGBA (4ch) images then break the
    downstream torch.cat over a batch of images, which requires a
    consistent channel dimension. Normalize every tensor to 3 channels.

    Also move the tensor to the GPU (matching _pil_to_cuda_chw) so a CPU
    input does not trip a device mismatch against the CUDA normalization
    constants downstream. No-op if already on the device.
    """
    image = image.cuda()
    if image.dim() == 2:  # (H, W) grayscale -> (1, H, W)
        image = image.unsqueeze(0)
    c = image.shape[0]
    if c == 3:
        return image
    if c == 1:
        return image.repeat(3, 1, 1)
    # RGBA or other multi-channel layouts: keep the first 3 channels.
    return image[:3]


def _resize_bicubic_if_needed(
    image: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """Match the checkpoint processor's ``PIL.Image.resize(..., BICUBIC)``.

    Kimi's HF processors only ever downscale (NaViT scale <= 1.0), and PIL's
    bicubic widens the kernel support by the scale factor, i.e. it always
    antialiases; ``F.interpolate`` needs ``antialias=True`` to do the same.
    PIL also returns uint8, so quantize the resized result back to integer
    pixel values before normalization (round-to-nearest, clipped to [0, 255]);
    without this the float overshoot leaks past the [-1, 1] range the model
    was trained on.
    """
    image = image.float()
    if image.shape[-2:] == (target_height, target_width):
        return image
    return (
        F.interpolate(
            image,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round_()
        .clamp_(0.0, 255.0)
    )


def _grid_thw_from_resize_config(config: dict, patch_size: int) -> tuple[int, int, int]:
    height = config["new_height"] + config["pad_height"]
    width = config["new_width"] + config["pad_width"]
    return 1, height // patch_size, width // patch_size


def _default_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return _pil_to_cuda_chw(image)
    return _ensure_chw_rgb(image)


def _process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    to_chw: Callable[
        [Union[torch.Tensor, Image.Image]], torch.Tensor
    ] = _default_to_cuda_chw,
    post_resize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Process a single image on GPU: resize -> pad -> normalize -> patchify.

    ``to_chw`` converts the input to a CUDA CHW tensor (models may keep an
    alpha channel here); ``post_resize`` runs on the resized ``(B, C, H, W)``
    batch before patchify (e.g. K3's transparent-background compositing).
    """
    image = to_chw(image)

    new_h, new_w = config["new_height"], config["new_width"]
    padded_h = new_h + config["pad_height"]
    padded_w = new_w + config["pad_width"]

    x = _resize_bicubic_if_needed(image.unsqueeze(0), new_h, new_w)
    if post_resize is not None:
        x = post_resize(x)

    return normalize_and_patchify(
        x, image_scale, image_bias, patch_size, padded_h, padded_w
    ).squeeze(0)


def _resize_images_by_source_shape(
    indexed_images: list[tuple[int, torch.Tensor]],
    target_height: int,
    target_width: int,
) -> list[torch.Tensor]:
    """Resize images while batching only inputs with an identical source layout.

    A NaViT target-size group can still contain images with different source
    dimensions.  Interpolation requires a rectangular batch, so preserve the
    individual path for those images and batch only equal ``(shape, dtype)``
    inputs.  The returned tensors retain the caller's original image order.
    """
    by_source_shape = defaultdict(list)
    for index, image in indexed_images:
        by_source_shape[(tuple(image.shape), image.dtype)].append((index, image))

    resized_by_index = {}
    for images in by_source_shape.values():
        if len(images) == 1:
            index, image = images[0]
            resized_by_index[index] = _resize_bicubic_if_needed(
                image.unsqueeze(0), target_height, target_width
            )
            continue

        source_batch = torch.cat([image.unsqueeze(0) for _, image in images], dim=0)
        resized_batch = _resize_bicubic_if_needed(
            source_batch, target_height, target_width
        )
        for local_index, (index, _) in enumerate(images):
            resized_by_index[index] = resized_batch[local_index : local_index + 1]

    return [resized_by_index[index] for index, _ in indexed_images]


def _gpu_preprocess_images(
    images: list[Union[torch.Tensor, Image.Image]],
    resize_configs: list[dict],
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    to_chw: Callable[
        [Union[torch.Tensor, Image.Image]], torch.Tensor
    ] = _default_to_cuda_chw,
    post_resize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    per_image_sink: Optional[Callable[[int, torch.Tensor], Any]] = None,
    chunk_bytes: Optional[int] = None,
) -> tuple[list, torch.Tensor]:
    """GPU preprocessing pipeline for a batch of images.

    Groups images with the same target padded size for batch processing, and
    bounds peak GPU memory in two ways:

    - Groups are processed in sub-batches of at most ``chunk_bytes`` of fp32
      pixel data, so a many-image request never materializes one giant
      resize/patchify batch.
    - Each image's patch tensor is handed to ``per_image_sink`` as soon as it
      is produced. A sink that moves the tensor off-GPU (IPC pool copy or
      ``.cpu()``) keeps peak usage at one sub-batch plus one image, however
      many images the request carries. Without a sink, the per-image tensor
      itself is kept.

    Returns a per-image list of sink outputs (patch tensors when no sink is
    given) and the stacked ``grid_thws``. Callers that need the legacy
    request-wide concat can ``torch.cat`` the returned list, at the cost of
    the doubled peak this signature exists to avoid.
    """
    n = len(images)
    if n == 0:
        return [], torch.empty(0, 3, dtype=torch.int64)

    if chunk_bytes is None:
        chunk_bytes = envs.SGLANG_MM_GPU_PREPROCESS_CHUNK_BYTES.get()

    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, resize_configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        target_h = config["new_height"]
        target_w = config["new_width"]
        groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))

    all_entries = [None] * n
    all_grids = [None] * n

    def emit(idx: int, patches: torch.Tensor) -> None:
        all_entries[idx] = per_image_sink(idx, patches) if per_image_sink else patches

    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        # fp32 working set per image in this group (resize output and the
        # patchify result are both this size).
        per_image_bytes = padded_h * padded_w * 3 * 4
        images_per_chunk = max(1, chunk_bytes // max(per_image_bytes, 1))

        for chunk_start in range(0, len(group), images_per_chunk):
            chunk = group[chunk_start : chunk_start + images_per_chunk]
            if len(chunk) == 1:
                idx, image, config = chunk[0]
                patches = _process_single_image(
                    image,
                    config,
                    image_scale,
                    image_bias,
                    patch_size,
                    to_chw=to_chw,
                    post_resize=post_resize,
                )
                emit(idx, patches)
                del patches
                all_grids[idx] = _grid_thw_from_resize_config(config, patch_size)
                continue

            indexed_images = []
            for idx, image, _ in chunk:
                indexed_images.append((idx, to_chw(image)))

            # One NaViT target group can include several original resolutions.
            # Batch only source-compatible images, which removes redundant
            # bicubic launches for common multi-image requests without padding
            # random-size inputs to a larger source resolution.
            resized = _resize_images_by_source_shape(indexed_images, target_h, target_w)
            del indexed_images
            if post_resize is not None:
                # Runs before the concat: a hook may change the channel count
                # (K3 composites RGBA onto a background, returning RGB), and
                # mixed 3/4-channel sources cannot be concatenated first.
                resized = [post_resize(part) for part in resized]
            batch = torch.cat(resized, dim=0)
            del resized

            T = 1
            gh, gw = padded_h // patch_size, padded_w // patch_size
            batch = normalize_and_patchify(
                batch,
                image_scale,
                image_bias,
                patch_size,
                padded_h,
                padded_w,
            )

            grid = (T, gh, gw)
            for i, (idx, _, _) in enumerate(chunk):
                # `batch[i]` is a view pinning the whole sub-batch storage;
                # clone so the sink owns exactly one image and the sub-batch
                # can be freed before the next chunk is processed.
                emit(idx, batch[i].clone())
                all_grids[idx] = grid
            del batch

    grid_thws = torch.tensor(all_grids, dtype=torch.int64)
    return all_entries, grid_thws


class MMFeatureStreamSink:
    """Per-request sink: hash and transport-wrap each image feature as it is
    produced, so the request-wide patch set never resides on the GPU at once.

    The patch hash is computed on the freshly produced GPU tensor, and
    ``hash_list`` folds in each image's grid. The tensor is then either copied
    into the bounded CUDA-IPC pool (falling back to ``.cpu()`` when the pool
    is full, exactly like the per-item wrap it replaces) or moved to host
    memory for non-IPC transports. Either way the GPU copy is dropped before
    the next image is processed.
    """

    def __init__(self, sglang_processor):
        self._sglang_processor = sglang_processor
        self._hashes: dict[int, int] = {}
        self._proxies: list = []

    def __call__(self, index: int, patches: torch.Tensor):
        from sglang.srt.managers.mm_utils import hash_feature

        if not envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            self._hashes[index] = hash_feature(patches)
        processor = self._sglang_processor
        if getattr(processor, "use_cuda_ipc", False):
            proxy = processor._wrap_tensor_for_cuda_ipc(patches)
            if envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.get():
                from sglang.srt.multimodal.transport.cuda_ipc import (
                    CudaIpcTensorTransportProxy,
                )

                if isinstance(proxy, CudaIpcTensorTransportProxy):
                    self._proxies.append(proxy)
            return proxy
        return patches.cpu()

    def hash_list(self, count: int, grid_thws) -> list:
        """Per-image identity hashes: patch bytes plus the image's grid.

        Entries are None when hashing is skipped; consumers must handle it.
        """
        from sglang.srt.managers.mm_utils import hash_feature_with_grid

        hashes = []
        for index in range(count):
            patch_hash = self._hashes.get(index)
            hashes.append(
                None
                if patch_hash is None
                else hash_feature_with_grid(patch_hash, grid_thws[index].tolist())
            )
        return hashes

    def cancel_all(self, context: str) -> int:
        processor = self._sglang_processor
        pool = getattr(processor, "cudaipc_mmfeature_pool", None)
        if pool is None:
            return 0
        from sglang.srt.multimodal.transport.lease_lifecycle import cancel_proxies

        proxies = list(self._proxies)
        self._proxies.clear()
        return cancel_proxies(pool, proxies, context=context)


# ---------------------------------------------------------------------------
# Kimi K2.5 GPU processor wrapper
# ---------------------------------------------------------------------------


class KimiGPUProcessorWrapper:
    """Wraps Kimi's HF processor to do GPU image preprocessing.

    GPU path: nvJPEG CUDA tensor / PIL -> _gpu_preprocess_images()
    CPU fallback: PIL -> medias kwarg -> original HF KimiK25Processor.__call__

    Exposes attributes that base class's process_mm_data needs so it behaves
    like a normal HF processor from the outside.
    """

    def __init__(
        self,
        hf_processor,
        image_token,
        image_token_id,
        patch_size,
        merge_kernel_size,
        in_patch_limit,
        patch_limit_on_one_side,
        fixed_output_tokens,
        image_mean,
        image_std,
    ):
        self._hf_processor = hf_processor
        self._image_token = image_token
        self._image_token_id = image_token_id
        self._patch_size = patch_size
        self._merge_kernel_size = merge_kernel_size
        self._in_patch_limit = in_patch_limit
        self._patch_limit_on_one_side = patch_limit_on_one_side
        self._fixed_output_tokens = fixed_output_tokens
        self._image_mean = image_mean
        self._image_std = image_std
        self._gpu_norm_tensors = None

        # Explicitly expose attributes that base class process_mm_data needs:
        # - image_processor: checked via isinstance(..., BaseImageProcessor)
        # - tokenizer: used for tokenization
        # - media_processor: used by CPU fallback path
        self.image_processor = hf_processor.image_processor
        self.tokenizer = hf_processor.tokenizer
        self.media_processor = hf_processor.media_processor

    def __call__(self, text=None, images=None, **kwargs):
        # process_mm_data passes images via kwargs["images"]
        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)
        feature_sink = kwargs.pop("sglang_feature_sink", None)

        if images and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids, feature_sink)
        return self._cpu_call(text, images, **kwargs)

    def _prepare_input_ids(self, input_text, resize_configs, original_input_ids):
        if original_input_ids is not None:
            return _expand_image_token_ids(
                original_input_ids,
                self._image_token_id,
                [config["num_tokens"] for config in resize_configs],
            )

        parts = input_text.split(self._image_token)
        result = [parts[0]]
        for config, part in zip(resize_configs, parts[1:]):
            result.append(self._image_token * config["num_tokens"] + part)
        expanded_text = "".join(result)
        return self._hf_processor.tokenizer(expanded_text, return_tensors="pt")[
            "input_ids"
        ]

    def _gpu_call(self, text, images, original_input_ids=None, feature_sink=None):
        """Bypass HF KimiK25VisionProcessor.preprocess entirely -- use GPU ops."""
        input_text = text[0] if isinstance(text, list) else text

        # 1. Compute resize configs (CPU math)
        resize_configs = []
        for image in images:
            w, h = _get_image_dimensions(image)
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        # 2. Reuse the request's existing tokenization when available. The
        # media placeholder expansion is exact and avoids tokenizing thousands
        # of repeated ``<|media_pad|>`` strings.
        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids
        )

        # 3. GPU image preprocessing. With a sink, each image is hashed and
        # handed to the transport as it is produced; "pixel_values" is then a
        # per-image list rather than a request-wide concat, so peak GPU usage
        # stays bounded regardless of the request's image count.
        image_scale, image_bias = self._get_gpu_norm_tensors()
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_scale,
            image_bias,
            self._patch_size,
            per_image_sink=feature_sink,
        )

        ret = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            # Use SGL-standard key so get_new_expanded_mm_items() can split
            # per-image for cache granularity (it looks up 'image_grid_thw').
            "image_grid_thw": grid_thws,
        }
        if feature_sink is not None:
            ret[PRECOMPUTED_FEATURE_HASHES_KEY] = feature_sink.hash_list(
                len(images), grid_thws
            )
        return ret

    def _cpu_call(self, text, images, **kwargs):
        """Fallback: token expansion + medias kwarg -> original HF processor."""
        input_text = text[0] if isinstance(text, list) else text

        if images:
            # Token expansion via media_tokens_calculator
            parts = input_text.split(self._image_token)
            result = [parts[0]]
            for image, part in zip(images, parts[1:]):
                num_tokens = self._hf_processor.media_processor.media_tokens_calculator(
                    {"type": "image", "image": image}
                )
                result.append(self._image_token * num_tokens + part)
            input_text = "".join(result)

            # Convert to medias format for Kimi's HF processor
            kwargs["medias"] = [{"type": "image", "image": img} for img in images]

        out = self._hf_processor(text=[input_text], **kwargs)
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out

    def _get_gpu_norm_tensors(self, device="cuda"):
        if self._gpu_norm_tensors is None:
            image_scale = torch.tensor(
                [1.0 / (255.0 * std) for std in self._image_std],
                device=device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            image_bias = torch.tensor(
                [-mean / std for mean, std in zip(self._image_mean, self._image_std)],
                device=device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            self._gpu_norm_tensors = (image_scale, image_bias)
        return self._gpu_norm_tensors


# ---------------------------------------------------------------------------
# Kimi K2.5 SGLang multimodal processor
# ---------------------------------------------------------------------------


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    models = [KimiK25ForConditionalGeneration]
    gpu_image_decode = True  # nvJPEG for JPEG, PIL fallback for others
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        media_proc_cfg = _processor.media_processor.media_proc_cfg
        processor = KimiGPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
        )
        # Initialize the executor from the final GPU wrapper. Cloning the raw
        # HF processor here would silently bypass Kimi's GPU preprocessing.
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        expected_image_count = len(image_data or [])
        if self.validate_tokenized_image_placeholders(
            input_text, self.mm_tokens.image_token_id, expected_image_count
        ):
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )

        if len(base_output.images) != expected_image_count:
            raise ValueError(
                "Kimi image placeholders must map one-to-one to image data: "
                f"expected {expected_image_count}, loaded {len(base_output.images)}"
            )

        # Stream each image's feature (hash -> transport wrap -> free) as it is
        # produced, bounding the tokenizer process's GPU footprint regardless
        # of the request's image count or resolution.
        if envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.get() and getattr(
            self, "use_cuda_ipc", False
        ):
            sink = MMFeatureStreamSink(self)
            try:
                mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
                    base_output,
                    self.mm_tokens,
                    sglang_original_input_ids=base_output.input_ids,
                    sglang_feature_sink=sink,
                )
                # K2.5/K2.7 encoder-DP assigns an image to exactly one TP rank.
                # Keep its IPC proxy lazy until that assignment is known,
                # avoiding a full image copy to every rank.
                if self.use_cuda_ipc and self.server_args.mm_enable_dp_encoder:
                    for item in mm_items:
                        item.model_specific_data[
                            DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY
                        ] = True
            except BaseException:
                sink.cancel_all(f"Kimi K2.5 request={request_obj.rid}")
                raise
        else:
            mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
                base_output,
                self.mm_tokens,
                sglang_original_input_ids=base_output.input_ids,
                sglang_feature_sink=MMFeatureStreamSink(self),
            )

            # K2.5/K2.7 encoder-DP assigns an image to exactly one TP rank. Keep
            # its IPC proxy lazy until that assignment is known, avoiding a full
            # image copy to every rank. The scheduler only honors this marker once
            # the processor has already set the item's hash and pad value.
            if self.use_cuda_ipc and self.server_args.mm_enable_dp_encoder:
                for item in mm_items:
                    item.model_specific_data[
                        DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY
                    ] = True

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        return self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
