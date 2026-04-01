"""
Multi-modality utils
"""

import copy
import hashlib
import pickle
import time
from abc import abstractmethod
from collections import defaultdict
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_rank
from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    CudaIpcTensorTransportProxy,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.multimodal.mm_utils import (
    consume_dp_mm_timing_accumulator,
    reset_dp_mm_timing_accumulator,
)
from sglang.srt.observability.request_waypoint_logger import (
    emit_request_waypoint,
    ms_from_s,
    request_waypoints_enabled,
    sum_grid_patches,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import flatten_nested_list, is_npu, print_warning_once
from sglang.utils import logger

_is_npu = is_npu()

# NOTE: Using the shared logger from sglang.utils instead of creating a module-specific logger
# to ensure consistent logging behavior across the codebase. This prevents issues with log
# propagation that can cause some log messages (like 'server is fired up') to not appear
# in the console when multimodal support is enabled.

# TODO(mick): nccl
# cuda_ipc: for intranode tensor sharing


def count_logical_image_items(mm_input: MultimodalInputs | None) -> int:
    if mm_input is None:
        return 0
    return count_logical_items([item for item in mm_input.mm_items if item.is_image()])


def count_logical_items(items: List[MultimodalDataItem] | None) -> int:
    if not items:
        return 0
    total = 0
    for item in items:
        if item is None:
            continue
        if item.offsets:
            total += len(item.offsets)
        else:
            total += 1
    return total


def count_image_patches(items: List[MultimodalDataItem] | None) -> int:
    if not items:
        return 0
    return sum_grid_patches(
        [
            getattr(item, "image_grid_thw", None)
            for item in items
            if item is not None
            and item.is_image()
            and getattr(item, "image_grid_thw", None) is not None
        ]
    )


def count_cached_logical_image_items(
    mm_input: MultimodalInputs | None,
    *,
    prefix_match_tokens: int,
) -> int:
    if mm_input is None or prefix_match_tokens <= 0:
        return 0
    cached = 0
    for item in mm_input.mm_items:
        if not item.is_image():
            continue
        if item.offsets:
            cached += sum(1 for _start, end in item.offsets if end < prefix_match_tokens)
        else:
            cached += 1
    return cached


def count_encoded_logical_image_items(
    mm_input: MultimodalInputs | None,
    *,
    extend_prefix_len: int,
    extend_seq_len: int,
) -> int:
    if mm_input is None or extend_seq_len <= 0:
        return 0
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len
    encoded = 0
    for item in mm_input.mm_items:
        if not item.is_image():
            continue
        if item.offsets:
            encoded += sum(
                1 for start, end in item.offsets if end >= chunk_start and start < chunk_end
            )
        else:
            encoded += 1
    return encoded


def empty_image_embed_waypoint_stats() -> dict[str, int]:
    return {
        "image_item_count_prefix_cached": 0,
        "image_item_count_embedding_cached": 0,
        "image_item_count_vit_encoded": 0,
        "image_patch_count_prefix_cached": 0,
        "image_patch_count_embedding_cached": 0,
        "image_patch_count_vit_encoded": 0,
    }


def merge_image_embed_waypoint_stats(*stats_list: dict[str, int] | None) -> dict[str, int]:
    merged = empty_image_embed_waypoint_stats()
    for stats in stats_list:
        if not stats:
            continue
        for key in merged:
            merged[key] += int(stats.get(key, 0))
    return merged
TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


_GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None
_BUFFER_OFFSET = 0

_is_default_tensor_transport = None


def init_feature_buffer(device):
    global _GPU_FEATURE_BUFFER, _BUFFER_OFFSET
    if (
        device == "cpu"
        or envs.SGLANG_MM_BUFFER_SIZE_MB.get() == 0
        or _GPU_FEATURE_BUFFER is not None
    ):
        return
    try:
        size_mb = envs.SGLANG_MM_BUFFER_SIZE_MB.get()
        num_elements = int(size_mb * 1024 * 1024 / 4)
        _GPU_FEATURE_BUFFER = torch.empty(
            num_elements, dtype=torch.float32, device=device
        )
        logger.info(f"Preallocated {size_mb}MB GPU buffer")
    except RuntimeError as e:
        _GPU_FEATURE_BUFFER = None


def reset_buffer_offset():
    global _BUFFER_OFFSET
    _BUFFER_OFFSET = 0


def is_feature_buffer_initialized():
    global _GPU_FEATURE_BUFFER
    if _GPU_FEATURE_BUFFER is None:
        return False
    return True


def try_add_to_buffer(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    global _BUFFER_OFFSET

    if _GPU_FEATURE_BUFFER is None:
        return tensor

    tensor_size = tensor.numel()

    if _BUFFER_OFFSET + tensor_size <= _GPU_FEATURE_BUFFER.numel():
        buffer_view = _GPU_FEATURE_BUFFER[_BUFFER_OFFSET : _BUFFER_OFFSET + tensor_size]
        buffer_view.copy_(tensor.flatten(), non_blocking=True)
        result = buffer_view.view(tensor.shape)
        _BUFFER_OFFSET += tensor_size
        return result
    else:
        return tensor


class TransportProxyTensor(torch.Tensor):
    """
    A convenient torch.Tensor subclass that carries extra metadata and supports
    efficient inter-process communications
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        name: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        transport_mode: TensorTransportMode = "default",
        *args,
        **kwargs,
    ):

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        instance = data.as_subclass(cls)

        instance._metadata = {
            "name": name,
            "fields": fields if fields is not None else {},
            "transport_mode": transport_mode,
        }

        return instance

    def __getstate__(self):
        """
        Called during pickling. Implements the serialization logic.
        """
        # acquire all serialize metadata from _metadata
        state = {
            "metadata": self._metadata,
            "tensor_data": None,
            "ipc_extra": None,
        }
        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and self.is_cuda:
            try:
                storage = self.untyped_storage()
                handle = storage._share_cuda_()

                state["ipc_extra"] = {
                    "handle": handle,
                    "shape": self.shape,
                    "dtype": self.dtype,
                    "stride": self.stride(),
                    "device_index": self.device.index,
                    "storage_offset": self.storage_offset(),
                }
                state["tensor_data"] = None
            except Exception as e:
                # Failed to get CUDA IPC handle (possibly tp). Falling back to default transport.
                state["metadata"]["transport_mode"] = "default"
                state["tensor_data"] = self.as_subclass(torch.Tensor)
        else:
            state["metadata"]["transport_mode"] = "default"
            state["tensor_data"] = self.as_subclass(torch.Tensor)

        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Called during unpickling. Implements the deserialization logic.
        """
        self._metadata = state["metadata"]

        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and state["ipc_extra"] is not None:
            ipc_extra = state["ipc_extra"]
            handle, shape, dtype, stride, source_device_index, s_offset = (
                ipc_extra["handle"],
                ipc_extra["shape"],
                ipc_extra["dtype"],
                ipc_extra["stride"],
                ipc_extra["device_index"],
                ipc_extra["storage_offset"],
            )

            try:
                target_device = torch.device(f"cuda:{source_device_index}")
                with torch.cuda.device(target_device):
                    storage = torch.UntypedStorage._new_shared_cuda(*handle)
                    reconstructed_tensor = torch.empty(
                        0, dtype=dtype, device=target_device
                    ).set_(storage, storage_offset=s_offset, size=shape, stride=stride)
                    self.set_(reconstructed_tensor)
            except Exception as e:
                print(f"Error: Failed to deserialize from CUDA IPC handle ({e}).")
                raise e

        elif state["tensor_data"] is not None:
            self.set_(state["tensor_data"])
        else:
            raise pickle.UnpicklingError(
                "Invalid state for TransportProxyTensor: no tensor data found."
            )

    @property
    def name(self) -> Optional[str]:
        return self._metadata.get("name")

    @property
    def fields(self) -> Dict[str, Any]:
        return self._metadata.get("fields", {})

    @property
    def transport_mode(self) -> TensorTransportMode:
        return self._metadata.get("transport_mode", "default")


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Pad the input ids sequence containing data tokens, and replace them with pad_values
        """
        pass


class MultiModalityDataPaddingPatternTokenPairs(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be enclosed by special token pairs (e.g. <image>...</image>, data_token_pairs)

    The padded value in a region enclosed by a token pair with be the same one, as the MultimodalDataItem's pad value

    This strategy should be applied when data content is marked by start/end token pairs in the input sequence.
    """

    def __init__(
        self,
        data_token_pairs: Optional[List[Tuple[int, int]]],
        data_start_token_ids: Optional[List[int]] = None,
    ) -> None:
        """

        Args:
            data_start_token_ids marks the start of a single multimodal data
            See Minicpmo's slice_start_id for example
        """
        self.data_token_id_pairs = data_token_pairs
        self.data_start_token_ids = data_start_token_ids or [
            s for s, _e in data_token_pairs
        ]

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        data_token_pairs = self.data_token_id_pairs
        mm_inputs.data_offsets = []
        if data_token_pairs is None:
            data_token_pairs = [mm_inputs.im_start_id, mm_inputs.im_end_id]
        if data_token_pairs is None:
            print_warning_once(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids
        start_token_ids = {s for s, _e in data_token_pairs}
        end_tokens_ids = {e for _s, e in data_token_pairs}

        padded_ids = []
        last_idx = 0
        data_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens_ids]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            if input_ids[start_idx] in self.data_start_token_ids:
                data_idx += 1
                mm_inputs.data_offsets += [start_idx]

            if data_idx >= len(pad_values):
                data_idx = len(pad_values) - 1

            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids), "Length validation fails"
        return padded_ids


class MultiModalityDataPaddingPatternMultimodalTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as repetitions of a single token
    e.g. <image><image>....<image>, or <audio><audio>...<audio>
    """

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Replaces multimodal tokens in input_ids with corresponding pad_values from mm_items.
        Each modality (image, audio, video) is handled separately based on its token_id.
        """
        if not input_ids or not mm_inputs.mm_items:
            return input_ids

        input_ids_tensor = torch.as_tensor(input_ids)

        # Replace multimodal tokens using per-item offsets
        items_by_modality = defaultdict(list)
        for item in mm_inputs.mm_items:
            items_by_modality[item.modality].append(item)

        token_id_map = {
            Modality.IMAGE: mm_inputs.im_token_id,
            Modality.AUDIO: mm_inputs.audio_token_id,
            Modality.VIDEO: mm_inputs.video_token_id,
        }

        for modality, items in items_by_modality.items():
            token_id = token_id_map.get(modality)

            if not items or token_id is None:
                continue

            for i, item in enumerate(items):
                per_offset_pad_values = getattr(item, "per_offset_pad_values", None)
                for j, offset in enumerate(items[i].offsets):
                    pad_value = (
                        per_offset_pad_values[j]
                        if per_offset_pad_values and j < len(per_offset_pad_values)
                        else item.pad_value
                    )
                    input_ids_tensor[offset[0] : offset[1] + 1] = pad_value

        ret_input_ids = input_ids_tensor.tolist()
        return ret_input_ids


embedding_cache: Optional[MultiModalStaticCache] = None


def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def _warn_embedding_cache_full():
    print_warning_once(
        "Multimodal embedding cache is full. This typically occurs when a single "
        "embedding exceeds the cache size limit. Consider increasing the "
        "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
        "embedding size."
    )


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:
    """
    Extract a chunk of embeddings based on the specified prefix length, sequence length, and offset ranges.

    Args:
        embedding: The full embedding tensor to extract a chunk from
        extend_prefix_len: The starting position (prefix length) for extraction
        extend_seq_len: The number of tokens to extract
        items_offset: List of [start, end] offset ranges for multimodal items in the input sequence

    Returns:
        A tuple containing:
        - The extracted embedding chunk as a tensor
        - The start index used for extraction
        - The end index used for extraction

    Note:
        If there's no overlap between the requested range and the offset ranges,
        an empty tensor is returned with zeros for start and end indices.
    """
    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.
    """
    precomputed_embeddings = []
    for idx, item in enumerate(items):
        if item.precomputed_embeddings is None:
            precomputed_embeddings.append(None)
            continue
        seq_start_idx = prefix_length[idx]
        seq_end_idx = seq_start_idx + extend_length[idx] - 1
        prefix_embedding_length = []
        extend_embedding_length = []
        for mm_start_idx, mm_end_idx in items_offset_list[idx]:
            if mm_start_idx > seq_end_idx:
                break
            if seq_start_idx > mm_start_idx:
                prefix_embedding_length.append(
                    min(seq_start_idx - mm_start_idx, mm_end_idx - mm_start_idx + 1)
                )
            if mm_end_idx >= seq_start_idx:
                extend_embedding_length.append(
                    min(
                        mm_end_idx - seq_start_idx + 1,
                        seq_end_idx - mm_start_idx + 1,
                        mm_end_idx - mm_start_idx + 1,
                        seq_end_idx - seq_start_idx + 1,
                    )
                )
        prefix_embedding_length = int(np.sum(prefix_embedding_length))
        extend_embedding_length = int(np.sum(extend_embedding_length))
        precomputed_embeddings.append(
            item.precomputed_embeddings[
                prefix_embedding_length : prefix_embedding_length
                + extend_embedding_length
            ]
        )

    if any(feature is not None for feature in precomputed_embeddings):
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )
        result = torch.concat(precomputed_embeddings)
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None


DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]], torch.Tensor | EVSEmbeddingResult
]


def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:
    """Move item features to the target device (in-place, non-blocking)."""
    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, bool]:
    """
    Fallback: encode all items at once, cache combined result, extract chunk.
    Used for non-bundled items or EVS results.
    """
    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = embedding_cache.get(item_hashes)
    cache_hit = embedding_per_req is not None

    if embedding_per_req is None:
        _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        embedding_per_req = (
            EmbeddingResult(embedding=embedding)
            if isinstance(embedding, torch.Tensor)
            else embedding
        )
        if not embedding_cache.set(embedding_items_hash, embedding_per_req):
            _warn_embedding_cache_full()

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids, cache_hit


def _get_chunked_embedding_by_item(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], dict[str, int]]:
    """
    Per-image chunk-aware encoding: only encode items overlapping with the
    current chunk, cache each item individually, and report waypoint stats for
    image items.
    """
    waypoint_image_stats = empty_image_embed_waypoint_stats()
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len  # exclusive

    if extend_seq_len <= 0:
        return None, waypoint_image_stats

    overlapping = []
    for idx, (item, offset) in enumerate(zip(embedding_items_per_req, items_offset)):
        start, end = offset
        if end >= chunk_start and start < chunk_end:
            overlapping.append((idx, item, start, end))

    if not overlapping:
        return None, waypoint_image_stats

    cached_embeddings = {}
    hit_items = []
    miss_items = []
    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding
            hit_items.append(item)
        else:
            miss_items.append((idx, item, start, end))

    miss_item_list = [item for _, item, _, _ in miss_items]
    if miss_items:
        _move_items_to_device(miss_item_list, device)
        all_miss_embedding = data_embedding_func(miss_item_list)
        all_miss_embedding = all_miss_embedding.reshape(
            -1, all_miss_embedding.shape[-1]
        )

        token_counts = [end - start + 1 for _, _, start, end in miss_items]
        split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)

        for (idx, item, _, _), emb in zip(miss_items, split_embeddings):
            cached_embeddings[idx] = emb
            emb_result = EmbeddingResult(embedding=emb)
            if not embedding_cache.set(item.hash, emb_result):
                _warn_embedding_cache_full()

    chunk_slices = []
    for idx, _, start, end in overlapping:
        emb = cached_embeddings[idx]
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)
        local_start = overlap_start - start
        local_end = overlap_end - start + 1
        chunk_slices.append(emb[local_start:local_end])

    if all(item.is_image() for _, item, _, _ in overlapping):
        waypoint_image_stats["image_item_count_embedding_cached"] += count_logical_items(
            hit_items
        )
        waypoint_image_stats["image_item_count_vit_encoded"] += count_logical_items(
            miss_item_list
        )
        waypoint_image_stats["image_patch_count_embedding_cached"] += count_image_patches(
            hit_items
        )
        waypoint_image_stats["image_patch_count_vit_encoded"] += count_image_patches(
            miss_item_list
        )

    return torch.cat(chunk_slices, dim=0), waypoint_image_stats


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, int]]:
    """Chunked prefill embedding with waypoint stats for image reuse."""
    embedding_list = []
    waypoint_image_stats = empty_image_embed_waypoint_stats()
    device = input_ids.device
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0
        is_image_only = bool(embedding_items_per_req) and all(
            item.is_image() for item in embedding_items_per_req
        )
        all_prefixed = all(offset_end < prefix_length[i] for _, offset_end in items_offset)

        if all_prefixed:
            if is_image_only:
                waypoint_image_stats["image_item_count_prefix_cached"] += count_logical_items(
                    embedding_items_per_req
                )
                waypoint_image_stats["image_patch_count_prefix_cached"] += count_image_patches(
                    embedding_items_per_req
                )
            continue

        is_per_item = all(len(item.offsets) == 1 for item in embedding_items_per_req)

        if is_per_item:
            if is_image_only:
                prefix_cached_items = [
                    item
                    for item, (_start, end) in zip(embedding_items_per_req, items_offset)
                    if end < extend_prefix_len
                ]
                waypoint_image_stats["image_item_count_prefix_cached"] += count_logical_items(
                    prefix_cached_items
                )
                waypoint_image_stats["image_patch_count_prefix_cached"] += count_image_patches(
                    prefix_cached_items
                )

            chunk_embedding, chunk_waypoint_stats = _get_chunked_embedding_by_item(
                data_embedding_func,
                embedding_items_per_req,
                items_offset,
                extend_prefix_len,
                extend_seq_len,
                device,
            )
            waypoint_image_stats = merge_image_embed_waypoint_stats(
                waypoint_image_stats, chunk_waypoint_stats
            )
            if chunk_embedding is not None:
                embedding_list.append(chunk_embedding)
        else:
            chunk_embedding, input_ids, cache_hit = _get_chunked_embedding_full(
                data_embedding_func,
                embedding_items_per_req,
                items_offset,
                extend_prefix_len,
                extend_seq_len,
                input_ids,
                device,
            )
            if is_image_only:
                image_item_count = count_logical_items(embedding_items_per_req)
                image_patch_count = count_image_patches(embedding_items_per_req)
                if cache_hit:
                    waypoint_image_stats["image_item_count_embedding_cached"] += (
                        image_item_count
                    )
                    waypoint_image_stats["image_patch_count_embedding_cached"] += (
                        image_patch_count
                    )
                else:
                    waypoint_image_stats["image_item_count_vit_encoded"] += image_item_count
                    waypoint_image_stats["image_patch_count_vit_encoded"] += image_patch_count
            if chunk_embedding is not None:
                embedding_list.append(chunk_embedding)

    if len(embedding_list) == 0:
        return None, input_ids, waypoint_image_stats
    return torch.concat(embedding_list, dim=0), input_ids, waypoint_image_stats


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    logger,
) -> torch.Tensor:
    num_mm_tokens_in_embedding = embedding.shape[0]
    num_mm_tokens_in_input_ids = mask.sum().item()
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_global_server_args().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )
            # extract from the end: this is a compromise
            if embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, dict[str, int]]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request

    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
        - If EVS is used, the pruned input ids tensor; otherwise, the original input ids tensor
    """
    # 1. Get embedding
    waypoint_image_stats = empty_image_embed_waypoint_stats()
    embedding = _get_precomputed_embedding(
        embedding_items, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids, waypoint_image_stats = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids, waypoint_image_stats
    # 2. Get mask
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask, input_ids, waypoint_image_stats


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]
    has_image_items = any(item.is_image() for item in item_flatten_list)
    image_waypoint_stats = empty_image_embed_waypoint_stats()

    # deepstack_embeddings: per-modality
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    # 2. Get multimodal embedding separately
    # Try get mm embedding if any
    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            _all_pad_values = set()
            for item in items:
                _po = getattr(item, "per_offset_pad_values", None)
                if _po:
                    _all_pad_values.update(_po)
                else:
                    _all_pad_values.add(item.pad_value)
            placeholder_tensor = torch.as_tensor(
                sorted(_all_pad_values),
                device=input_ids.device,
            )
            # calculate per request items length offset
            items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
            items_offsets = []
            for i, mm_inputs in enumerate(mm_inputs_list):
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size[i + 1] = len(mm_items)
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )
            items_size = torch.cumsum(items_size, dim=0).tolist()

            embedding, mask, input_ids, modality_waypoint_stats = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
            )
            if modality == Modality.IMAGE:
                image_waypoint_stats = merge_image_embed_waypoint_stats(
                    image_waypoint_stats, modality_waypoint_stats
                )

            if use_deepstack.get(modality, None) and embedding is not None:
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            else:
                deepstack_embeddings += [None]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # deepstack embedding
    if use_deepstack:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)

        deepstack_embedding_shape = input_embeds.shape[:-1] + (
            input_embeds.shape[-1] * num_deepstack_embeddings,
        )
        # a zero-filled embedding, with the same length of input_embeds, but different hidden_size
        input_deepstack_embeds = torch.zeros(
            deepstack_embedding_shape,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    # 4. scatter embeddings into input embedding
    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        # in-place update
        indices = torch.where(mask.squeeze(dim=-1))[0]
        input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)
        if use_deepstack.get(modality, None):
            input_deepstack_embeds[indices] = deepstack_embeddings[i].to(
                input_embeds.device, input_embeds.dtype
            )

    if has_image_items:
        other_info["waypoint_image_embed_stats"] = image_waypoint_stats

    return input_embeds, other_info


def _embed_mm_inputs_with_split(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
):
    """Split batch into precomputed vs non-precomputed, embed each group, merge back."""
    precomputed_req_indices = []
    non_precomputed_req_indices = []
    for idx, mm_input in enumerate(mm_inputs_list):
        items = [item for item in mm_input.mm_items if item is not None]
        if items and all(
            getattr(item, "precomputed_embeddings", None) is not None for item in items
        ):
            precomputed_req_indices.append(idx)
        else:
            non_precomputed_req_indices.append(idx)

    embed_kwargs = dict(
        multimodal_model=multimodal_model,
        input_embedding=input_embedding,
        data_embedding_func_mapping=data_embedding_func_mapping,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack,
    )

    if not precomputed_req_indices or not non_precomputed_req_indices:
        return embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            **embed_kwargs,
        )

    all_seq_lens = forward_batch.extend_seq_lens_cpu
    mm_batch_indices = [
        i for i, mm in enumerate(forward_batch.mm_inputs) if mm is not None
    ]
    token_starts = []
    cumulative = 0
    for sl in all_seq_lens:
        token_starts.append(cumulative)
        cumulative += sl

    vocab_size = input_embedding.num_embeddings
    input_embeds = input_embedding(input_ids.clamp(min=0, max=vocab_size - 1))
    other_info = {}
    has_image_items = any(
        item.is_image()
        for mm_input in mm_inputs_list
        for item in mm_input.mm_items
        if item is not None
    )
    image_waypoint_stats = empty_image_embed_waypoint_stats()

    input_deepstack_embeds = None
    if use_deepstack and multimodal_model is not None:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)
        input_deepstack_embeds = torch.zeros(
            input_ids.shape[0],
            input_embedding.embedding_dim * num_deepstack_embeddings,
            device=input_ids.device,
            dtype=input_embedding.weight.dtype,
        )
        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    for group_req_indices in [precomputed_req_indices, non_precomputed_req_indices]:
        sub_mm_inputs = [mm_inputs_list[i] for i in group_req_indices]
        sub_prefix_lens = [extend_prefix_lens[i] for i in group_req_indices]
        sub_seq_lens = [extend_seq_lens[i] for i in group_req_indices]
        group_batch_indices = [mm_batch_indices[i] for i in group_req_indices]
        sub_slices = [
            input_ids[token_starts[bi] : token_starts[bi] + all_seq_lens[bi]]
            for bi in group_batch_indices
        ]
        sub_input_ids = torch.cat(sub_slices)

        sub_embeds, sub_info = embed_mm_inputs(
            mm_inputs_list=sub_mm_inputs,
            extend_prefix_lens=sub_prefix_lens,
            extend_seq_lens=sub_seq_lens,
            input_ids=sub_input_ids,
            **embed_kwargs,
        )
        image_waypoint_stats = merge_image_embed_waypoint_stats(
            image_waypoint_stats, sub_info.get("waypoint_image_embed_stats")
        )

        offset = 0
        for bi in group_batch_indices:
            req_len = all_seq_lens[bi]
            start = token_starts[bi]
            input_embeds[start : start + req_len] = sub_embeds[
                offset : offset + req_len
            ]
            if (
                input_deepstack_embeds is not None
                and "input_deepstack_embeds" in sub_info
            ):
                input_deepstack_embeds[start : start + req_len] = sub_info[
                    "input_deepstack_embeds"
                ][offset : offset + req_len]
            offset += req_len

    if has_image_items:
        other_info["waypoint_image_embed_stats"] = image_waypoint_stats

    return input_embeds, other_info


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_funcs: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    use_deepstack: Dict[Modality, bool] = {},
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    if not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank:
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            server_args = get_global_server_args()
            waypoint_started_at = 0.0
            waypoint_payload = None
            if request_waypoints_enabled() and get_attention_tp_rank() == 0:
                reset_dp_mm_timing_accumulator()
                item_flatten_list = []
                for mm_input in mm_inputs_list:
                    item_flatten_list.extend(
                        [item for item in mm_input.mm_items if item is not None]
                    )
                image_items = [item for item in item_flatten_list if item.is_image()]
                video_items = [item for item in item_flatten_list if item.is_video()]
                image_item_count_total = sum(
                    count_logical_image_items(mm_input) for mm_input in mm_inputs_list
                )
                waypoint_started_at = time.perf_counter()
                waypoint_payload = {
                    "rids": forward_batch.rids,
                    "forward_mode": str(forward_batch.forward_mode),
                    "batch_size": int(forward_batch.batch_size),
                    "mm_request_count": len(mm_inputs_list),
                    "extend_seq_lens": [int(x) for x in extend_seq_lens],
                    "image_item_count": image_item_count_total,
                    "image_item_count_total": image_item_count_total,
                    "image_patch_total": sum_grid_patches(
                        [
                            getattr(item, "image_grid_thw", None)
                            for item in image_items
                            if getattr(item, "image_grid_thw", None) is not None
                        ]
                    ),
                    "video_patch_total": sum_grid_patches(
                        [
                            getattr(item, "video_grid_thw", None)
                            for item in video_items
                            if getattr(item, "video_grid_thw", None) is not None
                        ]
                    ),
                    "precomputed_item_count": sum(
                        1
                        for item in item_flatten_list
                        if getattr(item, "precomputed_embeddings", None) is not None
                    ),
                    "feature_item_count": sum(
                        1
                        for item in item_flatten_list
                        if getattr(item, "feature", None) is not None
                    ),
                }
            if server_args and server_args.enable_adaptive_dispatch_to_encoder:
                # Split by precomputed vs non-precomputed so get_embedding_and_mask only sees uniform batches
                input_embeds, other_info = _embed_mm_inputs_with_split(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )
            else:
                input_embeds, other_info = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )
            if waypoint_payload is not None:
                image_waypoint_stats = other_info.get(
                    "waypoint_image_embed_stats", empty_image_embed_waypoint_stats()
                )
                waypoint_payload.update(
                    {
                        "image_item_count_prefix_cached": image_waypoint_stats[
                            "image_item_count_prefix_cached"
                        ],
                        "image_item_count_embedding_cached": image_waypoint_stats[
                            "image_item_count_embedding_cached"
                        ],
                        "image_item_count_vit_encoded": image_waypoint_stats[
                            "image_item_count_vit_encoded"
                        ],
                        "image_item_count_encoded": image_waypoint_stats[
                            "image_item_count_vit_encoded"
                        ],
                        "image_patch_count_prefix_cached": image_waypoint_stats[
                            "image_patch_count_prefix_cached"
                        ],
                        "image_patch_count_embedding_cached": image_waypoint_stats[
                            "image_patch_count_embedding_cached"
                        ],
                        "image_patch_count_vit_encoded": image_waypoint_stats[
                            "image_patch_count_vit_encoded"
                        ],
                        "image_patch_count_encoded": image_waypoint_stats[
                            "image_patch_count_vit_encoded"
                        ],
                    }
                )
                waypoint_payload["mm_embed_ms"] = ms_from_s(
                    time.perf_counter() - waypoint_started_at
                )
                dp_timing = consume_dp_mm_timing_accumulator()
                if dp_timing is not None:
                    waypoint_payload.update(
                        {
                            "dp_assign_ms": round(dp_timing["dp_assign_ms"], 3),
                            "vit_forward_ms": round(
                                dp_timing["vit_forward_ms"], 3
                            ),
                            "all_gather_ms": round(
                                dp_timing["all_gather_ms"], 3
                            ),
                            "reorder_ms": round(dp_timing["reorder_ms"], 3),
                        }
                    )
                emit_request_waypoint("waypoint.batch.mm_embed", waypoint_payload)

            # add for qwen3_vl deepstack
            if use_deepstack:
                kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
            # Offload GPU features to CPU instead of discarding them to balance memory
            # efficiency and data persistence.
            # In chunked-prefill, a request is processed across multiple batches, and
            # the original multimodal data must remain accessible until the entire
            # prefill phase is complete. Since the multimodal embedding cache is
            # best-effort, offloading to CPU ensures we have a reliable fallback
            # if a cache miss occurs in subsequent chunks, while still freeing up
            # critical GPU memory.
            if mm_inputs_list:
                for mm_input_obj in mm_inputs_list:
                    if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
                        for mm_item in mm_input_obj.mm_items:
                            feature = getattr(mm_item, "feature", None)
                            if isinstance(feature, torch.Tensor) and feature.is_cuda:
                                mm_item.feature = feature.to("cpu", non_blocking=True)
                            if get_global_server_args().language_only:
                                precomputed_embeddings = getattr(
                                    mm_item, "precomputed_embeddings", None
                                )
                                if (
                                    isinstance(precomputed_embeddings, torch.Tensor)
                                    and precomputed_embeddings.is_cuda
                                ):
                                    mm_item.precomputed_embeddings = (
                                        precomputed_embeddings.to(
                                            "cpu", non_blocking=True
                                        )
                                    )
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
        else:
            input_embeds = embed_tokens(input_ids)
        # Copy to pre-allocated buffer if available (for CUDA graph address stability)
        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
    else:
        input_embeds = None

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        **kwargs,
    )
    return hidden_states


def get_multimodal_data_bounds(
    input_ids: torch.Tensor, pad_values: List[int], token_pairs: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Returns a tensor indicating the bounds of multimodal data (images, video, audio, etc.)

    Returns:
        [bounds_count, 2]
    """
    # All the multimodal data in the batch should share the same special bound token ids.
    start_tokens = {s for s, _e in token_pairs}
    end_tokens = {e for _s, e in token_pairs}

    assert all(isinstance(t, int) for t in start_tokens)
    assert all(isinstance(t, int) for t in end_tokens)

    start_cond = torch.isin(
        input_ids, torch.as_tensor(start_tokens, device=input_ids.device)
    )
    end_cond = torch.isin(
        input_ids, torch.as_tensor(end_tokens, device=input_ids.device)
    )

    (data_start_tokens,) = torch.where(start_cond)
    (data_end_tokens,) = torch.where(end_cond)

    data_start_tokens_cpu = data_start_tokens.cpu().tolist()
    data_end_tokens_cpu = data_end_tokens.cpu().tolist()

    # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the multimodal data
    if len(data_start_tokens_cpu) != len(data_end_tokens_cpu):
        if (
            len(data_start_tokens_cpu) + 1 == len(data_end_tokens_cpu)
            and input_ids[0].item() in pad_values
            and data_end_tokens_cpu
            and data_start_tokens_cpu
            and data_end_tokens_cpu[0] < data_start_tokens_cpu[0]
        ):
            data_start_tokens_cpu.insert(0, 0)
    valid_mm_data_nums = min(len(data_start_tokens_cpu), len(data_end_tokens_cpu))

    if valid_mm_data_nums == 0:
        return torch.zeros((0, 2), device=input_ids.device)

    # Filter out pairs where start_token >= end_token
    valid_pairs = []
    for i in range(valid_mm_data_nums):
        start_token = data_start_tokens_cpu[i]
        end_token = data_end_tokens_cpu[i]
        if start_token < end_token:
            valid_pairs.append((start_token + 1, end_token - 1))

    if not valid_pairs:
        return torch.zeros((0, 2), device=input_ids.device)

    # Convert valid pairs to tensor
    valid_pairs_tensor = torch.as_tensor(valid_pairs, device=input_ids.device)
    return valid_pairs_tensor


def data_hash(data) -> int:
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensors = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        # GPU path: concat + triton hash (unchanged)
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
            tensor = torch.concat(tensors)
            return gpu_tensor_hash(tensor.cuda())
        # CPU path: hash each tensor incrementally without concat
        hasher = hashlib.sha256()
        for t in tensors:
            t = t.detach().contiguous()
            hasher.update(memoryview(t.view(torch.uint8).numpy()))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    # Single tensor
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()
    hasher = hashlib.sha256()
    hasher.update(memoryview(tensor.view(torch.uint8).numpy()))
    hash_bytes = hasher.digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def hash_feature(f):
    if isinstance(f, list):
        if isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        hasher = hashlib.sha256()
        hasher.update(memoryview(arr))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    elif isinstance(f, CudaIpcTensorTransportProxy):
        reconstruct_t = f.reconstruct_on_target_device(torch.cuda.current_device())
        return tensor_hash([reconstruct_t])
    return data_hash(f)


def extend_mrope_positions_for_retracted_request(
    mrope_positions: torch.Tensor, output_ids_len: int
) -> torch.Tensor:
    """
    Extend mrope_positions for retracted requests by appending positions for output_ids.

    When a request is retracted and has multimodal inputs with mrope_positions,
    we need to extend the positions to cover the output_ids that were already generated.
    For pure text tokens, all three dimensions use the same incremental sequence.

    Args:
        mrope_positions: The original mrope positions tensor, shape (3, origin_input_ids_len)
        output_ids_len: The number of output tokens to generate positions for

    Returns:
        Extended mrope_positions tensor with shape (3, origin_input_ids_len + output_ids_len)
    """
    if output_ids_len <= 0:
        return mrope_positions

    # Get the last position value corresponding to origin_input_ids
    # mrope_positions shape: (3, origin_input_ids_len)
    last_position = mrope_positions[:, -1]  # shape: (3,)

    # Generate pure text mrope positions for output_ids
    # All three dimensions for pure text are the same incremental sequence
    start_pos = last_position[0] + 1  # Start from last position + 1
    output_positions = (
        torch.arange(
            start_pos,
            start_pos + output_ids_len,
            dtype=torch.int64,
            device=mrope_positions.device,
        )
        .unsqueeze(0)
        .expand(3, -1)
    )  # shape: (3, output_ids_len)

    # Concatenate to the original mrope_positions
    return torch.cat([mrope_positions, output_positions], dim=1)


def _get_length(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, np.ndarray):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _slice_value(value, start, end):
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, np.ndarray):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]
    try:
        return value[start:end]
    except Exception:
        return value


def _slice_model_data(
    data: dict,
    index: int,
    start: int,
    end: int,
    num_items: int,
    total_feature_len: Optional[int],
):
    sliced = {}
    for key, value in data.items():
        length = _get_length(value)
        if length == num_items:
            sliced[key] = _slice_value(value, index, index + 1)
        elif total_feature_len is not None and length == total_feature_len:
            sliced[key] = _slice_value(value, start, end)
        else:
            sliced[key] = value
    return sliced


def _try_simple_split(item, num_items, expanded_mm_items):
    """Try to split a bundled item by matching feature dim-0 to offset count.
    Returns True if split succeeded, False otherwise."""
    feature = item.feature if item.feature is not None else item.precomputed_embeddings
    if feature is None:
        return False

    if isinstance(feature, (torch.Tensor, np.ndarray)):
        feature_count = feature.shape[0]
    elif isinstance(feature, (list, tuple)):
        feature_count = len(feature)
    else:
        return False

    if feature_count != num_items:
        return False

    for i in range(num_items):
        new_item = copy.copy(item)
        if item.feature is not None:
            if isinstance(item.feature, (list, tuple)):
                new_item.feature = [item.feature[i]]
            else:
                new_item.feature = item.feature[i : i + 1]
        if item.precomputed_embeddings is not None:
            if isinstance(item.precomputed_embeddings, (list, tuple)):
                new_item.precomputed_embeddings = [item.precomputed_embeddings[i]]
            else:
                new_item.precomputed_embeddings = item.precomputed_embeddings[i : i + 1]
        new_item.offsets = [item.offsets[i]]
        new_data = {}
        for k, v in item.model_specific_data.items():
            if isinstance(v, (list, tuple)) and len(v) == num_items:
                new_data[k] = [v[i]]
            elif (
                isinstance(v, (torch.Tensor, np.ndarray))
                and len(v.shape) > 0
                and v.shape[0] == num_items
            ):
                new_data[k] = v[i : i + 1]
            else:
                new_data[k] = v
        new_item.model_specific_data = new_data
        new_item.hash = None
        expanded_mm_items.append(new_item)
    return True


def get_new_expanded_mm_items(original_mm_items):
    expanded_mm_items = []
    for item in original_mm_items:
        is_bundled = item.offsets is not None and len(item.offsets) > 1

        if is_bundled:
            num_items = len(item.offsets)

            if item.is_image():
                image_grid_thw = item.model_specific_data.get("image_grid_thw")
                grid_len = _get_length(image_grid_thw)
                if image_grid_thw is None or grid_len != num_items:
                    # No grid info — fall back to simple split by feature dim-0
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                patches_per_item = []
                for grid in image_grid_thw:
                    grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                    patches_per_item.append(int(torch.prod(grid_tensor).item()))

                cumulative = torch.cumsum(
                    torch.tensor(patches_per_item, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                for i in range(num_items):
                    start, end = slice_indices[i], slice_indices[i + 1]
                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    new_item.offsets = [item.offsets[i]]
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=i,
                        start=start,
                        end=end,
                        num_items=num_items,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    new_item.pad_value = None
                    new_item.__dict__.pop("per_offset_pad_values", None)
                    expanded_mm_items.append(new_item)

            elif item.is_video():
                video_grid_thw = item.model_specific_data.get("video_grid_thw")
                if video_grid_thw is None:
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = _get_length(video_grid_thw)
                num_videos = grid_len

                # Calculate total frames and frames per video
                frames_per_video = []
                total_frames = 0
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        T = int(grid[0].item())  # T is the first element [T, H, W]
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        T = int(grid_tensor[0].item())
                    frames_per_video.append(T)
                    total_frames += T

                # num_items should equal total_frames when T > 1
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # Calculate patches per video: T * H * W for each video
                patches_per_video = []
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        patches_per_video.append(int(torch.prod(grid).item()))
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        patches_per_video.append(int(torch.prod(grid_tensor).item()))

                # Calculate cumulative patches to get slice indices for each video
                cumulative = torch.cumsum(
                    torch.tensor(patches_per_video, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                # Group frames by video: calculate frame indices for each video
                frame_start_indices = [0]
                for i in range(num_videos):
                    frame_start_indices.append(
                        frame_start_indices[-1] + frames_per_video[i]
                    )

                # Expand each video into a separate item
                for video_idx in range(num_videos):
                    start, end = (
                        slice_indices[video_idx],
                        slice_indices[video_idx + 1],
                    )
                    frame_start, frame_end = (
                        frame_start_indices[video_idx],
                        frame_start_indices[video_idx + 1],
                    )

                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    # Group offsets for this video (all frames of this video)
                    new_item.offsets = item.offsets[frame_start:frame_end]
                    # For video_grid_thw, slice the corresponding row [T, H, W] for this video
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=video_idx,
                        start=start,
                        end=end,
                        num_items=num_videos,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    new_item.pad_value = None
                    new_item.__dict__.pop("per_offset_pad_values", None)
                    expanded_mm_items.append(new_item)
            else:
                if not _try_simple_split(item, num_items, expanded_mm_items):
                    expanded_mm_items.append(item)

        else:
            expanded_mm_items.append(item)
    return expanded_mm_items


class ShmPointerMMData:
    """
    Wraps a tensor to be sent via a shared memory handle.
    This acts as a "pointer" to the tensor data across process boundaries.
    """

    def __init__(self, tensor: torch.Tensor):
        if not tensor.is_cpu:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dst.copy_(tensor.view(torch.uint8).reshape(-1))
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        self.shm_name = shm.name
        shm.close()
        self._shm_handle = None

    def __getstate__(self):
        return {
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.shm = None
        self._shm_handle = shared_memory.SharedMemory(name=self.shm_name)
        # Zero-copy view into shared memory (no clone, no unlink)
        self.tensor = torch.frombuffer(self._shm_handle.buf, dtype=self.dtype).reshape(
            self.shape
        )

    def materialize(self) -> torch.Tensor:
        """Clone tensor from shm to owned memory, then release shm handle."""
        tensor = self.tensor.clone()
        if self._shm_handle is not None:
            self._shm_handle.close()
            try:
                self._shm_handle.unlink()
            except FileNotFoundError:
                pass  # Another rank already unlinked
            self._shm_handle = None
        return tensor

    def __del__(self):
        # Only close; never unlink. Unlinking is materialize()'s job.
        if getattr(self, "_shm_handle", None) is not None:
            self._shm_handle.close()
            self._shm_handle = None


def _get_is_default_transport():
    global _is_default_tensor_transport
    if _is_default_tensor_transport is None:
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
        )

        _is_default_tensor_transport = (
            _determine_tensor_transport_mode(get_global_server_args()) == "default"
        )
    return _is_default_tensor_transport


def wrap_shm_features(obj):
    """
    Scan the object for multimodal tensors and wrap them in SHM pointers.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj

    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            if (
                hasattr(item, "feature")
                and isinstance(item.feature, torch.Tensor)
                and item.feature.is_cpu
            ):
                item.feature = ShmPointerMMData(item.feature)
    return obj


def has_shm_features(recv_reqs):
    """Return True if any request in the list contains ShmPointerMMData."""
    for req in recv_reqs:
        if hasattr(req, "batch"):
            if has_shm_features(req.batch):
                return True
        elif hasattr(req, "mm_inputs") and req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if isinstance(item.feature, ShmPointerMMData):
                    return True
    return False


def unwrap_shm_features(obj):
    """
    Restore ShmPointerMMData wrappers back into standard torch.Tensors.
    Handles both single requests and batch requests.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj
    # Handle batch requests
    if hasattr(obj, "batch"):
        for sub_obj in obj.batch:
            unwrap_shm_features(sub_obj)
        return obj
    # Handle single requests
    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        mm_items = obj.mm_inputs.mm_items
        for item in mm_items:
            if isinstance(item.feature, ShmPointerMMData):
                item.feature = item.feature.materialize()
    return obj
