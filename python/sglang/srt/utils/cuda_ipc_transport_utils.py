"""Select the legacy or lease-pool CUDA IPC transport at process startup.

The feature flag is read once at import time; runtime flips are unsupported.
"""

from sglang.srt.environ import envs

PRECOMPUTED_FEATURE_HASHES_KEY = "_sglang_precomputed_feature_hashes"

if envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.get():
    from sglang.srt.multimodal.transport.cuda_ipc import (  # noqa: F401
        DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
        MM_FEATURE_CACHE_SIZE,
        MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
        CudaIpcTensorTransportProxy,
        MmItemMemoryPool,
        _pool_handle_cache_clear,
        get_mm_feature_pool_size_per_worker,
    )
else:
    from sglang.srt.utils.cuda_ipc_transport_utils_legacy import (  # noqa: F401
        DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
        MM_FEATURE_CACHE_SIZE,
        MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
        CudaIpcTensorTransportProxy,
        MmItemMemoryPool,
        _pool_handle_cache_clear,
        get_mm_feature_pool_size_per_worker,
    )

__all__ = [
    "DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY",
    "MM_FEATURE_CACHE_SIZE",
    "MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL",
    "PRECOMPUTED_FEATURE_HASHES_KEY",
    "CudaIpcTensorTransportProxy",
    "MmItemMemoryPool",
    "_pool_handle_cache_clear",
    "get_mm_feature_pool_size_per_worker",
]
