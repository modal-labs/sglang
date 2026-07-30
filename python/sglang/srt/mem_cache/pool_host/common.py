from __future__ import annotations

import ctypes
import glob
import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap

logger = logging.getLogger(__name__)


def _distributed_rank() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


# Chunk base pointers of every buffer registered through _cuda_host_register,
# keyed by the buffer's base data_ptr, so destroy() can unregister each chunk.
_REGISTERED_CHUNK_PTRS: dict[int, list[int]] = {}
_REGISTERED_CHUNK_LOCK = threading.Lock()

_cudart_ctypes = None
_cudart_ctypes_lock = threading.Lock()


def _load_cudart_ctypes():
    """Load libcudart via ctypes for GIL-releasing cudaHostRegister calls.

    torch.cuda.cudart()'s pybind wrapper holds the GIL across the whole call,
    which serializes multi-threaded registration; ctypes drops the GIL for the
    duration of each foreign call. Returns None when no libcudart is loadable
    (callers must fall back to the torch binding).
    """
    global _cudart_ctypes
    if _cudart_ctypes is not None:
        return _cudart_ctypes
    with _cudart_ctypes_lock:
        if _cudart_ctypes is not None:
            return _cudart_ctypes
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        candidates = sorted(
            glob.glob(os.path.join(torch_lib_dir, "libcudart*.so*")), reverse=True
        ) + ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
                lib.cudaHostRegister.restype = ctypes.c_int
                lib.cudaHostRegister.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_uint,
                ]
                lib.cudaHostUnregister.restype = ctypes.c_int
                lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
                lib.cudaGetErrorString.restype = ctypes.c_char_p
                lib.cudaGetErrorString.argtypes = [ctypes.c_int]
                lib.cudaSetDevice.restype = ctypes.c_int
                lib.cudaSetDevice.argtypes = [ctypes.c_int]
            except (OSError, AttributeError):
                continue
            _cudart_ctypes = lib
            return lib
    return None


def _register_threads(num_chunks: int) -> int:
    threads = envs.SGLANG_HICACHE_HOST_REGISTER_THREADS.get()
    if threads <= 0:
        threads = min(8, os.cpu_count() or 1)
    return max(1, min(threads, num_chunks))


def _cuda_host_register(buffer: torch.Tensor) -> None:
    """Pin ``buffer`` with cudaHostRegister, in chunks.

    Chunking keeps every single call well under the Blackwell driver's
    single-registration failure zone (see SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB)
    and lets registration run on several threads. A device transfer that spans
    a chunk boundary degrades to a staged copy but stays correct; HiCache moves
    page-granular ranges, so spanning transfers are rare.
    """
    base_ptr = buffer.data_ptr()
    n_bytes = buffer.numel() * buffer.element_size()
    chunk_bytes = max(1, envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get()) * 1024**3
    chunks = [
        (base_ptr + off, min(chunk_bytes, n_bytes - off))
        for off in range(0, n_bytes, chunk_bytes)
    ]
    threads = _register_threads(len(chunks))
    lib = _load_cudart_ctypes()
    if threads > 1 and lib is None:
        logger.warning(
            "SGLANG_HICACHE_HOST_REGISTER_THREADS=%d requested but libcudart is "
            "not loadable via ctypes; registering %d chunks serially.",
            threads,
            len(chunks),
        )
        threads = 1
    device_index = torch.cuda.current_device()
    registered: list[int] = []
    registered_lock = threading.Lock()

    def _register_one(chunk: tuple[int, int]) -> None:
        ptr, size = chunk
        if lib is not None:
            lib.cudaSetDevice(device_index)
            rc = lib.cudaHostRegister(ptr, size, 0)
            err = lib.cudaGetErrorString(rc).decode() if rc != 0 else ""
        else:
            cudart = torch.cuda.cudart()
            rc = int(cudart.cudaHostRegister(ptr, size, 0))
            err = cudart.cudaGetErrorString(rc) if rc != 0 else ""
        if rc != 0:
            raise RuntimeError(
                f"cudaHostRegister failed (rc={rc}, {err}) for ptr={ptr:#x} "
                f"size={size} (chunk of {n_bytes}-byte buffer at "
                f"{base_ptr:#x}); host buffer is not pinned and device "
                f"transfers may silently return stale data."
            )
        with registered_lock:
            registered.append(ptr)

    start = time.perf_counter()
    try:
        if threads == 1:
            for chunk in chunks:
                _register_one(chunk)
        else:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for future in [pool.submit(_register_one, c) for c in chunks]:
                    future.result()
    except Exception:
        # Unpin whatever succeeded so a retry or teardown starts clean.
        for ptr in registered:
            _unregister_ptr(ptr)
        raise
    logger.info(
        "HiCache host buffer phase=cuda_host_register_chunks state=done "
        "bytes=%d gib=%.2f chunks=%d chunk_gib=%d threads=%d elapsed_s=%.3f",
        n_bytes,
        n_bytes / (1024**3),
        len(chunks),
        chunk_bytes // 1024**3,
        threads,
        time.perf_counter() - start,
    )
    with _REGISTERED_CHUNK_LOCK:
        _REGISTERED_CHUNK_PTRS[base_ptr] = [ptr for ptr, _ in chunks]


def _unregister_ptr(ptr: int) -> None:
    lib = _load_cudart_ctypes()
    if lib is not None:
        rc = int(lib.cudaHostUnregister(ptr))
        err = lib.cudaGetErrorString(rc).decode() if rc != 0 else ""
    else:
        cudart = torch.cuda.cudart()
        rc = int(cudart.cudaHostUnregister(ptr))
        err = cudart.cudaGetErrorString(rc) if rc != 0 else ""
    if rc != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning("cudaHostUnregister failed (rc=%d, %s) for ptr=%#x", rc, err, ptr)


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    with _REGISTERED_CHUNK_LOCK:
        ptrs = _REGISTERED_CHUNK_PTRS.pop(buffer.data_ptr(), [buffer.data_ptr()])
    for ptr in ptrs:
        _unregister_ptr(ptr)


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    rank, world_size = _distributed_rank()
    n_bytes = math.prod(dims) * torch.empty((), dtype=dtype).element_size()
    allocate_start = time.perf_counter()
    logger.info(
        "HiCache host buffer phase=allocate state=start rank=%d/%d "
        "allocator=%s bytes=%d gib=%.2f dims=%s dtype=%s",
        rank,
        world_size,
        type(allocator).__name__,
        n_bytes,
        n_bytes / (1024**3),
        dims,
        dtype,
    )
    try:
        buffer = allocator.allocate(dims, dtype=dtype, device=device)
    except Exception:
        logger.exception(
            "HiCache host buffer phase=allocate state=failed rank=%d/%d "
            "allocator=%s bytes=%d gib=%.2f dims=%s dtype=%s elapsed_s=%.3f",
            rank,
            world_size,
            type(allocator).__name__,
            n_bytes,
            n_bytes / (1024**3),
            dims,
            dtype,
            time.perf_counter() - allocate_start,
        )
        raise
    logger.info(
        "HiCache host buffer phase=allocate state=done rank=%d/%d "
        "allocator=%s bytes=%d gib=%.2f dims=%s dtype=%s elapsed_s=%.3f",
        rank,
        world_size,
        type(allocator).__name__,
        n_bytes,
        n_bytes / (1024**3),
        dims,
        dtype,
        time.perf_counter() - allocate_start,
    )
    if pin_memory:
        register_start = time.perf_counter()
        logger.info(
            "HiCache host buffer phase=cuda_host_register state=start rank=%d/%d "
            "bytes=%d gib=%.2f dims=%s dtype=%s ptr=%#x",
            rank,
            world_size,
            n_bytes,
            n_bytes / (1024**3),
            dims,
            dtype,
            buffer.data_ptr(),
        )
        try:
            _cuda_host_register(buffer)
        except Exception:
            logger.exception(
                "HiCache host buffer phase=cuda_host_register state=failed "
                "rank=%d/%d bytes=%d gib=%.2f dims=%s dtype=%s ptr=%#x "
                "elapsed_s=%.3f",
                rank,
                world_size,
                n_bytes,
                n_bytes / (1024**3),
                dims,
                dtype,
                buffer.data_ptr(),
                time.perf_counter() - register_start,
            )
            raise
        logger.info(
            "HiCache host buffer phase=cuda_host_register state=done rank=%d/%d "
            "bytes=%d gib=%.2f dims=%s dtype=%s ptr=%#x elapsed_s=%.3f",
            rank,
            world_size,
            n_bytes,
            n_bytes / (1024**3),
            dims,
            dtype,
            buffer.data_ptr(),
            time.perf_counter() - register_start,
        )
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
