from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import mmap
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import common as pool_host_common
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, mmap_allocator


class _FakeCudart:
    """ctypes-libcudart stand-in recording register/unregister calls."""

    def __init__(self, fail_ptrs=frozenset()):
        self.registered = []
        self.unregistered = []
        self.set_devices = []
        self.fail_ptrs = set(fail_ptrs)

    def cudaSetDevice(self, device):
        self.set_devices.append(device)
        return 0

    def cudaHostRegister(self, ptr, size, flags):
        if ptr in self.fail_ptrs:
            return 2  # cudaErrorMemoryAllocation
        self.registered.append((ptr, size))
        return 0

    def cudaHostUnregister(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def cudaGetErrorString(self, rc):
        return b"fake error"


class TestPopulateFastPath(unittest.TestCase):
    def test_alloc_mmap_populates_without_map_populate(self):
        # 3 MiB spans several populate shards once thread count > 1.
        dims = (3, 1024, 1024)
        with envs.SGLANG_HICACHE_HOST_POPULATE_THREADS.override(4):
            tensor = alloc_mmap(dims, torch.uint8)
        self.assertEqual(tensor.shape, dims)
        tensor[-1, -1, -1] = 7
        self.assertEqual(tensor[-1, -1, -1].item(), 7)

    def test_alloc_mmap_single_thread_override(self):
        with envs.SGLANG_HICACHE_HOST_POPULATE_THREADS.override(1):
            tensor = alloc_mmap((8, 1024), torch.float32)
        tensor[0, 0] = 1.5
        self.assertEqual(tensor[0, 0].item(), 1.5)

    def test_alloc_mmap_falls_back_to_map_populate(self):
        with mock.patch.object(
            mmap_allocator,
            "_madvise_populate_parallel",
            side_effect=OSError(22, "Invalid argument"),
        ):
            tensor = alloc_mmap((4, 1024), torch.float32)
        tensor[3, 1023] = 2.5
        self.assertEqual(tensor[3, 1023].item(), 2.5)

    def test_populate_shards_cover_range_exactly(self):
        captured = []

        def fake_madvise(addr, length, advice):
            captured.append((addr, length))
            return 0

        fake_libc = mock.Mock()
        fake_libc.madvise = fake_madvise
        alloc_bytes = 7 * mmap.PAGESIZE + mmap.PAGESIZE  # 8 pages
        with envs.SGLANG_HICACHE_HOST_POPULATE_THREADS.override(3):
            with mock.patch.object(mmap_allocator, "_libc", fake_libc):
                mmap_allocator._madvise_populate_parallel(0x1000, alloc_bytes)
        captured.sort()
        self.assertEqual(captured[0][0], 0x1000)
        self.assertEqual(sum(length for _, length in captured), alloc_bytes)
        for (addr, length), (next_addr, _) in zip(captured, captured[1:]):
            self.assertEqual(addr + length, next_addr)
            self.assertEqual(addr % mmap.PAGESIZE, 0)


class TestChunkedHostRegister(unittest.TestCase):
    def _register(self, tensor, fake, chunk_gb=1, threads=2):
        with envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.override(
            chunk_gb
        ), envs.SGLANG_HICACHE_HOST_REGISTER_THREADS.override(threads):
            with mock.patch.object(
                pool_host_common, "_load_cudart_ctypes", return_value=fake
            ), mock.patch.object(
                pool_host_common.torch.cuda, "current_device", return_value=0
            ):
                pool_host_common._cuda_host_register(tensor)

    def test_chunks_cover_buffer_and_unregister_uses_same_ptrs(self):
        fake = _FakeCudart()
        # 2.5 GiB buffer -> 3 chunks at 1 GiB.
        tensor = mock.Mock(spec=torch.Tensor)
        tensor.data_ptr.return_value = 0x10000
        tensor.numel.return_value = int(2.5 * 1024**3)
        tensor.element_size.return_value = 1

        self._register(tensor, fake, chunk_gb=1, threads=2)

        registered = sorted(fake.registered)
        self.assertEqual(len(registered), 3)
        self.assertEqual(registered[0][0], 0x10000)
        self.assertEqual(sum(size for _, size in registered), int(2.5 * 1024**3))
        for (ptr, size), (next_ptr, _) in zip(registered, registered[1:]):
            self.assertEqual(ptr + size, next_ptr)

        with mock.patch.object(
            pool_host_common, "_load_cudart_ctypes", return_value=fake
        ):
            pool_host_common._cuda_host_unregister(tensor)
        self.assertEqual(sorted(fake.unregistered), [ptr for ptr, _ in registered])
        # Registry entry is consumed.
        self.assertNotIn(0x10000, pool_host_common._REGISTERED_CHUNK_PTRS)

    def test_partial_failure_unregisters_registered_chunks(self):
        base = 0x20000
        fail_ptr = base + 1024**3  # second chunk fails
        fake = _FakeCudart(fail_ptrs={fail_ptr})
        tensor = mock.Mock(spec=torch.Tensor)
        tensor.data_ptr.return_value = base
        tensor.numel.return_value = 3 * 1024**3
        tensor.element_size.return_value = 1

        with self.assertRaises(RuntimeError):
            self._register(tensor, fake, chunk_gb=1, threads=1)
        # First chunk was registered, then rolled back.
        self.assertEqual(fake.registered, [(base, 1024**3)])
        self.assertEqual(fake.unregistered, [base])
        self.assertNotIn(base, pool_host_common._REGISTERED_CHUNK_PTRS)

    def test_unregister_without_registry_entry_falls_back_to_base_ptr(self):
        fake = _FakeCudart()
        tensor = mock.Mock(spec=torch.Tensor)
        tensor.data_ptr.return_value = 0x30000
        with mock.patch.object(
            pool_host_common, "_load_cudart_ctypes", return_value=fake
        ):
            pool_host_common._cuda_host_unregister(tensor)
        self.assertEqual(fake.unregistered, [0x30000])


if __name__ == "__main__":
    unittest.main()
