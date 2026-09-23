"""Unit tests for HiCache staged write-back host-pool dispatch."""

import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as manager_cache_controller
from sglang.srt.managers.cache_controller import CacheOperation as ManagerCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheController,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import (
    hicache_authority,
    hybrid_cache_controller,
    hybrid_pool_assembler,
)
from sglang.srt.mem_cache.hybrid_cache.hicache_authority import (
    HiCacheAuthority,
    WriteIntent,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    DSAIndexerPoolHost,
    HostPoolGroup,
    LogicalHostPool,
    MambaPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    enforce_hicache_host_budget,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MEMORY_POOL_HOST_MODULE = "sglang.srt.mem_cache.memory_pool_host"
MHA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mha"
MLA_POOL_HOST_MODULE = "sglang.srt.mem_cache.pool_host.mla"


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


def _ptr_key_from_layers(src_layers) -> tuple[int, ...]:
    return tuple(int(src_layers[i].data_ptr()) for i in range(len(src_layers)))


def _ptr_key_from_tensor(ptrs: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(ptr) for ptr in ptrs.cpu().tolist())


def _device_pool_stub(*, layer_num: int, **fields) -> SimpleNamespace:
    """Minimal device-pool stand-in with layer-split fields real pools expose."""
    return SimpleNamespace(
        layer_num=layer_num,
        layer_shard_enabled=False,
        **fields,
    )


def _cpu_staged_lf_pf_copy(
    src_registry,
    *,
    ptr_src,
    src_indices,
    dst_indices,
    dst,
    **_,
):
    src_layers = src_registry[_ptr_key_from_tensor(ptr_src)]
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    for layer_id, src in enumerate(src_layers):
        dst[dst_indices, layer_id] = src[src_indices]


def _cpu_staged_mha_lf_pf_copy(
    src_registry,
    *,
    k_ptr_src,
    v_ptr_src,
    src_indices,
    dst_indices,
    dst_k,
    dst_v,
    **_,
):
    k_src_layers = src_registry[_ptr_key_from_tensor(k_ptr_src)]
    v_src_layers = src_registry[_ptr_key_from_tensor(v_ptr_src)]
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    for layer_id, (k_src, v_src) in enumerate(zip(k_src_layers, v_src_layers)):
        dst_k[dst_indices, layer_id] = k_src[src_indices]
        dst_v[dst_indices, layer_id] = v_src[src_indices]


def _cpu_jit_one_layer_mha_copy(
    *,
    k_cache_dst,
    v_cache_dst,
    k_cache_src,
    v_cache_src,
    indices_dst,
    indices_src,
    **_,
):
    indices_dst = indices_dst.to(dtype=torch.int64, device="cpu")
    indices_src = indices_src.to(dtype=torch.int64, device="cpu")
    k_cache_dst[indices_dst] = k_cache_src[indices_src]
    v_cache_dst[indices_dst] = v_cache_src[indices_src]


def _cpu_jit_one_layer_mla_copy(
    *,
    cache_dst,
    cache_src,
    indices_dst,
    indices_src,
    **_,
):
    indices_dst = indices_dst.to(dtype=torch.int64, device="cpu")
    indices_src = indices_src.to(dtype=torch.int64, device="cpu")
    cache_dst[indices_dst] = cache_src[indices_src]


def _cpu_per_layer_pf_lf_copy(
    *,
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    **_,
):
    src_indices = src_indices.to(dtype=torch.int64, device="cpu")
    dst_indices = dst_indices.to(dtype=torch.int64, device="cpu")
    dst[dst_indices] = src[src_indices, layer_id]


class _FakeEvent:
    def __init__(self):
        self.waited_streams = []

    def record(self):
        pass

    def wait(self, stream=None):
        self.waited_streams.append(stream)

    def query(self):
        return False


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class TestHiCacheStagedWriteBackDispatch(unittest.TestCase):
    def _patched_transfers(self, src_registry=None, module=MEMORY_POOL_HOST_MODULE):
        staged_side_effect = None
        if src_registry is not None:
            staged_side_effect = lambda **kwargs: _cpu_staged_lf_pf_copy(
                src_registry, **kwargs
            )
        return (
            mock.patch(
                f"{module}.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                side_effect=staged_side_effect,
            ),
            mock.patch(
                f"{module}.transfer_kv_all_layer_mla_lf_pf",
                create=True,
            ),
            mock.patch(
                f"{module}.transfer_kv_per_layer_mla_pf_lf",
                side_effect=_cpu_per_layer_pf_lf_copy,
                create=True,
            ),
        )

    def test_mla_dedup_dummy_host_pools_are_allocator_only(self):
        mla_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            size=8,
            start_layer=0,
            end_layer=2,
        )
        mla_host = MLATokenToKVPoolHost(
            mla_device_pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(mla_host._is_dummy)
        self.assertIsNone(mla_host.kv_buffer)
        self.assertIsNone(mla_host.data_ptrs)
        self.assertEqual(mla_host.get_contiguous_buf_infos(), ([], [], []))
        slots = mla_host.alloc(2)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.tolist(), [0, 1])
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            mla_host.load_to_device_per_layer(
                mla_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

        dsa_device_pool = _device_pool_stub(
            layer_num=2,
            store_dtype=torch.float16,
            size=8,
            start_layer=0,
            end_layer=2,
            index_head_dim=8,
            quant_block_size=4,
        )
        indexer_host = DSAIndexerPoolHost(
            dsa_device_pool,
            mla_host,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )

        self.assertTrue(indexer_host._is_dummy)
        self.assertIsNone(indexer_host.index_k_with_scale_buffer)
        self.assertIsNone(indexer_host.index_k_device_ptrs)
        self.assertEqual(indexer_host.size, mla_host.size)
        with self.assertRaisesRegex(AssertionError, "load on a dummy"):
            indexer_host.load_to_device_per_layer(
                dsa_device_pool, slots, slots, layer_id=0, io_backend="kernel"
            )

    def test_mha_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        head_num = 1
        head_dim = 4
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        k_layers = [
            (torch.arange(8 * head_num * head_dim, dtype=torch.uint8) + layer_id * 40)
            .reshape(8, head_num, head_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        v_layers = [
            (
                torch.arange(8 * head_num * head_dim, dtype=torch.uint8)
                + 100
                + layer_id * 40
            )
            .reshape(8, head_num, head_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        expected_k = [layer[device_indices].clone() for layer in k_layers]
        expected_v = [layer[device_indices].clone() for layer in v_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            k_buffer=k_layers,
            v_buffer=v_layers,
            k_data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in k_layers], dtype=torch.uint64
            ),
            v_data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in v_layers], dtype=torch.uint64
            ),
        )

        host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host.layout = "page_first"
        host.page_size = 1
        host.layer_num = layer_num
        host.head_num = head_num
        host.head_dim = head_dim
        host.element_dim = head_num * head_dim
        host.token_stride_size = host.element_dim
        host.layout_dim = host.token_stride_size * layer_num
        host.dtype = torch.uint8
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(
            2, 8, layer_num, head_num, head_dim, dtype=torch.uint8
        )
        host.k_data_refs = [host.k_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.v_data_refs = [host.v_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_k_buffer = torch.empty(
            4, layer_num, head_num, head_dim, dtype=torch.uint8
        )
        host.staging_v_buffer = torch.empty_like(host.staging_k_buffer)
        src_registry = {
            _ptr_key_from_layers(k_layers): k_layers,
            _ptr_key_from_layers(v_layers): v_layers,
        }

        with (
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.jit_transfer_hicache_all_layer_staged_lf_pf",
                side_effect=lambda **kwargs: _cpu_staged_mha_lf_pf_copy(
                    src_registry, **kwargs
                ),
            ) as staged,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.transfer_kv_all_layer_lf_pf",
                create=True,
            ) as fallback,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer",
                side_effect=_cpu_jit_one_layer_mha_copy,
            ) as load,
            mock.patch(
                f"{MHA_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use_write_back_jit_kernel,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer in k_layers + v_layers:
                layer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        can_use_write_back_jit_kernel.assert_not_called()
        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(k_layers[layer_id][device_indices], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(v_layers[layer_id][device_indices], expected_v[layer_id])
            )
            self.assertTrue(
                torch.equal(host.k_buffer[host_indices, layer_id], expected_k[layer_id])
            )
            self.assertTrue(
                torch.equal(host.v_buffer[host_indices, layer_id], expected_v[layer_id])
            )

    def test_mla_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        kv_cache_dim = 5
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        device_layers = [
            (torch.arange(8 * kv_cache_dim, dtype=torch.uint8) + layer_id * 50)
            .reshape(8, 1, kv_cache_dim)
            .clone()
            for layer_id in range(layer_num)
        ]
        expected = [layer[device_indices].clone() for layer in device_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            kv_buffer=device_layers,
            data_ptrs=torch.tensor(
                [layer.data_ptr() for layer in device_layers], dtype=torch.uint64
            ),
        )

        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.device_pool = device_pool
        host.layout = "page_first"
        host.page_size = 1
        host.dcp_size = 1
        host.dcp_rank = 0
        host.layer_num = layer_num
        host.kv_cache_dim = kv_cache_dim
        host.token_stride_size = kv_cache_dim
        host.layout_dim = host.token_stride_size * layer_num
        host.dtype = torch.uint8
        host._is_dummy = False
        host.can_use_jit = True
        host.can_use_write_back_jit = True
        host.kv_buffer = torch.zeros(8, layer_num, 1, kv_cache_dim, dtype=torch.uint8)
        host.data_refs = [host.kv_buffer.transpose(0, 1)[i] for i in range(layer_num)]
        host.staging_buffer = torch.empty(
            4, layer_num, 1, kv_cache_dim, dtype=torch.uint8
        )
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, _ = self._patched_transfers(
            src_registry, module=MLA_POOL_HOST_MODULE
        )
        with (
            staged_patch as staged,
            fallback_patch as fallback,
            mock.patch(
                f"{MLA_POOL_HOST_MODULE}.jit_transfer_hicache_one_layer_mla",
                side_effect=_cpu_jit_one_layer_mla_copy,
            ) as load,
            mock.patch(
                f"{MLA_POOL_HOST_MODULE}.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use_write_back_jit_kernel,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer in device_layers:
                layer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        can_use_write_back_jit_kernel.assert_not_called()
        for layer_id, layer in enumerate(device_layers):
            self.assertTrue(torch.equal(layer[device_indices], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_indices, layer_id], expected[layer_id])
            )

    @unittest.skip(
        "TODO: Mamba pool is currently incompatible with write-back staging "
        "kernel; re-enable once the staging bug is fixed."
    )
    def test_mamba_backup_then_load_roundtrip_uses_staged(self):
        num_layers = 2
        host_indices = _indices(0, 4)
        device_indices = _indices(4, 8)
        temporal = torch.arange(num_layers * 8 * 3, dtype=torch.uint8).reshape(
            num_layers, 8, 1, 3
        )
        conv = (torch.arange(num_layers * 8 * 2, dtype=torch.uint8) + 97).reshape(
            num_layers, 8, 1, 2
        )
        device_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=temporal.clone(), conv=[conv.clone()])
        )
        expected_temporal = device_pool.mamba_cache.temporal[:, device_indices].clone()
        expected_conv = device_pool.mamba_cache.conv[0][:, device_indices].clone()

        host = MambaPoolHost.__new__(MambaPoolHost)
        host.layout = "page_first"
        host.num_mamba_layers = num_layers
        host.device_pool = SimpleNamespace(device="cpu")
        host.temporal_buffer = torch.zeros(8, num_layers, 1, 3, dtype=torch.uint8)
        host.conv_buffer = [
            torch.zeros(8, num_layers, 1, 2, dtype=torch.uint8),
        ]
        host.conv_state_shapes = [(2,)]
        host.temporal_staging_buffer = torch.empty(
            4, num_layers, 1, 3, dtype=torch.uint8
        )
        host.conv_staging_buffers = [
            torch.empty(4, num_layers, 1, 2, dtype=torch.uint8),
        ]
        host._temporal_can_use_jit = True
        host._conv_can_use_jit = [True]
        host.can_use_write_back_jit = True
        host.temporal_device_ptrs = torch.tensor(
            [layer.data_ptr() for layer in device_pool.mamba_cache.temporal],
            dtype=torch.uint64,
        )
        host.conv_device_ptrs = [
            torch.tensor(
                [layer.data_ptr() for layer in device_pool.mamba_cache.conv[0]],
                dtype=torch.uint64,
            )
        ]

        src_registry = {
            _ptr_key_from_layers(device_pool.mamba_cache.temporal): list(
                device_pool.mamba_cache.temporal
            ),
            _ptr_key_from_layers(device_pool.mamba_cache.conv[0]): list(
                device_pool.mamba_cache.conv[0]
            ),
        }

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            device_pool.mamba_cache.temporal.zero_()
            device_pool.mamba_cache.conv[0].zero_()
            for layer_id in range(num_layers):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 2)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, 4)
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.temporal[:, device_indices], expected_temporal
            )
        )
        self.assertTrue(
            torch.equal(
                device_pool.mamba_cache.conv[0][:, device_indices], expected_conv
            )
        )

    def test_deepseek_v4_paged_pool_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        slot_page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_rows = torch.tensor([0, 2], dtype=torch.int64)
        device_rows = torch.tensor([1, 3], dtype=torch.int64)
        device_buffers = [
            (torch.arange(5 * 4, dtype=torch.uint8) + layer_id * 50).reshape(5, 4)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_rows].clone() for buffer in device_buffers]

        host = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        host.pool_name = "c4"
        host.layout = "page_first"
        host.slot_page_size = slot_page_size
        host.layer_num = layer_num
        host.item_bytes = 4
        host.dtype = torch.uint8
        host.device_buffers = device_buffers
        host.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_buffers], dtype=torch.uint64
        )
        host.kv_buffer = torch.zeros(
            4, host.layer_num, host.item_bytes, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, host.item_bytes, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_buffers): device_buffers}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_buffers:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=None,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_buffers):
            self.assertTrue(torch.equal(buffer[device_rows], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_rows, layer_id], expected[layer_id])
            )

    def test_deepseek_v4_state_pool_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        swa_page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_rows = torch.tensor([0, 2], dtype=torch.int64)
        device_rows = torch.tensor([1, 3], dtype=torch.int64)
        device_page_views = [
            (torch.arange(5 * 5, dtype=torch.uint8) + layer_id * 60).reshape(5, 5)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_rows].clone() for buffer in device_page_views]

        host = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
        host.pool_name = "c4_state"
        host.layout = "page_first"
        host.swa_page_size = swa_page_size
        host.layer_num = layer_num
        host.state_page_bytes = 5
        host.dtype = torch.uint8
        host.device_page_views = device_page_views
        host.device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_page_views], dtype=torch.uint64
        )
        host.kv_buffer = torch.zeros(
            4, host.layer_num, host.state_page_bytes, dtype=torch.uint8
        )
        host.staging_buffer = torch.empty(
            4, host.layer_num, host.state_page_bytes, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_page_views): device_page_views}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_page_views:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=None,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_page_views):
            self.assertTrue(torch.equal(buffer[device_rows], expected[layer_id]))
            self.assertTrue(
                torch.equal(host.kv_buffer[host_rows, layer_id], expected[layer_id])
            )

    def test_dsa_indexer_backup_then_load_roundtrip_uses_staged(self):
        layer_num = 2
        page_size = 2
        host_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        device_indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        host_page_indices = torch.tensor([0, 2], dtype=torch.int64)
        device_page_indices = torch.tensor([1, 3], dtype=torch.int64)
        indexer_page_stride_size = 8
        device_layers = [
            (
                torch.arange(5 * indexer_page_stride_size, dtype=torch.uint8)
                + layer_id * 70
            ).reshape(5, 1, indexer_page_stride_size)
            for layer_id in range(layer_num)
        ]
        expected = [buffer[device_page_indices].clone() for buffer in device_layers]
        device_pool = _device_pool_stub(
            layer_num=layer_num,
            index_k_with_scale_buffer=device_layers,
        )

        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.device_pool = device_pool
        host.layout = "page_first"
        host.page_size = page_size
        host.layer_num = layer_num
        host.indexer_page_stride_size = indexer_page_stride_size
        host.indexer_layout_dim = host.layer_num * host.indexer_page_stride_size
        host.index_k_device_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in device_layers], dtype=torch.uint64
        )
        host.index_k_with_scale_buffer = torch.zeros(
            4, host.layer_num, 1, host.indexer_page_stride_size, dtype=torch.uint8
        )
        host._is_dummy = False
        host.staging_buffer = torch.empty(
            4, host.layer_num, 1, host.indexer_page_stride_size, dtype=torch.uint8
        )
        host.can_use_jit = False
        host.can_use_write_back_jit = True
        src_registry = {_ptr_key_from_layers(device_layers): device_layers}

        staged_patch, fallback_patch, load_patch = self._patched_transfers(src_registry)
        with staged_patch as staged, fallback_patch as fallback, load_patch as load:
            host.backup_from_device_all_layer(
                device_pool=device_pool,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel",
            )
            for buffer in device_layers:
                buffer.zero_()
            for layer_id in range(layer_num):
                host.load_to_device_per_layer(
                    device_pool=device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(fallback.call_count, 0)
        self.assertEqual(load.call_count, layer_num)
        for layer_id, buffer in enumerate(device_layers):
            self.assertTrue(
                torch.equal(buffer[device_page_indices], expected[layer_id])
            )
            self.assertTrue(
                torch.equal(
                    host.index_k_with_scale_buffer[host_page_indices, layer_id],
                    expected[layer_id],
                )
            )

    def test_logical_host_pool_preserves_page_first_group_layout(self):
        logical_host_pool = LogicalHostPool(8, 2, layout="page_first")
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=logical_host_pool,
                    device_pool=None,
                    layer_mapper=lambda _: 0,
                    is_primary_index_anchor=True,
                )
            ]
        )

        self.assertEqual(group.layout, "page_first")
        self.assertTrue(group.can_use_write_back_jit)

    def test_write_back_jit_hybrid_write_keeps_extra_host_indices_on_cpu(self):
        captured = {}

        class FakeHostGroup:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                io_backend,
                pool_transfers=None,
            ):
                captured["host_indices"] = host_indices
                captured["pool_transfers"] = pool_transfers

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [
            CacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.DEEPSEEK_V4_C4,
                        host_indices=_indices(0, 4),
                        device_indices=_indices(4, 8),
                    )
                ],
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = lambda *args: None
        controller.move_hybrid_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_hybrid_indices.assert_not_called()
        self.assertEqual(captured["host_indices"].device.type, "cpu")
        self.assertEqual(captured["pool_transfers"][0].host_indices.device.type, "cpu")
        self.assertEqual(
            controller.ack_write_queue[0].finish_event.waited_streams, [None]
        )

    def test_hybrid_write_moves_indices_without_write_back_jit(self):
        captured = {}

        class FakeHostGroup:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                io_backend,
                pool_transfers=None,
            ):
                captured["host_indices"] = host_indices
                captured["pool_transfers"] = pool_transfers

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4,
                    host_indices=_indices(0, 4),
                    device_indices=_indices(4, 8),
                )
            ],
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = lambda *args: None
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, op.pool_transfers)
        )

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_hybrid_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")
        self.assertEqual(captured["pool_transfers"][0].host_indices.device.type, "cpu")

    def test_write_back_jit_cache_controller_keeps_host_indices_on_cpu(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [
            ManagerCacheOperation(
                host_indices=_indices(0, 4),
                device_indices=_indices(4, 8),
                node_id=1,
            )
        ]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            side_effect=AssertionError(
                "write-back JIT kernel write should not move indices"
            )
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_indices.assert_not_called()
        self.assertEqual(captured["host_indices"].device.type, "cpu")
        self.assertEqual(
            controller.ack_write_queue[0].finish_event.waited_streams, [None]
        )

    def test_cache_controller_moves_indices_without_write_back_jit(self):
        captured = {}

        class FakeHostPool:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ):
                captured["host_indices"] = host_indices

        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostPool()
        controller.mem_pool_device = None
        controller.has_draft = False
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        controller.move_indices.assert_called_once()
        self.assertEqual(captured["host_indices"].device.type, "cpu")

    def test_mla_dedup_peer_still_writes_local_draft_pool(self):
        target_writes = []
        draft_writes = []

        class FakeTargetHostPool:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(self, *args):
                target_writes.append(args)

        class FakeDraftHostPool:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(self, *args):
                draft_writes.append(args)

        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.mla_broadcaster = SimpleNamespace(is_src=False)
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        self.assertEqual(target_writes, [])
        self.assertEqual(len(draft_writes), 1)
        self.assertEqual(draft_writes[0][1].device.type, "cpu")

    def test_mla_dedup_load_restores_draft_on_every_rank(self):
        draft_loads = []
        broadcasts = []

        class FakeDraftHostPool:
            layer_num = 2

            def load_to_device_per_layer(self, *args):
                draft_loads.append(args)

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def __init__(self):
                self.completed_layers = []

            def complete(self, layer_index):
                self.completed_layers.append(layer_index)

        producer_event = FakeProducerEvent()
        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.layer_num = 3
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[producer_event]
        )
        controller.mla_broadcaster = SimpleNamespace(
            is_src=False,
            prepare_broadcast=lambda device_indices, stream: (
                device_indices,
                None,
            ),
            broadcast_loaded_layer=lambda layer_id, prepared: broadcasts.append(
                (layer_id, prepared)
            ),
        )
        controller.load_stream = object()
        controller.ack_load_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_loading()

        self.assertEqual([layer_id for layer_id, _ in broadcasts], [0, 1, 2])
        self.assertEqual(len(draft_loads), 2)
        self.assertEqual(producer_event.completed_layers, [0, 1, 2])
        self.assertEqual(len(controller.ack_load_queue), 1)
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.num_tokens, 4)
        self.assertIsNot(ack.start_event, producer_event.start_event)
        self.assertIsNot(ack.finish_event, producer_event.finish_event)

    def test_mla_dedup_source_load_and_broadcast_are_layerwise(self):
        operations = []

        class FakeTargetHostPool:
            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                operations.append(("target", layer_id))

        class FakeDraftHostPool:
            layer_num = 2

            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                operations.append(("draft", layer_id))

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def complete(self, layer_index):
                operations.append(("complete", layer_index))

        op = ManagerCacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.layer_num = 3
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[FakeProducerEvent()]
        )
        controller.mla_broadcaster = SimpleNamespace(
            is_src=True,
            prepare_broadcast=lambda device_indices, stream: operations.append(
                ("prepare", None)
            )
            or (device_indices, None),
            broadcast_loaded_layer=lambda layer_id, prepared: operations.append(
                ("broadcast", layer_id)
            ),
        )
        controller.load_stream = object()
        controller.ack_load_queue = []
        controller.move_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices)
        )

        with mock.patch.object(
            manager_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_loading()

        self.assertEqual(
            operations,
            [
                ("prepare", None),
                ("target", 0),
                ("broadcast", 0),
                ("draft", 0),
                ("complete", 0),
                ("target", 1),
                ("broadcast", 1),
                ("draft", 1),
                ("complete", 1),
                ("target", 2),
                ("broadcast", 2),
                ("complete", 2),
            ],
        )

    def test_mla_layer_broadcast_reuses_full_staging_capacity(self):
        broadcaster = MLAHostDedupBroadcaster.__new__(MLAHostDedupBroadcaster)
        broadcaster.is_src = True
        broadcaster.src_global_rank = 0
        broadcaster.group = object()

        layer_buffers = [
            torch.arange(24, dtype=torch.float32).reshape(6, 1, 4),
            torch.arange(24, 48, dtype=torch.float32).reshape(6, 1, 4),
        ]
        target = torch.tensor([0, 2, 5], dtype=torch.int64)
        # Capacity is six rows, modeling the retained 2 layers * 3-row chunk.
        staging = torch.empty(2 * 3 * 4, dtype=torch.float32)

        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            broadcaster._bcast_layer(layer_buffers, staging, target, 4, layer_id=1)

        broadcast.assert_called_once()
        expected = layer_buffers[1].index_select(0, target)
        torch.testing.assert_close(
            staging[: expected.numel()].view_as(expected), expected
        )

        broadcaster.is_src = False
        received = [torch.zeros_like(layer) for layer in layer_buffers]
        with mock.patch.object(torch.distributed, "broadcast"):
            broadcaster._bcast_layer(received, staging, target, 4, layer_id=1)
        torch.testing.assert_close(received[1].index_select(0, target), expected)

    def test_hybrid_mla_dedup_peer_still_writes_local_draft_pool(self):
        target_writes = []
        draft_writes = []

        class FakeTargetHostPool:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(self, *args, **kwargs):
                target_writes.append((args, kwargs))

        class FakeDraftHostPool:
            layout = "page_first"
            can_use_write_back_jit = True

            def backup_from_device_all_layer(self, *args):
                draft_writes.append(args)

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.mem_pool_host_draft = FakeDraftHostPool()
        controller.mem_pool_device_draft = object()
        controller.has_draft = True
        controller.mla_broadcaster = SimpleNamespace(is_src=False)
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._record_transfer_indices_on_stream = lambda *args: None

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        self.assertEqual(target_writes, [])
        self.assertEqual(len(draft_writes), 1)
        self.assertEqual(draft_writes[0][1].device.type, "cpu")

    def test_hybrid_mla_dedup_loads_extra_pools_layerwise(self):
        operations = []
        pool_transfers = [mock.sentinel.pool_transfer]

        class FakeTargetHostPool:
            def load_to_device_per_layer(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                pool_transfers=None,
            ):
                operations.append(("load", layer_id, pool_transfers))

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def complete(self, layer_index):
                operations.append(("complete", layer_index))

        op = CacheOperation(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeTargetHostPool()
        controller.mem_pool_device = object()
        controller.has_draft = False
        controller.layer_num = 2
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[FakeProducerEvent()]
        )
        controller.mla_broadcaster = SimpleNamespace(
            is_src=True,
            prepare_broadcast=lambda device_indices, stream: (
                device_indices,
                None,
            ),
            broadcast_loaded_layer=lambda layer_id, prepared: operations.append(
                ("broadcast", layer_id)
            ),
        )
        controller.load_stream = object()
        controller.ack_load_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, pool_transfers)
        )
        controller._record_transfer_indices_on_stream = mock.Mock()

        with (
            mock.patch.object(
                hybrid_cache_controller, "device_module", _FakeDeviceModule
            ),
            mock.patch.object(
                manager_cache_controller, "device_module", _FakeDeviceModule
            ),
        ):
            controller.start_loading()

        self.assertEqual(
            operations,
            [
                ("load", 0, pool_transfers),
                ("broadcast", 0),
                ("complete", 0),
                ("load", 1, pool_transfers),
                ("broadcast", 1),
                ("complete", 1),
            ],
        )
        controller._record_transfer_indices_on_stream.assert_called_once_with(
            controller.load_stream,
            op.host_indices,
            op.device_indices,
            pool_transfers,
        )

    def test_hybrid_mla_dedup_peer_broadcasts_only_target_and_loads_mamba(self):
        operations = []
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeHostGroup:
            anchor_entry = SimpleNamespace(
                layer_mapper=lambda global_id: {1: 0, 3: 1}.get(global_id)
            )

            def load_extra_to_device_per_layer(
                self, layer_id, io_backend, pool_transfers=None, pool_names=None
            ):
                if layer_id in {0, 2}:
                    operations.append(("mamba", layer_id))

        class FakeProducerEvent:
            start_event = _FakeEvent()
            finish_event = _FakeEvent()

            def complete(self, layer_index):
                operations.append(("complete", layer_index))

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.load_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = object()
        controller.has_draft = False
        controller.layer_num = 4
        controller.layer_done_counter = SimpleNamespace(
            update_producer=lambda: 0, events=[FakeProducerEvent()]
        )
        controller.mla_broadcaster = SimpleNamespace(
            is_src=False,
            prepare_broadcast=lambda device_indices, stream: (device_indices, None),
            broadcast_loaded_layer=lambda layer_id, prepared: operations.append(
                ("broadcast", layer_id)
            ),
        )
        controller.load_stream = object()
        controller.ack_load_queue = []
        controller.move_indices = mock.Mock(
            side_effect=lambda host_indices, device_indices: (
                host_indices,
                device_indices,
            )
        )

        with (
            mock.patch.object(
                hybrid_cache_controller, "device_module", _FakeDeviceModule
            ),
            mock.patch.object(
                manager_cache_controller, "device_module", _FakeDeviceModule
            ),
        ):
            controller.start_loading()

        self.assertEqual(
            operations,
            [
                ("mamba", 0),
                ("complete", 0),
                ("broadcast", 0),
                ("complete", 1),
                ("mamba", 2),
                ("complete", 2),
                ("broadcast", 1),
                ("complete", 3),
            ],
        )

    def test_hybrid_mla_dedup_peer_still_writes_rank_local_mamba(self):
        writes = []
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
        )
        op = CacheOperation(
            host_indices=_indices(0, 2),
            device_indices=_indices(2, 4),
            node_id=1,
            pool_transfers=[transfer],
        )

        class FakeHostGroup:
            layout = "page_first"
            can_use_write_back_jit = False

            def backup_from_device_all_layer(self, *args, **kwargs):
                raise AssertionError("peer must not write target MLA")

            def backup_extra_from_device_all_layer(
                self, io_backend, pool_transfers=None, pool_names=None
            ):
                writes.append((io_backend, pool_transfers, pool_names))

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.io_backend = "kernel"
        controller.mem_pool_host = FakeHostGroup()
        controller.mem_pool_device = object()
        controller.has_draft = False
        controller.mla_broadcaster = SimpleNamespace(is_src=False)
        controller.write_stream = object()
        controller.ack_write_queue = []
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, [transfer])
        )
        controller._record_transfer_indices_on_stream = lambda *args: None

        with mock.patch.object(
            hybrid_cache_controller, "device_module", _FakeDeviceModule
        ):
            controller.start_writing()

        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1], [transfer])
        self.assertEqual(writes[0][2], {PoolName.MAMBA})

    def test_mla_dedup_aggregate_host_budget_is_fail_closed(self):
        budget = mock.Mock(get=mock.Mock(return_value=1))
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_HICACHE_HOST_BUDGET_GIB", budget
        ):
            with self.assertRaisesRegex(ValueError, "requires 1.12 GiB"):
                enforce_hicache_host_budget(
                    target_bytes=600_000_000,
                    rank_local_bytes={"mamba": 300_000_000},
                    tp_size=2,
                    context="unit-test",
                )

        budget.get.return_value = 2
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_HICACHE_HOST_BUDGET_GIB", budget
        ):
            self.assertEqual(
                enforce_hicache_host_budget(
                    target_bytes=600_000_000,
                    rank_local_bytes={"mamba": 300_000_000},
                    tp_size=2,
                    context="unit-test",
                ),
                1_200_000_000,
            )

    def test_hybrid_mla_dedup_preflight_includes_draft_before_host_alloc(self):
        params = SimpleNamespace(
            page_size=64,
            hicache_draft_kv_pool=mock.sentinel.draft_pool,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            req_to_token_pool=SimpleNamespace(mamba_allocator=object()),
        )
        server_args = SimpleNamespace(
            enable_mla_hicache_host_dedup=True,
            hicache_ratio=3.0,
            hicache_size=230,
            hicache_mamba_ratio=6.75,
        )

        with (
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_mla_host_pool_bytes",
                return_value=(100, 10),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_mamba_host_pool_bytes",
                return_value=(20, 2),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "estimate_draft_host_pool_bytes",
                return_value=(30, 10),
            ),
            mock.patch.object(
                hybrid_pool_assembler,
                "mla_dedup_rank_and_size",
                return_value=(0, 8),
            ),
            mock.patch.object(
                hybrid_pool_assembler, "enforce_hicache_host_budget"
            ) as enforce,
            mock.patch.object(
                hybrid_pool_assembler,
                "maybe_prebuild_mla_host_dedup",
                side_effect=RuntimeError("stop before allocation"),
            ),
            mock.patch.object(
                hybrid_pool_assembler, "build_kv_host_pool"
            ) as build_host_pool,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before allocation"):
                hybrid_pool_assembler.build_hybrid_mamba_stack(
                    params=params,
                    server_args=server_args,
                    kv_pool=mock.sentinel.kv_pool,
                    mamba_pool=mock.sentinel.mamba_pool,
                    full_layer_mapping={0: 0},
                    mamba_layer_mapping={1: 0},
                    load_cache_event=None,
                    storage_backend=None,
                    use_mla=True,
                )

        enforce.assert_called_once_with(
            target_bytes=100,
            rank_local_bytes={
                "mamba": 20,
                "draft": 30,
                "allocator_metadata": 218,
            },
            tp_size=8,
            context=(
                "hybrid MLA+Mamba L2 "
                "(target_tokens=10, mamba_slots=2, draft_tokens=10)"
            ),
        )
        build_host_pool.assert_not_called()

    def test_mamba_host_sizing_preserves_legacy_fixed_size_without_opt_in(self):
        args = SimpleNamespace(
            hicache_ratio=3.0,
            hicache_size=230,
            hicache_mamba_ratio=None,
            enable_mla_hicache_host_dedup=False,
        )

        self.assertEqual(
            hybrid_pool_assembler._get_mamba_host_sizing(args),
            (3.0, 230),
        )

    def test_mamba_host_sizing_is_independent_for_dedup_or_explicit_ratio(self):
        args = SimpleNamespace(
            hicache_ratio=3.0,
            hicache_size=230,
            hicache_mamba_ratio=None,
            enable_mla_hicache_host_dedup=True,
        )
        self.assertEqual(
            hybrid_pool_assembler._get_mamba_host_sizing(args),
            (3.0, 0),
        )

        args.enable_mla_hicache_host_dedup = False
        args.hicache_mamba_ratio = 6.75
        self.assertEqual(
            hybrid_pool_assembler._get_mamba_host_sizing(args),
            (6.75, 0),
        )

    def test_mla_dedup_requires_dense_stage_local_layer_ids(self):
        hybrid_pool_assembler._require_dense_layer_ids(
            mappings=({0: 0, 2: 1}, {1: 0, 3: 1}),
            transfer_layer_num=4,
            context="unit-test",
        )

        with self.assertRaisesRegex(ValueError, "dense stage-local layer ids"):
            hybrid_pool_assembler._require_dense_layer_ids(
                mappings=({4: 0}, {5: 0}),
                transfer_layer_num=2,
                context="unit-test",
            )

    def test_dedup_draft_requires_the_target_slot_domain(self):
        controller = SimpleNamespace(
            mla_broadcast_enabled=True,
            mem_pool_device=SimpleNamespace(size=1024, page_size=64),
        )
        dflash = SimpleNamespace(is_dflash=lambda: True)
        kv_cache_builder._validate_dedup_draft_index_domain(
            cache_controller=controller,
            draft_pool=SimpleNamespace(size=1024, page_size=64),
            spec_algorithm=dflash,
        )

        with self.assertRaisesRegex(ValueError, "share one global KV slot domain"):
            kv_cache_builder._validate_dedup_draft_index_domain(
                cache_controller=controller,
                draft_pool=SimpleNamespace(size=512, page_size=64),
                spec_algorithm=dflash,
            )

    def test_hicache_metrics_use_stable_per_pool_labels(self):
        class FakePool:
            def __init__(self, total, available):
                self.size = total
                self.logical_size = total
                self._available = available

            def available_size(self):
                return self._available

        kv = FakePool(100, 60)
        mamba = FakePool(8, 3)
        # The draft host pool mirrors the target KV slots: nothing allocates
        # from or frees its own allocator, so it always reports fully free.
        draft = FakePool(100, 100)
        group = SimpleNamespace(
            entries=[
                SimpleNamespace(name=PoolName.KV, host_pool=kv),
                SimpleNamespace(name=PoolName.MAMBA, host_pool=mamba),
            ]
        )
        tree_cache = SimpleNamespace(
            token_to_kv_pool_host=kv,
            host_pool_group=group,
            cache_controller=SimpleNamespace(mem_pool_host_draft=draft),
        )
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.scheduler = SimpleNamespace(
            enable_hierarchical_cache=True, tree_cache=tree_cache
        )
        reporter.stats = SchedulerStats()

        reporter._log_hicache_stats()

        self.assertEqual(
            reporter.stats.hicache_host_pool_total_slots,
            {"kv": 100, "mamba": 8, "draft": 100},
        )
        self.assertEqual(
            reporter.stats.hicache_host_pool_used_slots,
            {"kv": 40, "mamba": 5, "draft": 40},
        )

    def _run_authority_worker_write(self, finish_event, ack_write_queue, pop_ack):
        """Drive one authority-worker write on a fake worker stream, then drain
        the fence on a fake scheduler stream. pop_ack simulates the scheduler's
        _poll_write_acks consuming the ack right after the commit (trivially
        short D2H) — the fence must come from commit_write_with_event, not from
        ack_write_queue[-1]."""

        def commit_write_with_event(reservation):
            ack_write_queue.append(
                SimpleNamespace(finish_event=finish_event, node_ids=(1,))
            )
            if pop_ack:
                ack_write_queue.pop()
            return reservation.host_indices, finish_event

        controller = SimpleNamespace(
            ack_write_queue=ack_write_queue,
            reserve_write=lambda *args, **kwargs: SimpleNamespace(
                host_indices=_indices(0, 4), node_id=1
            ),
            commit_write_with_event=commit_write_with_event,
        )
        authority = HiCacheAuthority(cache=None, controller=controller, is_rank0=True)
        authority.shutdown()
        scheduler_stream = mock.Mock()
        worker_stream = mock.Mock()
        fake_module = SimpleNamespace(current_stream=lambda: worker_stream)
        with mock.patch.object(
            hicache_authority.torch, "get_device_module", return_value=fake_module
        ):
            authority._process_intent(
                WriteIntent(
                    node_id=1,
                    op=hicache_authority.OP_WRITE,
                    kv_device=_indices(0, 4),
                    extra_pools=None,
                    kv_len=4,
                )
            )
            worker_stream.wait_event.assert_not_called()
            fake_module.current_stream = lambda: scheduler_stream
            authority.fence_forward_stream()
            authority.fence_forward_stream()
        scheduler_stream.wait_event.assert_called_once_with(finish_event)
        worker_stream.wait_event.assert_not_called()

    def test_authority_worker_write_is_fenced_on_scheduler_stream(self):
        ack_write_queue = []
        self._run_authority_worker_write(_FakeEvent(), ack_write_queue, pop_ack=False)
        self.assertEqual(len(ack_write_queue), 1)

    def test_authority_worker_fence_survives_immediate_ack_pop(self):
        ack_write_queue = [
            SimpleNamespace(finish_event=_FakeEvent(), node_ids=(7,)),
        ]
        finish_event = _FakeEvent()
        self._run_authority_worker_write(finish_event, ack_write_queue, pop_ack=True)
        # Only the older ack remains (ack_write_queue[-1] would be the wrong
        # event); the fence still targeted the new write's event.
        self.assertEqual(len(ack_write_queue), 1)
        self.assertIsNot(ack_write_queue[-1].finish_event, finish_event)

    def _atomic_handoff_authority(self, finish_event, cache):
        controller = SimpleNamespace(
            ack_write_queue=[],
            reserve_write=lambda *args, **kwargs: SimpleNamespace(
                host_indices=_indices(0, 4), node_id=1
            ),
            commit_write_with_event=lambda reservation: (
                reservation.host_indices,
                finish_event,
            ),
        )
        authority = HiCacheAuthority(cache=cache, controller=controller, is_rank0=True)
        authority.shutdown()
        return authority

    def _atomic_handoff_intent(self):
        return WriteIntent(
            node_id=1,
            op=hicache_authority.OP_WRITE,
            kv_device=_indices(0, 4),
            extra_pools=None,
            kv_len=4,
        )

    def test_atomic_write_handoff_off_is_noop(self):
        with envs.SGLANG_ENABLE_HICACHE_ATOMIC_WRITE_HANDOFF.override(False):
            authority = self._atomic_handoff_authority(_FakeEvent(), cache=mock.Mock())
        self.assertFalse(authority.atomic_write_handoff)
        stream = mock.Mock()
        fake_module = SimpleNamespace(current_stream=lambda: stream)
        with mock.patch.object(
            hicache_authority.torch, "get_device_module", return_value=fake_module
        ):
            authority._process_intent(self._atomic_handoff_intent())
            with authority.forward_launch_section():
                self.assertFalse(authority._launch_lock.locked())
                # Drain still happens per launch (outside the section) when off.
                self.assertEqual(len(authority._forward_fences), 1)
        authority.cache.fence_state_read.assert_not_called()

    def test_atomic_write_handoff_serializes_commit_against_launch(self):
        finish_event = _FakeEvent()
        cache = mock.Mock()
        with envs.SGLANG_ENABLE_HICACHE_ATOMIC_WRITE_HANDOFF.override(True):
            authority = self._atomic_handoff_authority(finish_event, cache=cache)
        self.assertTrue(authority.atomic_write_handoff)
        scheduler_stream = mock.Mock()
        fake_module = SimpleNamespace(current_stream=lambda: scheduler_stream)
        committed = threading.Event()

        def worker():
            authority._process_intent(self._atomic_handoff_intent())
            committed.set()

        with mock.patch.object(
            hicache_authority.torch, "get_device_module", return_value=fake_module
        ):
            with authority.forward_launch_section():
                # Scheduler holds the launch lock: the worker's commit must wait.
                thread = threading.Thread(target=worker)
                thread.start()
                self.assertFalse(committed.wait(0.2))
                self.assertEqual(len(authority._forward_fences), 0)
                cache.fence_state_read.assert_not_called()
            thread.join(timeout=5)
            self.assertTrue(committed.is_set())
            # Worker fenced on the latest forward before committing, and its
            # event is drained at the next launch inside the section.
            cache.fence_state_read.assert_called_once_with()
            self.assertEqual(list(authority._forward_fences), [finish_event])
            with authority.forward_launch_section():
                scheduler_stream.wait_event.assert_called_once_with(finish_event)
                self.assertEqual(len(authority._forward_fences), 0)

    def test_hybrid_controller_commit_write_with_event_returns_finish_event(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = []
        sentinel = object()
        controller.start_writing = lambda: sentinel
        reservation = SimpleNamespace(
            host_indices=_indices(0, 4),
            device_indices=_indices(4, 8),
            node_id=1,
            priority=None,
            pool_reservation=SimpleNamespace(transfers=None),
        )
        host_indices, finish_event = controller.commit_write_with_event(reservation)
        self.assertIs(host_indices, reservation.host_indices)
        self.assertIs(finish_event, sentinel)
        self.assertEqual(len(controller.write_queue), 1)
        self.assertIs(controller.commit_write(reservation), reservation.host_indices)


if __name__ == "__main__":
    unittest.main()
