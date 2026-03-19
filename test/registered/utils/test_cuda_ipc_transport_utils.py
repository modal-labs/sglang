import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.async_mm_data_processor import AsyncMMDataProcessor
from sglang.srt.managers.scheduler import DisaggregationMode, run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors import base_processor as base_processor_module
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils import cuda_ipc_transport_utils as cuda_ipc_module
from sglang.srt.utils.cuda_ipc_transport_utils import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    _normalize_pool_cache_key,
    _pool_handle_cache_clear,
    _pool_handle_cache_set,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="stage-b-test-small-1-gpu")


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[7]], dtype=torch.int64))


class _DummyProcessorWrapper:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()


class _DummyMMProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        return {}


class _FakeScheduler:
    def __init__(self, *args, **kwargs):
        self.max_total_num_tokens = 1
        self.max_req_input_len = 1
        self.enable_pdmux = False
        self.enable_overlap = False
        self.disaggregation_mode = DisaggregationMode.NULL

    def event_loop_normal(self):
        return


class TestCudaIpcTransportUtils(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for CUDA IPC transport tests.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")
        _pool_handle_cache_clear()
        self._pools = []

    def tearDown(self):
        _pool_handle_cache_clear()
        for pool in self._pools:
            pool.shutdown()

    def _make_pool(self, memory_size=4096):
        pool = MmItemMemoryPool(memory_size=memory_size, recycle_interval=3600.0)
        self._pools.append(pool)
        return pool

    def _make_tensor(self, start: int, shape=(4, 4), dtype=torch.float16):
        numel = 1
        for dim in shape:
            numel *= dim
        return (
            torch.arange(start, start + numel, device="cuda", dtype=dtype)
            .reshape(shape)
            .contiguous()
        )

    def _make_proxy(self, pool: MmItemMemoryPool, tensor: torch.Tensor):
        sync_flag, available_slice, byte_offset = pool.return_a_slice_tensor_with_flag(
            tensor
        )
        self.assertIsInstance(available_slice, torch.Tensor)
        available_slice.copy_(tensor.view(torch.int8).view(-1), non_blocking=False)
        return CudaIpcTensorTransportProxy(
            data=available_slice,
            info_data=tensor,
            sync_buffer_meta=sync_flag,
            pool_ipc_handle=pool._pool_ipc_handle,
            pool_byte_offset=byte_offset,
            pool_device_index=pool._pool_device_index,
        )

    def test_pool_cache_reuses_same_handle(self):
        pool = self._make_pool()
        tensor_a = self._make_tensor(0)
        tensor_b = self._make_tensor(16)
        proxy_a = self._make_proxy(pool, tensor_a)
        proxy_b = self._make_proxy(pool, tensor_b)

        open_calls = 0
        original_open = cuda_ipc_module._open_pooled_storage_uncached

        def counted_open(pool_handle):
            nonlocal open_calls
            open_calls += 1
            return original_open(pool_handle)

        with patch.object(
            cuda_ipc_module,
            "_open_pooled_storage_uncached",
            side_effect=counted_open,
        ):
            rebuilt_a = proxy_a.reconstruct_on_target_device(torch.cuda.current_device())
            rebuilt_b = proxy_b.reconstruct_on_target_device(torch.cuda.current_device())

        self.assertTrue(torch.equal(rebuilt_a, tensor_a))
        self.assertTrue(torch.equal(rebuilt_b, tensor_b))
        self.assertEqual(open_calls, 1)

    def test_cached_failure_invalidates_and_retries_uncached(self):
        pool = self._make_pool()
        tensor = self._make_tensor(32)
        proxy = self._make_proxy(pool, tensor)
        cache_key = _normalize_pool_cache_key(
            pool._pool_ipc_handle, pool._pool_device_index
        )
        bad_storage = object()
        _pool_handle_cache_set(cache_key, bad_storage)

        open_calls = 0
        original_open = cuda_ipc_module._open_pooled_storage_uncached

        def counted_open(pool_handle):
            nonlocal open_calls
            open_calls += 1
            return original_open(pool_handle)

        with patch.object(
            cuda_ipc_module,
            "_open_pooled_storage_uncached",
            side_effect=counted_open,
        ):
            rebuilt = proxy.reconstruct_on_target_device(torch.cuda.current_device())

        self.assertTrue(torch.equal(rebuilt, tensor))
        self.assertEqual(open_calls, 1)
        self.assertIsNot(cuda_ipc_module._pool_storage_cache[cache_key], bad_storage)

    def test_falls_back_to_legacy_metadata_when_pooled_retry_fails(self):
        pool = self._make_pool()
        tensor = self._make_tensor(64)
        proxy = self._make_proxy(pool, tensor)
        cache_key = _normalize_pool_cache_key(
            pool._pool_ipc_handle, pool._pool_device_index
        )
        _pool_handle_cache_set(cache_key, object())

        with patch.object(
            cuda_ipc_module,
            "_open_pooled_storage_uncached",
            side_effect=RuntimeError("pooled open failed"),
        ):
            rebuilt = proxy.reconstruct_on_target_device(torch.cuda.current_device())

        self.assertTrue(torch.equal(rebuilt, tensor))

    def test_distinct_pool_handles_do_not_cross_poison(self):
        pool_a = self._make_pool()
        pool_b = self._make_pool()
        tensor_a1 = self._make_tensor(96)
        tensor_a2 = self._make_tensor(112)
        tensor_b1 = self._make_tensor(128)
        tensor_b2 = self._make_tensor(144)
        proxy_b1 = self._make_proxy(pool_b, tensor_b1)
        proxy_b2 = self._make_proxy(pool_b, tensor_b2)
        proxy_a1 = self._make_proxy(pool_a, tensor_a1)
        proxy_a2 = self._make_proxy(pool_a, tensor_a2)

        open_calls = 0
        original_open = cuda_ipc_module._open_pooled_storage_uncached

        def counted_open(pool_handle):
            nonlocal open_calls
            open_calls += 1
            return original_open(pool_handle)

        key_a = _normalize_pool_cache_key(
            pool_a._pool_ipc_handle, pool_a._pool_device_index
        )
        key_b = _normalize_pool_cache_key(
            pool_b._pool_ipc_handle, pool_b._pool_device_index
        )

        with patch.object(
            cuda_ipc_module,
            "_open_pooled_storage_uncached",
            side_effect=counted_open,
        ):
            rebuilt_b1 = proxy_b1.reconstruct_on_target_device(torch.cuda.current_device())
            rebuilt_b2 = proxy_b2.reconstruct_on_target_device(torch.cuda.current_device())
            self.assertEqual(open_calls, 1)
            _pool_handle_cache_set(key_a, object())
            rebuilt_a1 = proxy_a1.reconstruct_on_target_device(torch.cuda.current_device())
            self.assertEqual(open_calls, 2)
            rebuilt_a2 = proxy_a2.reconstruct_on_target_device(torch.cuda.current_device())
            rebuilt_b3 = self._make_proxy(pool_b, tensor_b1).reconstruct_on_target_device(
                torch.cuda.current_device()
            )

        self.assertTrue(torch.equal(rebuilt_a1, tensor_a1))
        self.assertTrue(torch.equal(rebuilt_a2, tensor_a2))
        self.assertTrue(torch.equal(rebuilt_b1, tensor_b1))
        self.assertTrue(torch.equal(rebuilt_b2, tensor_b2))
        self.assertTrue(torch.equal(rebuilt_b3, tensor_b1))
        self.assertEqual(open_calls, 2)
        self.assertIn(key_a, cuda_ipc_module._pool_storage_cache)
        self.assertIn(key_b, cuda_ipc_module._pool_storage_cache)

    def test_base_processor_pool_miss_falls_back_to_cpu_tensor(self):
        cuda_tensor = self._make_tensor(160)
        fake_pool = MagicMock()
        fake_pool.return_a_slice_tensor_with_flag.return_value = (None, None, None)
        fake_pool.shutdown = MagicMock()

        server_args = SimpleNamespace(
            keep_mm_feature_on_device=False,
            disable_fast_image_processor=True,
            skip_tokenizer_init=False,
        )
        hf_config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=1))

        with patch.object(base_processor_module, "SGL_USE_CUDA_IPC", True), patch.object(
            base_processor_module, "MmItemMemoryPool", return_value=fake_pool
        ):
            processor = _DummyMMProcessor(
                hf_config,
                server_args,
                _DummyProcessorWrapper(),
                transport_mode="cuda_ipc",
            )
            base_output = BaseMultiModalProcessorOutput(
                input_text="unused",
                images=[{"format": "processor_output", "pixel_values": cuda_tensor}],
            )
            mm_tokens = MultimodalSpecialTokens(image_token_id=7)
            mm_items, _input_ids, _ret = processor.process_and_combine_mm_data(
                base_output, mm_tokens
            )

        self.assertEqual(len(mm_items), 1)
        self.assertIsInstance(mm_items[0].feature, torch.Tensor)
        self.assertFalse(mm_items[0].feature.is_cuda)
        processor.shutdown()
        fake_pool.shutdown.assert_called_once()

    def test_async_mm_data_processor_shutdown_delegates_to_mm_processor(self):
        mm_processor = MagicMock()
        processor = AsyncMMDataProcessor(mm_processor, max_concurrent_calls=1)
        processor.shutdown()
        mm_processor.shutdown.assert_called_once()

    def test_tokenizer_manager_shutdown_clears_local_pool_cache(self):
        manager = SimpleNamespace(mm_data_processor=MagicMock(), _shutdown=False)
        with patch(
            "sglang.srt.utils.cuda_ipc_transport_utils._pool_handle_cache_clear"
        ) as clear_mock:
            TokenizerManager.shutdown(manager)

        manager.mm_data_processor.shutdown.assert_called_once()
        clear_mock.assert_called_once()

    def test_run_scheduler_process_clears_cache_on_exit(self):
        server_args = SimpleNamespace(
            pp_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            tp_size=1,
            ep_size=1,
            numa_node=None,
            enable_trace=False,
            remote_instance_weight_loader_use_transfer_engine=lambda: False,
        )
        pipe_writer = MagicMock()
        fake_parent = MagicMock()

        with patch("sglang.srt.managers.scheduler.setproctitle.setproctitle"), patch(
            "sglang.srt.managers.scheduler.faulthandler.enable"
        ), patch("sglang.srt.managers.scheduler.kill_itself_when_parent_died"), patch(
            "sglang.srt.managers.scheduler.psutil.Process"
        ) as process_cls, patch(
            "sglang.srt.managers.scheduler.configure_logger"
        ), patch(
            "sglang.srt.managers.scheduler.suppress_other_loggers"
        ), patch(
            "sglang.srt.managers.scheduler.Scheduler", _FakeScheduler
        ), patch(
            "sglang.srt.utils.cuda_ipc_transport_utils._pool_handle_cache_clear"
        ) as clear_mock:
            process_cls.return_value.parent.return_value = fake_parent
            run_scheduler_process(
                server_args,
                port_args=SimpleNamespace(),
                gpu_id=0,
                tp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                dp_rank=None,
                pipe_writer=pipe_writer,
            )

        pipe_writer.send.assert_called_once()
        clear_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
