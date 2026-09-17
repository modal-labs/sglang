"""CPU tests for bounded CUDA IPC lease acknowledgements."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_utils, schedule_batch, scheduler
from sglang.srt.managers.mm_utils import PerImageRequestInfo
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.models import kimi_k3, kimi_k25
from sglang.srt.multimodal import mm_utils as multimodal_mm_utils
from sglang.srt.multimodal.transport import cuda_ipc, lease_guard
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _proxy(*, generation=1, total_consumer_count=1, ready_byte_offset=8):
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.proxy_state = {"ipc_extra": {"pool_handle": ("pool",)}}
    proxy.generation = generation
    proxy.total_consumer_count = total_consumer_count
    proxy.ready_byte_offset = ready_byte_offset
    proxy.ack_byte_offset = ready_byte_offset + 16
    proxy.transport_name = "CUDA IPC"
    proxy._consumer_acknowledged = False
    return proxy


def test_stale_read_is_refused_and_counted():
    lease_guard.reset()
    proxy = _proxy(generation=2)
    assert lease_guard.check_and_record_write(proxy, rank=3)

    with pytest.raises(
        lease_guard.StaleLeaseError,
        match=r"slot 1 generation 2.*max acked 2.*rank 3",
    ):
        lease_guard.check_read(proxy, rank=3)

    assert lease_guard.stats["stale_read_refused"] == 1


def test_stale_write_skips_stream_write_and_logs(caplog):
    lease_guard.reset()
    proxy = _proxy(generation=2)
    assert lease_guard.check_and_record_write(proxy, rank=0)

    with (
        caplog.at_level("DEBUG", logger="sglang.srt.multimodal.transport.lease_guard"),
        patch.object(cuda_ipc, "stream_write_value32") as stream_write,
    ):
        proxy._acknowledge_on_stream(100, 0, 1, 0)

    stream_write.assert_not_called()
    assert proxy._consumer_acknowledged
    assert lease_guard.stats["stale_write_refused"] == 1
    assert "Refused stale CUDA IPC lease write" in caplog.text


def test_generation_guard_is_monotonic_per_slot():
    lease_guard.reset()
    generation_two = _proxy(generation=2)
    generation_one = _proxy(generation=1)
    generation_three = _proxy(generation=3)

    assert lease_guard.check_and_record_write(generation_two, rank=0)
    assert not lease_guard.check_and_record_write(generation_one, rank=0)
    assert lease_guard.check_and_record_write(generation_three, rank=0)
    assert lease_guard.stats["stale_write_refused"] == 1


def test_full_group_ack_records_the_calling_rank():
    lease_guard.reset()
    proxy = _proxy(generation=4, total_consumer_count=2)

    with (
        patch.object(cuda_ipc, "stream_write_value32") as stream_write,
        patch.object(
            lease_guard,
            "check_write",
            wraps=lease_guard.check_write,
        ) as check_write,
    ):
        proxy._acknowledge_on_stream(100, 0, 2, consumer_rank=1)

    assert stream_write.call_count == 2
    check_write.assert_called_once_with(proxy, rank=1)
    assert lease_guard._max_acked_gen[(("pool",), proxy.ready_byte_offset)] == 4


def test_stale_full_group_ack_skips_all_stream_writes():
    lease_guard.reset()
    proxy = _proxy(generation=4, total_consumer_count=2)
    assert lease_guard.check_and_record_write(proxy, rank=1)

    with patch.object(cuda_ipc, "stream_write_value32") as stream_write:
        proxy._acknowledge_on_stream(100, 0, 2, consumer_rank=1)

    stream_write.assert_not_called()
    assert proxy._consumer_acknowledged
    assert lease_guard.stats["stale_write_refused"] == 1


def test_ack_guard_reservation_is_transactional():
    lease_guard.reset()
    proxy = _proxy(generation=4)
    key = lease_guard._key(proxy)

    with (
        patch.object(
            cuda_ipc,
            "stream_write_value32",
            side_effect=[RuntimeError("stream write failed"), None],
        ) as stream_write,
        pytest.raises(RuntimeError, match="stream write failed"),
    ):
        proxy._acknowledge_on_stream(100, 0, 1, consumer_rank=0)

    assert not proxy._consumer_acknowledged
    assert key not in lease_guard._max_acked_gen

    with patch.object(cuda_ipc, "stream_write_value32") as stream_write:
        proxy._acknowledge_on_stream(100, 0, 1, consumer_rank=0)
    assert proxy._consumer_acknowledged
    stream_write.assert_called_once()
    assert lease_guard._max_acked_gen[key] == proxy.generation


def test_deferred_cache_hits_ack_each_rank_only_with_flag():
    items = [Mock(), Mock()]
    parallel = SimpleNamespace(attn_tp_rank=3, attn_tp_size=8)

    with (
        patch.object(mm_utils, "get_parallel", return_value=parallel),
        patch.object(mm_utils.torch.cuda, "current_device", return_value=0),
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
    ):
        mm_utils._acknowledge_deferred_cuda_ipc_cache_hits(items)

    for item in items:
        item.materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)

    for item in items:
        item.reset_mock()
    with (
        patch.object(mm_utils, "get_parallel", return_value=parallel),
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False),
    ):
        mm_utils._acknowledge_deferred_cuda_ipc_cache_hits(items)

    for item in items:
        item.acknowledge_deferred_cuda_ipc_feature.assert_not_called()


def test_k3_dp_non_owner_acknowledges_deferred_proxy():
    item_owner = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.ones(1, 2),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 1]])},
    )
    item_non_owner = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=_proxy(),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 1]])},
    )
    item_owner.reconstruct = Mock()
    item_non_owner.materialize_deferred_cuda_ipc_feature = Mock()

    model = object.__new__(kimi_k3.KimiK3ForConditionalGeneration)
    model.use_data_parallel = True
    vision_device = SimpleNamespace(type="cuda", index=0)
    model.vision_tower = SimpleNamespace(
        device=vision_device,
        patch_embed=SimpleNamespace(
            proj=SimpleNamespace(weight=SimpleNamespace(dtype=torch.float32))
        ),
    )
    model.mm_projector = lambda value: value

    def run_dp(_vision, _pixel_values, _grid_thw_list, **kwargs):
        local_features = kwargs["load_local_pixel_values"]([0])
        assert local_features.shape == (1, 2)
        return torch.ones(1, 2)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            multimodal_mm_utils,
            "run_dp_sharded_mrope_vision_model",
            side_effect=run_dp,
        ),
        patch.object(
            kimi_k3,
            "materialize_multimodal_features",
            return_value=torch.ones(1, 2),
        ),
        patch.object(
            kimi_k3, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
        ),
    ):
        model.get_image_feature([item_owner, item_non_owner])

    item_owner.reconstruct.assert_called_once_with(0, ipc_consumer_count=1)
    item_non_owner.materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)


def test_k25_dp_non_owner_acknowledges_deferred_proxy():
    item_owner = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.ones(1, 2),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 1]])},
    )
    item_non_owner = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=_proxy(),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 1, 1]])},
    )
    item_owner.reconstruct = Mock()
    item_non_owner.materialize_deferred_cuda_ipc_feature = Mock()

    model = object.__new__(kimi_k25.KimiK25ForConditionalGeneration)
    model.use_data_parallel = True
    vision_device = SimpleNamespace(type="cuda", index=0)
    model.vision_tower = SimpleNamespace(
        device=vision_device,
        patch_embed=SimpleNamespace(
            proj=SimpleNamespace(weight=SimpleNamespace(dtype=torch.float32))
        ),
    )
    model.mm_projector = lambda value: value

    def run_dp(_vision, _pixel_values, _grid_thw_list, **kwargs):
        local_features = kwargs["load_local_pixel_values"]([0])
        assert local_features.shape == (1, 2)
        return torch.ones(1, 2)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            kimi_k25,
            "run_dp_sharded_mrope_vision_model",
            side_effect=run_dp,
        ),
        patch.object(
            kimi_k25,
            "materialize_multimodal_features",
            return_value=torch.ones(1, 2),
        ),
        patch.object(
            kimi_k25, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
        ),
    ):
        model.get_image_feature([item_owner, item_non_owner])

    item_owner.reconstruct.assert_called_once_with(0, ipc_consumer_count=1)
    item_non_owner.materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)


def test_broadcast_materialization_passes_group_consumer_count():
    proxy = _proxy(total_consumer_count=4)
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
    item.reconstruct = Mock()
    item.set_pad_value = Mock()
    output = MultimodalProcessorOutput(input_ids=[1], mm_items=[item])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch,
            "CudaIpcTensorTransportProxy",
            CudaIpcTensorTransportProxy,
        ),
        patch.object(torch.cuda, "current_device", return_value=0),
    ):
        schedule_batch.MultimodalInputs.from_processor_output(
            output,
            ipc_consumer_count=schedule_batch._full_consumer_count(output.mm_items),
        )

    item.reconstruct.assert_called_once_with(0, ipc_consumer_count=4)


def _prefix_item(offsets):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=_proxy(),
    )
    item.materialize_deferred_cuda_ipc_feature = Mock()
    return item


def test_prefix_resident_ack_only_covers_wholly_cached_items():
    items = [
        _prefix_item([(0, 9)]),
        _prefix_item([(5, 15)]),
        _prefix_item([(20, 29)]),
    ]
    mm_inputs = schedule_batch.MultimodalInputs(mm_items=items)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch,
            "CudaIpcTensorTransportProxy",
            CudaIpcTensorTransportProxy,
        ),
        patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
    ):
        assert mm_inputs.acknowledge_prefix_resident_items(15) == 1

    items[0].materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)
    items[1].materialize_deferred_cuda_ipc_feature.assert_not_called()
    items[2].materialize_deferred_cuda_ipc_feature.assert_not_called()


def test_prefix_resident_ack_requires_all_offsets_inside_prefix():
    inside = _prefix_item([(0, 3), (5, 9)])
    crossing = _prefix_item([(0, 3), (5, 15)])
    mm_inputs = schedule_batch.MultimodalInputs(mm_items=[inside, crossing])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch,
            "CudaIpcTensorTransportProxy",
            CudaIpcTensorTransportProxy,
        ),
        patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
    ):
        assert mm_inputs.acknowledge_prefix_resident_items(15) == 1

    inside.materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)
    crossing.materialize_deferred_cuda_ipc_feature.assert_not_called()


def test_prefix_resident_ack_request_guard_is_idempotent():
    mm_inputs = Mock()
    req = SimpleNamespace(
        multimodal_inputs=mm_inputs,
        mm_prefix_ack_done=False,
        prefix_indices=torch.empty(7),
    )

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        schedule_batch._acknowledge_prefix_resident_requests([req])
        schedule_batch._acknowledge_prefix_resident_requests([req])

    mm_inputs.acknowledge_prefix_resident_items.assert_called_once_with(7)
    assert req.mm_prefix_ack_done


def test_per_image_cache_hits_and_duplicate_hashes_ack_before_encoding():
    mm_utils.init_mm_embedding_cache(max_size=1024)
    item_a = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=1,
        offsets=[(0, 9)],
        feature=_proxy(),
    )
    item_b = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=2,
        offsets=[(10, 19)],
        feature=_proxy(),
    )
    item_c = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=2,
        offsets=[(20, 29)],
        feature=_proxy(),
    )
    for item in (item_a, item_b, item_c):
        item.materialize_deferred_cuda_ipc_feature = Mock()
    mm_utils.embedding_cache.set(
        1, mm_utils.EmbeddingResult(embedding=torch.ones(10, 2))
    )
    requests = [
        PerImageRequestInfo(
            req_idx=0,
            items=[item_a, item_b],
            items_offset=[(0, 9), (10, 19)],
            extend_prefix_len=0,
            extend_seq_len=20,
        ),
        PerImageRequestInfo(
            req_idx=1,
            items=[item_c],
            items_offset=[(20, 29)],
            extend_prefix_len=0,
            extend_seq_len=30,
        ),
    ]

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            mm_utils,
            "CudaIpcTensorTransportProxy",
            CudaIpcTensorTransportProxy,
        ),
        patch.object(mm_utils.torch.cuda, "current_device", return_value=0),
    ):
        mm_utils._batch_encode_per_image_misses(
            lambda items: torch.ones(10, 2), requests, torch.device("cpu")
        )

    item_a.materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)
    item_b.materialize_deferred_cuda_ipc_feature.assert_not_called()
    item_c.materialize_deferred_cuda_ipc_feature.assert_called_once_with(0)

    for item in (item_a, item_b, item_c):
        item.materialize_deferred_cuda_ipc_feature.reset_mock()
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False):
        mm_utils._batch_encode_per_image_misses(
            lambda items: torch.ones(10, 2), requests, torch.device("cpu")
        )
    for item in (item_a, item_b, item_c):
        item.materialize_deferred_cuda_ipc_feature.assert_not_called()


def test_scheduler_rejection_releases_raw_mm_proxy():
    proxy = _proxy()
    proxy.release_without_reconstruction = Mock()
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
    raw_mm_inputs = SimpleNamespace(mm_items=[item])
    recv_req = SimpleNamespace(
        session_params=SimpleNamespace(id="missing-session"),
        session_id=None,
        mm_inputs=raw_mm_inputs,
        rid="reject",
        input_text="",
        input_ids=[1],
        sampling_params=SimpleNamespace(),
        http_worker_ipc=None,
    )
    req = SimpleNamespace(
        tokenizer=None,
        set_finish_with_abort=Mock(),
    )
    instance = object.__new__(scheduler.Scheduler)
    instance.server_args = SimpleNamespace(enable_session_radix_cache=False)
    instance.session_controller = {}
    instance.model_config = SimpleNamespace(vocab_size=8)
    instance.tokenizer = None
    instance.init_req_max_new_tokens = Mock()
    instance._add_request_to_queue = Mock()

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch,
            "CudaIpcTensorTransportProxy",
            CudaIpcTensorTransportProxy,
        ),
        patch.object(scheduler, "Req", return_value=req),
    ):
        instance.handle_generate_request(recv_req)

    proxy.release_without_reconstruction.assert_called_once_with(1)


def test_mm_attach_failure_releases_features():
    """An exception while attaching mm inputs (pad/mrope path) must release the
    image_inputs fetched by _get_multimodal_inputs instead of orphaning them."""
    image_inputs = MultimodalInputs(mm_items=[])
    image_inputs.release_features = Mock()
    recv_req = Mock()
    recv_req.session_params = None
    recv_req.session_id = None
    recv_req.input_embeds = None
    recv_req.bootstrap_port = 8998
    recv_req.input_ids = [1]
    recv_req.mm_inputs = SimpleNamespace(mm_items=[])
    req = SimpleNamespace(
        return_sampling_mask=False,
        tokenizer=None,
    )
    instance = object.__new__(scheduler.Scheduler)
    instance.server_args = SimpleNamespace(
        enable_session_radix_cache=False,
        disaggregation_bootstrap_port=8998,
    )
    instance.session_controller = {}
    instance.model_config = SimpleNamespace(vocab_size=8, hf_eos_token_id=None)
    instance.tokenizer = None
    instance.metrics_reporter = SimpleNamespace(enable_metrics=False)
    instance.dllm_config = None
    instance.disaggregation_mode = scheduler.DisaggregationMode.NULL
    instance.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: False,
        is_none=lambda: True,
    )
    instance._maybe_namespace_elastic_radix_cache = Mock()
    instance._get_multimodal_inputs = Mock(return_value=image_inputs)
    instance._try_apply_padded_mm_input_ids = Mock(
        side_effect=RuntimeError("pad failed")
    )

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(scheduler, "Req", return_value=req),
        patch.object(scheduler.SessionController, "adjust_mm_offsets"),
        pytest.raises(RuntimeError, match="pad failed"),
    ):
        instance.handle_generate_request(recv_req)

    image_inputs.release_features.assert_called_once_with()


@pytest.mark.parametrize(
    ("lease_enabled", "release_count"),
    [(True, 1), (False, 0)],
)
def test_mm_mrope_failure_releases_features_only_with_flag(
    lease_enabled, release_count
):
    image_inputs = MultimodalInputs(mm_items=[])
    image_inputs.release_features = Mock()
    recv_req = Mock()
    recv_req.session_params = None
    recv_req.session_id = None
    recv_req.input_embeds = None
    recv_req.bootstrap_port = 8998
    recv_req.input_ids = [1]
    recv_req.mm_inputs = SimpleNamespace(mm_items=[])
    req = SimpleNamespace(
        return_sampling_mask=False,
        tokenizer=None,
        origin_input_ids=[1],
        extend_image_inputs=Mock(),
    )
    instance = object.__new__(scheduler.Scheduler)
    instance.server_args = SimpleNamespace(
        enable_session_radix_cache=False,
        disaggregation_bootstrap_port=8998,
    )
    instance.session_controller = {}
    instance.model_config = SimpleNamespace(vocab_size=8, hf_eos_token_id=None)
    instance.tokenizer = None
    instance.metrics_reporter = SimpleNamespace(enable_metrics=False)
    instance.dllm_config = None
    instance.disaggregation_mode = scheduler.DisaggregationMode.NULL
    instance.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: False,
        is_none=lambda: True,
    )
    instance.pad_input_ids_func = None
    instance._maybe_namespace_elastic_radix_cache = Mock()
    instance._get_multimodal_inputs = Mock(return_value=image_inputs)
    instance._try_apply_padded_mm_input_ids = Mock(return_value=True)
    instance._maybe_compute_mrope_positions = Mock(
        side_effect=RuntimeError("mrope failed")
    )

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(lease_enabled),
        patch.object(scheduler, "Req", return_value=req),
        pytest.raises(RuntimeError, match="mrope failed"),
    ):
        instance.handle_generate_request(recv_req)

    assert image_inputs.release_features.call_count == release_count


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
