"""CPU scenario tests for multimodal lease ownership bug classes."""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_utils, schedule_batch
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.multimodal.processors.kimi_k25 import MMFeatureStreamSink
from sglang.srt.multimodal.transport import lease_guard, lease_lifecycle
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.multimodal.transport.lease_guard import StaleLeaseError
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import (
    Session,
    SessionController,
    SessionReqNode,
)
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakePool:
    def __init__(self):
        self._pool_ipc_handle = ("scenario",)
        self.cancelled = []

    def cancel_proxy(self, proxy):
        self.cancelled.append(proxy)


def _proxy():
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.proxy_state = {"ipc_extra": {"pool_handle": ("scenario",)}}
    return proxy


def _lease_proxy(generation=1, ready_byte_offset=0):
    proxy = _proxy()
    proxy.generation = generation
    proxy.ready_byte_offset = ready_byte_offset
    proxy.ack_byte_offset = 4
    proxy.total_consumer_count = 1
    proxy.transport_name = "CUDA IPC"
    proxy._consumer_acknowledged = False
    return proxy


def _item(proxy):
    return MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)


def _session_recv(rid, parent_rid=None):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", [1]),
        session_params=SimpleNamespace(
            id="s",
            rid=parent_rid,
            offset=None,
            replace=False,
            drop_previous_output=False,
        ),
        sampling_params=SamplingParams(max_new_tokens=1),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        evict_on_finish=False,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


def test_P44_nonowner_materialize_keeps_item_encodable():
    lease_guard.reset()
    proxy = _lease_proxy()
    ack_count = 0
    wait_count = 0

    def reconstruct(_device_index):
        nonlocal ack_count, wait_count
        ack_count += 1
        wait_count += 1
        return torch.ones(1)

    proxy.reconstruct_on_target_device = Mock(side_effect=reconstruct)
    item = _item(proxy)

    with patch.object(
        schedule_batch, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
    ):
        item.materialize_deferred_cuda_ipc_feature(0)

    assert isinstance(item.feature, torch.Tensor)
    assert ack_count == 1
    assert wait_count == 1
    item.reconstruct(0)
    assert ack_count == 1
    assert wait_count == 1

    legacy_proxy = _lease_proxy()
    legacy_item = _item(legacy_proxy)
    legacy_proxy.acknowledge_consumption = Mock(
        side_effect=lambda _count: lease_guard.record_write(legacy_proxy, rank=0)
    )
    legacy_proxy.reconstruct_on_target_device = (
        lambda _device_index: lease_guard.check_read(legacy_proxy, rank=0)
    )

    with patch.object(
        schedule_batch, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
    ):
        legacy_item.acknowledge_deferred_cuda_ipc_feature(1)
        with pytest.raises(StaleLeaseError):
            legacy_item.reconstruct(0)
    legacy_proxy.acknowledge_consumption.assert_called_once_with(1)
    lease_guard.reset()


def test_P44_cache_hit_materializes_under_lease_flag():
    proxy = _lease_proxy()
    proxy.reconstruct_on_target_device = Mock(return_value=torch.ones(1))
    item = _item(proxy)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
        ),
        patch(
            "sglang.srt.managers.mm_utils.torch.cuda.current_device",
            return_value=0,
        ),
    ):
        mm_utils._acknowledge_deferred_cuda_ipc_cache_hits([item])

    assert isinstance(item.feature, torch.Tensor)
    proxy = _lease_proxy()
    item = _item(proxy)
    item.acknowledge_deferred_cuda_ipc_feature = Mock()

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False),
        patch.object(
            mm_utils,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_rank=0, attn_tp_size=2),
        ),
        patch.object(
            mm_utils, "get_server_args", return_value=SimpleNamespace(tp_size=8)
        ),
    ):
        mm_utils._acknowledge_deferred_cuda_ipc_cache_hits([item])

    item.acknowledge_deferred_cuda_ipc_feature.assert_called_once_with(8)
    assert item.feature is proxy


def test_P44_stale_acknowledge_enqueues_no_wait():
    lease_guard.reset()
    recorded = _lease_proxy(generation=7)
    lease_guard.record_write(recorded, rank=0)
    fresh = _lease_proxy(generation=7)

    with (
        patch(
            "sglang.srt.multimodal.transport.memory_pool.stream_wait_value32"
        ) as wait_ready,
        patch(
            "sglang.srt.multimodal.transport.memory_pool.stream_write_value32"
        ) as write_ack_memory_pool,
        patch(
            "sglang.srt.multimodal.transport.cuda_ipc.stream_write_value32"
        ) as write_ack,
    ):
        fresh.acknowledge_consumption(1)

    assert fresh._consumer_acknowledged is True
    wait_ready.assert_not_called()
    write_ack_memory_pool.assert_not_called()
    write_ack.assert_not_called()
    lease_guard.reset()


def test_P44_prefix_resident_materializes_then_reencodes():
    lease_guard.reset()
    proxy = _lease_proxy()
    proxy.reconstruct_on_target_device = Mock(return_value=torch.ones(1))
    item = _item(proxy)
    item.offsets = [(0, 3)]
    mm_inputs = MultimodalInputs(mm_items=[item])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
        ),
        patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
    ):
        assert mm_inputs.acknowledge_prefix_resident_items(4) == 1

    assert isinstance(item.feature, torch.Tensor)
    item.reconstruct(0)
    proxy = _lease_proxy()
    legacy_item = _item(proxy)
    legacy_item.offsets = [(0, 3)]
    proxy.acknowledge_consumption = Mock(
        side_effect=lambda _count: lease_guard.record_write(proxy, rank=0)
    )
    proxy.reconstruct_on_target_device = lambda _device_index: lease_guard.check_read(
        proxy, rank=0
    )
    with patch.object(
        schedule_batch, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
    ):
        legacy_item.acknowledge_deferred_cuda_ipc_feature(1)
        with pytest.raises(StaleLeaseError):
            legacy_item.reconstruct(0)
    lease_guard.reset()


def test_WQH0_waiting_timeout_abort_releases_once():
    proxy = _proxy()
    item = _item(proxy)
    item.release_transport_proxies = Mock()
    req = Req("rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.multimodal_inputs = MultimodalInputs(mm_items=[item])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req.set_finish_with_abort("timeout")
        req.set_finish_with_abort("timeout again")

    item.release_transport_proxies.assert_called_once_with()


def test_WQHI_session_abort_close_releases_only_on_session_close():
    mm_inputs = Mock()
    mm_inputs.session_live_refs = 0
    req = Req("rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.multimodal_inputs = mm_inputs
    req.session = Session(32, "session")

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req.set_finish_with_abort("session abort")
    mm_inputs.release_features.assert_called_once_with()
    mm_inputs.release_features.reset_mock()

    close_req = Req("close-rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    close_req.multimodal_inputs = mm_inputs
    close_req.finished_reason = object()
    session = Session(32, "session")
    session.req_nodes["close-rid"] = SessionReqNode(close_req)
    controller = object.__new__(SessionController)
    controller.sessions = {"session": session}
    controller.tree_cache = Mock()
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        controller._close("session")
    mm_inputs.release_features.assert_called_once_with()


def test_W78M_clone_detachment_keeps_single_original_lease():
    pool = FakePool()
    proxies = [_proxy(), _proxy(), _proxy()]
    mm_inputs = MultimodalInputs(mm_items=[_item(proxy) for proxy in proxies])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            lease_lifecycle,
            "copy_lease_to_cpu",
            side_effect=[torch.ones(1), torch.ones(1), torch.ones(1)],
        ),
    ):
        clone = lease_lifecycle.detach_proxies_for_clones(pool, mm_inputs)
        assert all(isinstance(item.feature, torch.Tensor) for item in clone.mm_items)
        assert [item.feature for item in mm_inputs.mm_items] == proxies
        lease_lifecycle.cancel_undispatched_proxies(
            pool, mm_inputs.mm_items, context="scenario"
        )

    assert pool.cancelled == proxies


def test_I3_mid_processing_raise_cancels_all_sink_proxies():
    pool = FakePool()
    proxies = [_proxy(), _proxy()]
    processor = SimpleNamespace(
        use_cuda_ipc=True,
        cudaipc_mmfeature_pool=pool,
        _wrap_tensor_for_cuda_ipc=Mock(
            side_effect=[proxies[0], proxies[1], RuntimeError("wrap failed")]
        ),
    )
    sink = MMFeatureStreamSink(processor)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch("sglang.srt.managers.mm_utils.hash_feature", return_value=1),
        patch(
            "sglang.srt.multimodal.transport.cuda_ipc.CudaIpcTensorTransportProxy",
            type(proxies[0]),
        ),
    ):
        sink(0, torch.ones(1))
        sink(1, torch.ones(1))
        with pytest.raises(RuntimeError, match="wrap failed"):
            sink(2, torch.ones(1))
        sink.cancel_all("scenario")

    assert pool.cancelled == proxies


def test_X0B0_interleaved_ack_recycles_once():
    from test_lease_pool_invariants import MockLeasePool

    pool = MockLeasePool(ranks=2, slots=1)
    proxy = pool.lease(8)
    pool.publish(proxy)
    for rank in (0, 1, 1, 0):
        pool.ack(proxy, rank)
        pool.recycle(0)
    assert pool.slots[0].state == "FREE"
    assert pool.recycle_count == 1


def test_X0CF_orphaned_owner_cleanup_paths():
    pool = FakePool()
    proxies = [_proxy(), _proxy()]
    items = [_item(proxy) for proxy in proxies]
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        assert (
            lease_lifecycle.cancel_undispatched_proxies(pool, items, context="orphan")
            == 2
        )
    assert pool.cancelled == proxies

    raw_proxies = [_proxy(), _proxy()]
    for proxy in raw_proxies:
        proxy.release_without_reconstruction = Mock()
    raw = SimpleNamespace(mm_items=[_item(proxy) for proxy in raw_proxies])
    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch,
            "CudaIpcTensorTransportProxy",
            CudaIpcTensorTransportProxy,
        ),
    ):
        assert schedule_batch.release_raw_mm_inputs(raw) == 2
    for proxy in raw_proxies:
        proxy.release_without_reconstruction.assert_called_once_with(1)


def test_XlIe_XnUU_inflight_generation_and_unpublished_reads():
    from test_lease_pool_invariants import MockLeasePool

    pool = MockLeasePool(ranks=1, slots=1)
    first = pool.lease(8)
    pool.publish(first)
    pool.ack(first, 0)
    pool.recycle(0)
    second = pool.lease(8)
    with pytest.raises(RuntimeError, match="not published"):
        pool.read(second)
    pool.publish(second)
    with pytest.raises(StaleLeaseError):
        pool.read(first)


def _queued_abort_scheduler(waiting_queue):
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler._pending_chunked_abort_req = None
    scheduler.waiting_queue = waiting_queue
    scheduler.enable_hicache_storage = False
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
    scheduler.disaggregation_mode = SimpleNamespace()
    scheduler.dllm_config = None
    scheduler.grammar_manager = Mock()
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.running_batch = None
    scheduler.last_batch = None
    return scheduler


@pytest.mark.parametrize("flag", [True, False])
def test_P44_queued_client_abort_releases_unconsumed_leases(flag):
    """A request aborted while still waiting never reaches prefill, so no rank
    would otherwise ever acknowledge its leases (p44 stress leak)."""
    from sglang.srt.managers.io_struct import AbortReq

    item = _item(_proxy())
    item.release_transport_proxies = Mock()
    req = Req("rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.multimodal_inputs = MultimodalInputs(mm_items=[item])
    other = Req("other", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    other.multimodal_inputs = MultimodalInputs(mm_items=[_item(_proxy())])
    other.multimodal_inputs.mm_items[0].release_transport_proxies = Mock()
    scheduler = _queued_abort_scheduler([other, req])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(flag):
        scheduler.abort_request(AbortReq(rid="rid"))

    assert scheduler.waiting_queue == [other]
    other.multimodal_inputs.mm_items[0].release_transport_proxies.assert_not_called()
    if flag:
        item.release_transport_proxies.assert_called_once_with()
        assert req.multimodal_inputs is None
    else:
        item.release_transport_proxies.assert_not_called()
        assert req.multimodal_inputs is not None


def test_P44_session_first_turn_queued_abort_releases_turn_leases():
    from sglang.srt.managers.io_struct import AbortReq

    session = Session(32, "s", streaming=True)
    session._inflight = True
    session._inflight_rid = "rid"
    item = _item(_proxy())
    item.release_transport_proxies = Mock()
    req = Req("rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.multimodal_inputs = MultimodalInputs(mm_items=[item])
    scheduler = _queued_abort_scheduler([req])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="rid"))

    item.release_transport_proxies.assert_called_once_with()
    assert req.multimodal_inputs is None
    assert session._inflight is False
    assert session._inflight_rid is None
    assert session.req_nodes == {}


def test_P44_session_first_turn_queued_abort_flag_off_clears_inflight():
    session = Session(32, "s", streaming=True)
    req = session.create_req(
        _session_recv("rid"),
        tokenizer=None,
        vocab_size=32,
    )
    assert session._inflight is True
    assert session._inflight_rid == "rid"
    scheduler = _queued_abort_scheduler([req])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False):
        scheduler._release_dropped_waiting_req_mm_inputs(req)

    assert session._inflight is False
    assert session._inflight_rid is None
    next_req = session.create_req(
        _session_recv("next"),
        tokenizer=None,
        vocab_size=32,
    )
    assert next_req.finished_reason is None


def test_P44_streaming_session_reap_waits_for_inflight_request(caplog):
    session = Session(32, "s", streaming=True)
    session._inflight = True
    session._inflight_rid = "rid"
    controller = object.__new__(SessionController)
    controller.sessions = {"s": session}
    controller.tree_cache = Mock()
    controller._last_reap_time = 0

    with caplog.at_level("INFO", logger="sglang.srt.session.session_controller"):
        controller._close("s")
        assert session.close_on_finish is True
        assert controller.plan_reap(2) is None

        session.abort_req("rid")
        plan = controller.plan_reap(4)
        assert plan is not None
        assert plan.deferred == ["s"]
        controller.apply_reap(plan)

    assert "s" not in controller.sessions
    assert (
        caplog.messages.count("Deferring session close for s (unfinished request)") == 1
    )


def test_P44_session_later_turn_queued_abort_releases_only_new_turn_items():
    from sglang.srt.managers.io_struct import AbortReq

    session = Session(32, "s", streaming=True)
    p = _item(_proxy())
    p.release_transport_proxies = Mock()
    a = _item(_proxy())
    a.release_transport_proxies = Mock()
    shared = MultimodalInputs(
        mm_items=[p],
        image_pad_len=[3],
        mrope_positions=torch.zeros(3, 5),
        mrope_position_delta=torch.zeros(1, 1),
    )
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)
    req = Req("rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.multimodal_inputs = shared
    req.session_mm_inherited = True
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        req.extend_image_inputs(
            MultimodalInputs(
                mm_items=[a],
                image_pad_len=[5],
                mrope_positions=torch.zeros(3, 2),
                mrope_position_delta=torch.zeros(1, 1),
            )
        )
    session._inflight = True
    session._inflight_rid = "rid"
    scheduler = _queued_abort_scheduler([req])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="rid"))

    a.release_transport_proxies.assert_called_once_with()
    p.release_transport_proxies.assert_not_called()
    assert parent.multimodal_inputs is shared
    assert shared.mm_items == [p]
    assert shared.image_pad_len == [3]
    assert shared.mrope_positions.shape == (3, 5)
    assert shared.mrope_position_delta.shape == (1, 1)
    assert shared.mrope_position_delta_repeated_cache is None
    assert req.multimodal_inputs is None
    assert req.session_turn_mm_state is None
    assert session._inflight is False
    assert "parent" in session.req_nodes


def test_P44_preaborted_streaming_req_does_not_clear_other_inflight_turn():
    session = Session(32, "s", streaming=True)
    session._inflight = True
    session._inflight_rid = "A"

    req = Req("B", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.to_finish = FINISH_ABORT()

    inner = Mock()
    cache = StreamingSession(inner)
    cache.slots["s"] = SessionSlot(kv=SimpleNamespace())
    assert cache.find_active_slot(req) is None
    assert req.session is None
    assert session._inflight is True
    assert session._inflight_rid == "A"

    controller = object.__new__(SessionController)
    controller.sessions = {"s": session}
    controller.tree_cache = Mock()
    controller._close("s")
    assert session.close_on_finish is True
    controller.tree_cache.release_session.assert_not_called()


def test_P44_session_sibling_turn_queued_abort_keeps_later_sibling_items():
    from sglang.srt.managers.io_struct import AbortReq

    session = Session(32, "s", streaming=False)
    p = _item(_proxy())
    p.release_transport_proxies = Mock()
    a = _item(_proxy())
    a.release_transport_proxies = Mock()
    b = _item(_proxy())
    b.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[p], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)

    req_a = Req("a", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_a.session = session
    req_a.multimodal_inputs = shared
    req_a.session_mm_inherited = True
    req_b = Req("b", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_b.session = session
    req_b.multimodal_inputs = shared
    req_b.session_mm_inherited = True
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        req_a.extend_image_inputs(MultimodalInputs(mm_items=[a], image_pad_len=[5]))
        req_b.extend_image_inputs(MultimodalInputs(mm_items=[b], image_pad_len=[7]))
    scheduler = _queued_abort_scheduler([req_a])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="a"))

    a.release_transport_proxies.assert_called_once_with()
    p.release_transport_proxies.assert_not_called()
    b.release_transport_proxies.assert_not_called()
    assert shared.mm_items[0] is p
    assert shared.mm_items[1] is b
    assert req_b.multimodal_inputs is shared
    assert req_a.multimodal_inputs is None
    assert req_a.session_turn_mm_state is None


def test_P44_session_sibling_turn_queued_abort_removes_own_mrope_slice():
    from sglang.srt.managers.io_struct import AbortReq

    session = Session(32, "s", streaming=False)
    p = _item(_proxy())
    p.release_transport_proxies = Mock()
    a = _item(_proxy())
    a.release_transport_proxies = Mock()
    b = _item(_proxy())
    b.release_transport_proxies = Mock()
    shared = MultimodalInputs(
        mm_items=[p],
        image_pad_len=[3],
        mrope_positions=torch.full((3, 5), 0.0),
        mrope_position_delta=torch.full((1, 1), 0.0),
    )
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)

    req_a = Req("a", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_a.session = session
    req_a.multimodal_inputs = shared
    req_a.session_mm_inherited = True
    req_b = Req("b", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_b.session = session
    req_b.multimodal_inputs = shared
    req_b.session_mm_inherited = True
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        req_a.extend_image_inputs(
            MultimodalInputs(
                mm_items=[a],
                image_pad_len=[5],
                mrope_positions=torch.full((3, 2), 1.0),
                mrope_position_delta=torch.full((1, 1), 1.0),
            )
        )
        req_b.extend_image_inputs(
            MultimodalInputs(
                mm_items=[b],
                image_pad_len=[7],
                mrope_positions=torch.full((3, 4), 2.0),
                mrope_position_delta=torch.full((1, 1), 2.0),
            )
        )
    scheduler = _queued_abort_scheduler([req_a])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="a"))

    a.release_transport_proxies.assert_called_once_with()
    assert shared.mm_items == [p, b]
    assert shared.image_pad_len == [3, 7]
    assert shared.mrope_positions.shape == (3, 9)
    assert torch.equal(
        shared.mrope_positions,
        torch.cat(
            [
                torch.full((3, 5), 0.0),
                torch.full((3, 4), 2.0),
            ],
            dim=1,
        ),
    )
    assert shared.mrope_position_delta.shape == (2, 1)
    assert torch.equal(
        shared.mrope_position_delta,
        torch.cat(
            [
                torch.full((1, 1), 0.0),
                torch.full((1, 1), 2.0),
            ],
            dim=0,
        ),
    )
    assert shared.mrope_position_delta_repeated_cache is None
    assert req_b.multimodal_inputs is shared


def _repeated_sibling_mm_session():
    session = Session(32, "s", streaming=False)
    p = _item(_proxy())
    p.release_transport_proxies = Mock()
    a = _item(_proxy())
    a.release_transport_proxies = Mock()
    b = _item(_proxy())
    b.release_transport_proxies = Mock()
    shared = MultimodalInputs(
        mm_items=[p],
        image_pad_len=[3],
        mrope_positions=torch.full((3, 5), 0.0),
        mrope_position_delta=torch.full((1, 1), 0.0),
    )
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)

    req_a = Req("a", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_a.session = session
    req_a.multimodal_inputs = shared
    req_a.session_mm_inherited = True
    req_b = Req("b", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_b.session = session
    req_b.multimodal_inputs = shared
    req_b.session_mm_inherited = True
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        req_a.extend_image_inputs(
            MultimodalInputs(
                mm_items=[a],
                image_pad_len=[5],
                mrope_positions=torch.full((3, 2), 1.0),
                mrope_position_delta=torch.full((1, 1), 1.0),
            )
        )
        req_b.extend_image_inputs(
            MultimodalInputs(
                mm_items=[b],
                image_pad_len=[7],
                mrope_positions=torch.full((3, 4), 2.0),
                mrope_position_delta=torch.full((1, 1), 2.0),
            )
        )
    return session, shared, req_a, req_b, p, a, b


def test_P44_session_repeated_sibling_aborts_restore_parent_mrope():
    from sglang.srt.managers.io_struct import AbortReq

    session, shared, req_a, req_b, p, a, b = _repeated_sibling_mm_session()
    scheduler = _queued_abort_scheduler([req_a, req_b])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="a"))
        scheduler.abort_request(AbortReq(rid="b"))

    assert p.release_transport_proxies.call_count == 0
    a.release_transport_proxies.assert_called_once_with()
    b.release_transport_proxies.assert_called_once_with()
    assert shared.mm_items == [p]
    assert shared.image_pad_len == [3]
    assert torch.equal(shared.mrope_positions, torch.zeros(3, 5))
    assert shared.mrope_position_delta.shape == (1, 1)
    assert torch.equal(shared.mrope_position_delta, torch.zeros(1, 1))
    assert shared.session_turn_states == []


def test_P44_session_repeated_sibling_aborts_restore_parent_mrope_reverse_order():
    from sglang.srt.managers.io_struct import AbortReq

    session, shared, req_a, req_b, p, a, b = _repeated_sibling_mm_session()
    scheduler = _queued_abort_scheduler([req_a, req_b])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="b"))
        scheduler.abort_request(AbortReq(rid="a"))

    assert p.release_transport_proxies.call_count == 0
    a.release_transport_proxies.assert_called_once_with()
    b.release_transport_proxies.assert_called_once_with()
    assert shared.mm_items == [p]
    assert shared.image_pad_len == [3]
    assert torch.equal(shared.mrope_positions, torch.zeros(3, 5))
    assert shared.mrope_position_delta.shape == (1, 1)
    assert torch.equal(shared.mrope_position_delta, torch.zeros(1, 1))
    assert shared.session_turn_states == []


def test_P44_session_inherited_only_turn_queued_abort_releases_nothing():
    from sglang.srt.managers.io_struct import AbortReq

    session = Session(32, "s", streaming=False)
    p = _item(_proxy())
    p.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[p], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)
    req = Req("child", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.multimodal_inputs = shared
    req.session_mm_inherited = True
    scheduler = _queued_abort_scheduler([req])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler.abort_request(AbortReq(rid="child"))

    p.release_transport_proxies.assert_not_called()
    assert shared.mm_items == [p]
    assert req.multimodal_inputs is None
    assert req.session_turn_mm_state is None


def test_P44_session_queued_abort_flag_off_releases_nothing():
    from sglang.srt.managers.io_struct import AbortReq

    session = Session(32, "s", streaming=False)
    p = _item(_proxy())
    p.release_transport_proxies = Mock()
    a = _item(_proxy())
    a.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[p], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)
    req = Req("child", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.multimodal_inputs = shared
    req.session_mm_inherited = True
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False):
        req.extend_image_inputs(MultimodalInputs(mm_items=[a], image_pad_len=[5]))
    scheduler = _queued_abort_scheduler([req])

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False):
        scheduler.abort_request(AbortReq(rid="child"))

    p.release_transport_proxies.assert_not_called()
    a.release_transport_proxies.assert_not_called()
    assert shared.mm_items == [p, a]
    assert req.multimodal_inputs is shared
    assert req.session_turn_mm_state is None


def test_P44_session_abort_at_create_turn_gets_own_mm_object():
    session = Session(32, "s", streaming=False)
    parent_item = _item(_proxy())
    parent_item.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[parent_item], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        new_req = session.create_req(
            _session_recv("child", "parent"),
            tokenizer=None,
            vocab_size=32,
        )
        fresh_item = _item(_proxy())
        fresh_item.release_transport_proxies = Mock()
        new_req.extend_image_inputs(
            MultimodalInputs(mm_items=[fresh_item], image_pad_len=[5])
        )
        session.release_dropped_turn_mm_inputs(new_req)

    assert new_req.multimodal_inputs is None
    assert not new_req.session_mm_inherited
    fresh_item.release_transport_proxies.assert_called_once_with()
    parent_item.release_transport_proxies.assert_not_called()
    assert shared.mm_items == [parent_item]
    assert shared.image_pad_len == [3]


def test_P44_session_abort_at_create_turn_flag_off_copies_mm_object():
    session = Session(32, "s", streaming=False)
    shared = MultimodalInputs(mm_items=[_item(_proxy())], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    session.req_nodes["parent"] = SessionReqNode(parent)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False),
        patch.object(Req, "set_finish_with_abort") as set_finish_with_abort,
    ):
        new_req = session.create_req(
            _session_recv("child", "parent"),
            tokenizer=None,
            vocab_size=32,
        )

    set_finish_with_abort.assert_called_once()
    assert new_req.multimodal_inputs is not shared
    assert new_req.multimodal_inputs.mm_items is not shared.mm_items
    assert new_req.multimodal_inputs.image_pad_len is not shared.image_pad_len
    assert not new_req.session_mm_inherited


def test_P44_session_close_skips_unfinished_tree_turns():
    session = Session(32, "s", streaming=False)
    shared = Mock()
    finished = Req("finished", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    finished.multimodal_inputs = shared
    finished.finished_reason = object()
    unfinished = Req(
        "unfinished", "", array("q", [1]), SamplingParams(max_new_tokens=1)
    )
    unfinished.session = session
    unfinished.multimodal_inputs = shared
    own = Mock()
    own_req = Req("own", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    own_req.multimodal_inputs = own
    own_req.finished_reason = object()
    session.req_nodes = {
        "finished": SessionReqNode(finished),
        "unfinished": SessionReqNode(unfinished),
        "own": SessionReqNode(own_req),
    }
    controller = object.__new__(SessionController)
    controller.sessions = {"s": session}
    controller.tree_cache = Mock()

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        controller._close("s")

    shared.release_features.assert_not_called()
    assert unfinished.multimodal_inputs is shared
    assert unfinished.session is session
    own.release_features.assert_called_once_with()
    assert own_req.multimodal_inputs is None


def test_P44_session_close_keeps_multiple_unfinished_shared_referents():
    session = Session(32, "s", streaming=False)
    shared = Mock()
    req_a = Req("a", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_a.session = session
    req_a.multimodal_inputs = shared
    req_b = Req("b", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_b.session = session
    req_b.multimodal_inputs = shared
    session.req_nodes = {
        "a": SessionReqNode(req_a),
        "b": SessionReqNode(req_b),
    }
    controller = object.__new__(SessionController)
    controller.sessions = {"s": session}
    controller.tree_cache = Mock()

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        controller._close("s")

    shared.release_features.assert_not_called()
    assert shared.session_live_refs == 2
    assert req_a.session is session
    assert req_a.multimodal_inputs is shared
    assert req_b.session is session
    assert req_b.multimodal_inputs is shared

    scheduler = _queued_abort_scheduler([])
    req_a.finished_reason = object()
    req_b.finished_reason = object()
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler._maybe_clear_mm_inputs(SimpleNamespace(reqs=[req_a]))
    assert shared.session_live_refs == 1
    assert req_a.multimodal_inputs is None
    shared.release_features.assert_not_called()
    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler._maybe_clear_mm_inputs(SimpleNamespace(reqs=[req_b]))
    assert shared.session_live_refs == 0
    assert req_b.multimodal_inputs is None
    shared.release_features.assert_called_once_with()


def test_P44_detached_abort_refcounts_shared_mm_inputs():
    shared = Mock()
    shared.session_live_refs = 2
    req_a = Req("a", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_a.multimodal_inputs = shared
    req_b = Req("b", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_b.multimodal_inputs = shared

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req_a.set_finish_with_abort("a")
    assert shared.session_live_refs == 1
    shared.release_features.assert_not_called()
    assert req_a.multimodal_inputs is None

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req_b.set_finish_with_abort("b")
    assert shared.session_live_refs == 0
    shared.release_features.assert_called_once_with()
    assert req_b.multimodal_inputs is None


def test_P44_dropped_session_turn_refcounts_shared_mm_inputs():
    session = Session(32, "s", streaming=False)
    shared = Mock()
    shared.session_live_refs = 2
    req_a = Req("a", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_a.session = session
    req_a.multimodal_inputs = shared
    req_b = Req("b", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req_b.session = session
    req_b.multimodal_inputs = shared

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        session.release_dropped_turn_mm_inputs(req_a)
    assert shared.session_live_refs == 1
    shared.release_features.assert_not_called()
    assert req_a.multimodal_inputs is None

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        session.release_dropped_turn_mm_inputs(req_b)
    assert shared.session_live_refs == 0
    shared.release_features.assert_called_once_with()
    assert req_b.multimodal_inputs is None


def test_P44_session_close_flag_off_releases_unfinished_tree_turns():
    session = Session(32, "s", streaming=False)
    shared = Mock()
    finished = Req("finished", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    finished.multimodal_inputs = shared
    finished.finished_reason = object()
    unfinished = Req(
        "unfinished", "", array("q", [1]), SamplingParams(max_new_tokens=1)
    )
    unfinished.multimodal_inputs = shared
    session.req_nodes = {
        "finished": SessionReqNode(finished),
        "unfinished": SessionReqNode(unfinished),
    }
    controller = object.__new__(SessionController)
    controller.sessions = {"s": session}
    controller.tree_cache = Mock()

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False):
        controller._close("s")

    shared.release_features.assert_called_once_with()
    assert finished.multimodal_inputs is None
    assert unfinished.multimodal_inputs is None


def test_P44_session_dropped_turn_with_live_parent_releases_nothing():
    session = Session(32, "s", streaming=False)
    parent_item = _item(_proxy())
    parent_item.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[parent_item], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    child = Req("child", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    child.session = session
    child.multimodal_inputs = shared
    child.session_mm_inherited = True
    child.session_mm_parent = parent

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        session.release_dropped_turn_mm_inputs(child)

    parent_item.release_transport_proxies.assert_not_called()
    assert shared.mm_items == [parent_item]
    assert child.multimodal_inputs is None
    assert child.session_turn_mm_state is None
    assert child.session_mm_parent is None


def test_P44_session_preabort_releases_dropped_turn_leases():
    session = Session(32, "s", streaming=False)
    parent_item = _item(_proxy())
    parent_item.release_transport_proxies = Mock()
    new_item = _item(_proxy())
    new_item.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[parent_item], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    parent.finished_reason = object()
    session.req_nodes["parent"] = SessionReqNode(parent)
    req = Req("child", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.multimodal_inputs = shared
    req.session_mm_inherited = True
    req.session_mm_parent = parent

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req.extend_image_inputs(
            MultimodalInputs(mm_items=[new_item], image_pad_len=[5])
        )
        req.set_finish_with_abort("x")

    new_item.release_transport_proxies.assert_called_once_with()
    parent_item.release_transport_proxies.assert_not_called()
    assert shared.mm_items == [parent_item]
    assert shared.image_pad_len == [3]
    assert req.multimodal_inputs is None


def test_P44_session_preabort_flag_off_keeps_shared_mm_object_untouched():
    session = Session(32, "s", streaming=False)
    parent_item = _item(_proxy())
    parent_item.release_transport_proxies = Mock()
    new_item = _item(_proxy())
    new_item.release_transport_proxies = Mock()
    shared = MultimodalInputs(mm_items=[parent_item], image_pad_len=[3])
    parent = Req("parent", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    parent.multimodal_inputs = shared
    parent.finished_reason = object()
    session.req_nodes["parent"] = SessionReqNode(parent)
    req = Req("child", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.session = session
    req.multimodal_inputs = shared
    req.session_mm_inherited = True
    req.session_mm_parent = parent

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(False),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req.extend_image_inputs(
            MultimodalInputs(mm_items=[new_item], image_pad_len=[5])
        )
        req.set_finish_with_abort("x")

    parent_item.release_transport_proxies.assert_not_called()
    new_item.release_transport_proxies.assert_not_called()
    assert shared.mm_items == [parent_item, new_item]
    assert shared.image_pad_len == [3, 5]
    assert req.multimodal_inputs is None


def _mm_req(rid, priority=None):
    item = _item(_proxy())
    item.release_transport_proxies = Mock()
    req = Req(
        rid, "", array("q", [1]), SamplingParams(max_new_tokens=1), priority=priority
    )
    req.multimodal_inputs = MultimodalInputs(mm_items=[item])
    return req, item.release_transport_proxies


def _admission_scheduler(waiting_queue, *, priority):
    from sglang.srt.disaggregation.utils import DisaggregationMode

    scheduler = _queued_abort_scheduler(waiting_queue)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.max_queued_requests = len(waiting_queue)
    scheduler.enable_priority_scheduling = priority
    scheduler.schedule_low_priority_values_first = True
    scheduler.abort_on_priority_when_disabled = False
    scheduler.enable_hierarchical_cache = False
    scheduler.server_args = SimpleNamespace(schedule_policy="fcfs")
    return scheduler


def test_P44_queue_full_rejection_releases_incoming_leases():
    queued, queued_release = _mm_req("queued")
    incoming, incoming_release = _mm_req("incoming")
    scheduler = _admission_scheduler([queued], priority=False)

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler._add_request_to_queue(incoming)

    assert scheduler.waiting_queue == [queued]
    incoming_release.assert_called_once_with()
    queued_release.assert_not_called()


def test_P44_priority_eviction_releases_only_evicted_leases():
    queued, queued_release = _mm_req("queued", priority=10)
    queued.time_stats.set_wait_queue_entry_time()
    incoming, incoming_release = _mm_req("incoming", priority=1)
    scheduler = _admission_scheduler([queued], priority=True)
    scheduler._prefetch_kvcache = Mock()

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        scheduler._add_request_to_queue(incoming)

    assert scheduler.waiting_queue == [incoming]
    queued_release.assert_called_once_with()
    incoming_release.assert_not_called()
    assert incoming.multimodal_inputs is not None


def test_P44_waiting_timeout_releases_timed_out_leases():
    stale, stale_release = _mm_req("stale")
    stale.time_stats.wait_queue_entry_time = 1.0
    fresh, fresh_release = _mm_req("fresh")
    fresh.time_stats.set_wait_queue_entry_time()
    scheduler = _queued_abort_scheduler([stale, fresh])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        envs.SGLANG_REQ_WAITING_TIMEOUT.override(1),
    ):
        scheduler._abort_on_waiting_timeout()

    assert scheduler.waiting_queue == [fresh]
    stale_release.assert_called_once_with()
    fresh_release.assert_not_called()


def test_P44_retract_resets_prefix_ack_state():
    """A retracted request re-prefills and must re-ack prefix-resident leases."""
    proxy_a = _lease_proxy()
    proxy_a.reconstruct_on_target_device = Mock(return_value=torch.ones(1))
    item_a = _item(proxy_a)
    item_a.offsets = [(0, 3)]
    proxy_b = _lease_proxy(generation=2, ready_byte_offset=8)
    proxy_b.reconstruct_on_target_device = Mock(return_value=torch.ones(1))
    item_b = _item(proxy_b)
    item_b.offsets = [(4, 7)]
    req = Req("rid", "", array("q", [1] * 8), SamplingParams(max_new_tokens=1))
    req.multimodal_inputs = MultimodalInputs(mm_items=[item_a, item_b])
    req.prefix_indices = torch.arange(4)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "CudaIpcTensorTransportProxy", CudaIpcTensorTransportProxy
        ),
        patch.object(schedule_batch.torch.cuda, "current_device", return_value=0),
    ):
        schedule_batch._acknowledge_prefix_resident_requests([req])
        assert isinstance(item_a.feature, torch.Tensor)
        assert item_b.feature is proxy_b
        assert req.mm_prefix_ack_done is True

        req.reset_for_retract()
        assert req.mm_prefix_ack_done is False

        req.prefix_indices = torch.arange(8)
        schedule_batch._acknowledge_prefix_resident_requests([req])
        assert isinstance(item_b.feature, torch.Tensor)
        proxy_a.reconstruct_on_target_device.assert_called_once()
