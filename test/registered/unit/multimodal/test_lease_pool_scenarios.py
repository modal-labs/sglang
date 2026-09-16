"""CPU scenario tests for multimodal lease ownership bug classes."""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.multimodal.processors.kimi_k25 import MMFeatureStreamSink
from sglang.srt.multimodal.transport import lease_lifecycle
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.multimodal.transport.lease_guard import StaleLeaseError
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import (
    Session,
    SessionController,
    SessionReqNode,
)
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


def _item(proxy):
    return MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)


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
    req = Req("rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    req.multimodal_inputs = mm_inputs
    req.session = object()

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            schedule_batch, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
    ):
        req.set_finish_with_abort("session abort")
    mm_inputs.release_features.assert_not_called()

    close_req = Req("close-rid", "", array("q", [1]), SamplingParams(max_new_tokens=1))
    close_req.multimodal_inputs = mm_inputs
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
