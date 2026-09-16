"""CPU tests for producer-side multimodal lease lifecycle handling."""

import asyncio
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import tokenizer_manager
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.kimi_k25 import MMFeatureStreamSink
from sglang.srt.multimodal.transport import lease_lifecycle
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakePool:
    def __init__(self, failing=None):
        self._pool_ipc_handle = ("h",)
        self.cancelled = []
        self.failing = failing

    def cancel_proxy(self, proxy):
        if proxy is self.failing:
            raise RuntimeError("cancel failed")
        self.cancelled.append(proxy)


def _proxy(pool_handle=("h",)):
    proxy = Mock(spec=CudaIpcTensorTransportProxy)
    proxy.proxy_state = {"ipc_extra": {"pool_handle": pool_handle}}
    return proxy


def test_cancel_undispatched_proxies_dedupes_and_continues():
    pool = FakePool()
    owned = _proxy()
    foreign = _proxy(("other",))
    failing = _proxy()
    pool.failing = failing
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=owned,
            precomputed_embeddings=owned,
        ),
        MultimodalDataItem(modality=Modality.IMAGE, feature=foreign),
        MultimodalDataItem(modality=Modality.IMAGE, feature=failing),
    ]

    with envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True):
        cancelled = lease_lifecycle.cancel_undispatched_proxies(
            pool, items, context="test"
        )

    assert cancelled == 1
    assert pool.cancelled == [owned]
    assert items[0].feature is None
    assert items[0].precomputed_embeddings is None
    assert items[1].feature is foreign
    assert items[2].feature is failing


def test_transport_imports_are_isolated_by_flag():
    code = (
        "import sys; "
        "import sglang.srt.managers.tokenizer_manager; "
        "import sglang.srt.multimodal.processors.kimi_k3; "
        "print('cuda_ipc=' + str('sglang.srt.multimodal.transport.cuda_ipc' in sys.modules)); "
        "print('lease_lifecycle=' + str('sglang.srt.multimodal.transport.lease_lifecycle' in sys.modules)); "
        "print('lease_guard=' + str('sglang.srt.multimodal.transport.lease_guard' in sys.modules))"
    )
    flag_off_env = os.environ.copy()
    flag_off_env.pop("SGLANG_MM_CUDA_IPC_LEASE_POOL", None)
    flag_off = subprocess.run(
        [sys.executable, "-c", code],
        env=flag_off_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "cuda_ipc=False" in flag_off.stdout
    assert "lease_lifecycle=False" in flag_off.stdout
    assert "lease_guard=False" in flag_off.stdout

    flag_on_env = os.environ.copy()
    flag_on_env["SGLANG_MM_CUDA_IPC_LEASE_POOL"] = "1"
    subprocess.run(
        [sys.executable, "-c", code],
        env=flag_on_env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_flag_off_preserves_legacy_shim_identity():
    # Base commit dev/instinct/2026-09-15 cuda_ipc_transport_utils.py.
    expected_sha256 = "6d022533a343131a5d234c8ec393d51afa70a1e8a5e564fa66a013abd2f9d7b3"
    legacy_path = (
        Path(__file__).parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "utils"
        / "cuda_ipc_transport_utils_legacy.py"
    )
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == expected_sha256

    code = """
import sys
from sglang.srt.utils import cuda_ipc_transport_utils as shim
from sglang.srt.utils import cuda_ipc_transport_utils_legacy as legacy
assert all(getattr(shim, name) is getattr(legacy, name) for name in shim.__all__)
assert "sglang.srt.multimodal.transport.cuda_ipc" not in sys.modules
assert "sglang.srt.multimodal.transport.lease_guard" not in sys.modules
assert "sglang.srt.multimodal.transport.lease_lifecycle" not in sys.modules
print("legacy identity and import isolation passed")
"""
    flag_off_env = os.environ.copy()
    flag_off_env.pop("SGLANG_MM_CUDA_IPC_LEASE_POOL", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=flag_off_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "legacy identity and import isolation passed" in result.stdout

    flag_on_code = """
from sglang.srt.utils import cuda_ipc_transport_utils as shim
from sglang.srt.multimodal.transport import cuda_ipc
assert shim.CudaIpcTensorTransportProxy is cuda_ipc.CudaIpcTensorTransportProxy
assert shim.MmItemMemoryPool is cuda_ipc.MmItemMemoryPool
print("lease-pool shim selection passed")
"""
    flag_on_env = os.environ.copy()
    flag_on_env["SGLANG_MM_CUDA_IPC_LEASE_POOL"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", flag_on_code],
        env=flag_on_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lease-pool shim selection passed" in result.stdout


def test_detach_proxies_for_clones_copies_each_proxy_once():
    pool = FakePool()
    owned = _proxy()
    foreign = _proxy(("other",))
    first = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=owned,
        precomputed_embeddings=owned,
    )
    second = MultimodalDataItem(modality=Modality.IMAGE, feature=foreign)
    mm_inputs = MultimodalInputs(mm_items=[first, second])
    copied = torch.tensor([7])

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(lease_lifecycle, "copy_lease_to_cpu", return_value=copied) as copy,
    ):
        clone = lease_lifecycle.detach_proxies_for_clones(pool, mm_inputs)

    assert mm_inputs.mm_items[0].feature is owned
    assert mm_inputs.mm_items[0].precomputed_embeddings is owned
    assert clone.mm_items[0].feature is copied
    assert clone.mm_items[0].precomputed_embeddings is copied
    assert clone.mm_items[1].feature is foreign
    copy.assert_called_once_with(pool, owned)


def test_parallel_clone_failure_cancels_all_originals():
    pool = FakePool()
    first_proxy = _proxy()
    second_proxy = _proxy()
    tokenized_objs = [
        SimpleNamespace(
            mm_inputs=MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, feature=first_proxy)
                ]
            )
        ),
        SimpleNamespace(
            mm_inputs=MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, feature=second_proxy)
                ]
            )
        ),
    ]

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch.object(
            lease_lifecycle,
            "detach_proxies_for_clones",
            side_effect=[tokenized_objs[0].mm_inputs, ValueError("clone failed")],
        ),
        pytest.raises(ValueError, match="clone failed"),
    ):
        tokenizer_manager._detach_parallel_clone_inputs(pool, tokenized_objs)

    assert pool.cancelled == [first_proxy, second_proxy]


def test_tokenize_failure_cancels_undispatched_proxies():
    pool = FakePool()
    proxy = _proxy()
    mm_output = MultimodalProcessorOutput(
        input_ids=[1],
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)],
    )
    processor = SimpleNamespace(
        use_cuda_ipc=True,
        prefer_tokenized_input=False,
        cudaipc_mmfeature_pool=pool,
    )
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.mm_processor = processor
    manager.server_args = SimpleNamespace(
        language_only=False,
        encoder_transfer_backend=None,
        disable_radix_cache=True,
    )
    manager.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[]))
    manager.max_req_input_len = 16
    manager.tokenizer = None
    manager._validate_mm_limits = lambda obj: None
    manager._validate_one_request = Mock(side_effect=ValueError("too long"))
    obj = SimpleNamespace(
        rid="tokenize-failure",
        text="",
        input_ids=[1],
        input_embeds=None,
        image_data=["image"],
        audio_data=None,
        video_data=None,
        need_wait_for_mm_inputs=False,
        mm_hashes=None,
        contains_mm_input=lambda: True,
    )

    async def process_mm_data_async(**kwargs):
        return mm_output

    processor.process_mm_data_async = process_mm_data_async

    async def run():
        with (
            envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
            pytest.raises(ValueError, match="too long"),
        ):
            await manager._tokenize_one_request(obj)

    asyncio.run(run())

    assert pool.cancelled == [proxy]


def test_feature_sink_cancels_proxies_but_not_cpu_fallback():
    pool = FakePool()
    proxy = _proxy()
    processor = SimpleNamespace(
        use_cuda_ipc=True,
        cudaipc_mmfeature_pool=pool,
        _wrap_tensor_for_cuda_ipc=Mock(side_effect=[proxy, torch.ones(1)]),
    )
    sink = MMFeatureStreamSink(processor)

    with (
        envs.SGLANG_MM_CUDA_IPC_LEASE_POOL.override(True),
        patch("sglang.srt.managers.mm_utils.hash_feature", return_value=1),
        patch(
            "sglang.srt.multimodal.transport.cuda_ipc.CudaIpcTensorTransportProxy",
            type(proxy),
        ),
    ):
        assert sink(0, torch.ones(1)) is proxy
        fallback = sink(1, torch.ones(1))
        sink.cancel_all("test")

    assert isinstance(fallback, torch.Tensor)
    assert pool.cancelled == [proxy]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
