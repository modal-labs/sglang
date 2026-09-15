from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_TEST_WORKSPACE_BYTES = 4096
_TEST_MAX_TILE_N = 256


class _FakeArena:
    device = torch.device("cuda:3")

    def __init__(self, size: int):
        self._size = size

    def numel(self):
        return self._size

    def data_ptr(self):
        return 0x1234


class _FakeRunner:
    device = "cuda"
    gpu_id = 3

    def __init__(self, compatible_method_count: int):
        self._methods = [object() for _ in range(compatible_method_count)]
        self.bind_calls: list[tuple[object, int]] = []

    def get_trtllm_gen_moe_eager_workspace_methods(self):
        return self._methods

    def init_trtllm_gen_moe_eager_workspace(self, workspace, *, max_tile_n):
        self.bind_calls.append((workspace, max_tile_n))
        return len(self._methods)


def _worker(target_runner):
    worker = object.__new__(TpModelWorker)
    worker.server_args = SimpleNamespace(
        enable_pdmux=False,
        enable_two_batch_overlap=False,
        enable_memory_saver=False,
    )
    worker.model_runner_list = []
    worker._model_runner = target_runner
    worker.trtllm_gen_moe_eager_workspace = None
    worker.trtllm_gen_moe_eager_workspace_max_tile_n = None
    return worker


def test_one_arena_is_allocated_and_shared_by_target_and_draft():
    target = _FakeRunner(compatible_method_count=2)
    draft = _FakeRunner(compatible_method_count=1)
    worker = _worker(target)
    arena = _FakeArena(_TEST_WORKSPACE_BYTES)

    with (
        envs.SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES.override(
            _TEST_WORKSPACE_BYTES
        ),
        envs.SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N.override(_TEST_MAX_TILE_N),
        patch("sglang.srt.managers.tp_worker.torch.empty", return_value=arena) as empty,
    ):
        result = worker.init_trtllm_gen_moe_eager_workspace(
            additional_model_runners=[draft]
        )

    assert result is arena
    empty.assert_called_once_with(
        _TEST_WORKSPACE_BYTES,
        dtype=torch.uint8,
        device=torch.device("cuda", 3),
    )
    assert target.bind_calls == [(arena, _TEST_MAX_TILE_N)]
    assert draft.bind_calls == [(arena, _TEST_MAX_TILE_N)]
    assert worker.trtllm_gen_moe_eager_workspace is arena


@pytest.mark.parametrize(
    ("workspace_bytes", "max_tile_n"),
    [
        (_TEST_WORKSPACE_BYTES, 0),
        (0, _TEST_MAX_TILE_N),
    ],
)
def test_workspace_and_tactic_cap_must_be_enabled_together(workspace_bytes, max_tile_n):
    worker = _worker(_FakeRunner(compatible_method_count=1))

    with (
        envs.SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES.override(workspace_bytes),
        envs.SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N.override(max_tile_n),
        pytest.raises(ValueError, match="must both be zero or both be non-zero"),
    ):
        worker.init_trtllm_gen_moe_eager_workspace()


def test_enabled_workspace_rejects_incompatible_models_without_allocating():
    worker = _worker(_FakeRunner(compatible_method_count=0))

    with (
        envs.SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES.override(
            _TEST_WORKSPACE_BYTES
        ),
        envs.SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N.override(_TEST_MAX_TILE_N),
        patch("sglang.srt.managers.tp_worker.torch.empty") as empty,
        pytest.raises(RuntimeError, match="no loaded target or draft model"),
    ):
        worker.init_trtllm_gen_moe_eager_workspace()

    empty.assert_not_called()


def test_enabled_workspace_does_not_hide_allocation_failure():
    worker = _worker(_FakeRunner(compatible_method_count=1))

    with (
        envs.SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES.override(
            _TEST_WORKSPACE_BYTES
        ),
        envs.SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N.override(_TEST_MAX_TILE_N),
        patch(
            "sglang.srt.managers.tp_worker.torch.empty",
            side_effect=RuntimeError("synthetic CUDA OOM"),
        ),
        pytest.raises(RuntimeError, match="synthetic CUDA OOM"),
    ):
        worker.init_trtllm_gen_moe_eager_workspace()


@pytest.mark.parametrize(
    ("server_arg", "error"),
    [
        ("enable_pdmux", "--enable-pdmux"),
        ("enable_two_batch_overlap", "--enable-two-batch-overlap"),
        ("enable_memory_saver", "--enable-memory-saver"),
    ],
)
def test_enabled_workspace_rejects_unsafe_ownership_modes(server_arg, error):
    worker = _worker(_FakeRunner(compatible_method_count=1))
    setattr(worker.server_args, server_arg, True)

    with (
        envs.SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES.override(
            _TEST_WORKSPACE_BYTES
        ),
        envs.SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N.override(_TEST_MAX_TILE_N),
        patch("sglang.srt.managers.tp_worker.torch.empty") as empty,
        pytest.raises(RuntimeError, match=error),
    ):
        worker.init_trtllm_gen_moe_eager_workspace()

    empty.assert_not_called()


def test_scheduler_initializes_workspace_after_loads_and_before_kv_sizing():
    scheduler = object.__new__(Scheduler)
    calls = []
    scheduler.init_tp_model_worker = Mock(side_effect=lambda: calls.append("target"))
    scheduler.maybe_init_draft_worker = Mock(side_effect=lambda: calls.append("draft"))
    scheduler.init_trtllm_gen_moe_eager_workspace = Mock(
        side_effect=lambda: calls.append("workspace")
    )
    scheduler.init_memory_pools = Mock(side_effect=lambda: calls.append("kv"))
    scheduler.init_all_attention_backends = Mock(
        side_effect=RuntimeError("stop after lifecycle order")
    )

    with pytest.raises(RuntimeError, match="stop after lifecycle order"):
        Scheduler.init_model_worker(scheduler)

    assert calls == ["target", "draft", "workspace", "kv"]


def test_scheduler_discovers_dflash_direct_and_underlying_draft_runners():
    direct_runner = _FakeRunner(compatible_method_count=0)
    underlying_runner = _FakeRunner(compatible_method_count=0)
    initializer = Mock()
    scheduler = object.__new__(Scheduler)
    scheduler.tp_worker = SimpleNamespace(
        init_trtllm_gen_moe_eager_workspace=initializer
    )
    scheduler.draft_worker = SimpleNamespace(
        draft_model_runner=direct_runner,
        draft_worker=SimpleNamespace(
            model_runner_list=[],
            model_runner=underlying_runner,
        ),
    )

    Scheduler.init_trtllm_gen_moe_eager_workspace(scheduler)

    initializer.assert_called_once_with(
        additional_model_runners=[direct_runner, underlying_runner]
    )


def test_scheduler_discovers_nested_multilayer_draft_runners_once():
    first = _FakeRunner(compatible_method_count=0)
    second = _FakeRunner(compatible_method_count=0)
    tp_worker = SimpleNamespace(
        model_runner=first,
        model_runner_list=[first, second],
    )
    eagle_worker = SimpleNamespace(
        draft_runner=first,
        draft_runners=[first, second],
        draft_worker=tp_worker,
    )
    spec_worker = SimpleNamespace(draft_worker=eagle_worker)

    assert Scheduler._collect_trtllm_gen_moe_model_runners(spec_worker) == [
        first,
        second,
    ]
