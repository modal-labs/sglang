import torch

from sglang.kernels.ops.attention import prefill_attention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeLauncher:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _attention_args():
    q = torch.empty((17, 2, 128))
    k = torch.empty((17, 2, 128))
    v = torch.empty_like(k)
    output = torch.empty_like(q)
    start_loc = torch.tensor([0], dtype=torch.int32)
    seq_len = torch.tensor([17], dtype=torch.int32)
    return q, k, v, output, start_loc, seq_len


def test_noncausal_context_attention_uses_persisted_autotuner(monkeypatch):
    autotuned = _FakeLauncher()
    fixed = _FakeLauncher()
    monkeypatch.setattr(
        prefill_attention, "_vision_context_attention_fwd_kernel", autotuned
    )
    monkeypatch.setattr(prefill_attention, "_fwd_kernel", fixed)
    monkeypatch.setattr(prefill_attention, "_is_cuda", False)
    monkeypatch.setattr(prefill_attention, "_is_hip", False)

    prefill_attention.context_attention_fwd(
        *_attention_args(), max_input_len=17, is_causal=False
    )

    assert not fixed.calls
    assert len(autotuned.calls) == 1
    grid, _, kwargs = autotuned.calls[0]
    assert grid({"BLOCK_M": 64}) == (1, 2, 1)
    assert "BLOCK_M" not in kwargs
    assert "BLOCK_N" not in kwargs
    assert kwargs["IS_CAUSAL"] is False


def test_causal_context_attention_preserves_fixed_launch(monkeypatch):
    autotuned = _FakeLauncher()
    fixed = _FakeLauncher()
    monkeypatch.setattr(
        prefill_attention, "_vision_context_attention_fwd_kernel", autotuned
    )
    monkeypatch.setattr(prefill_attention, "_fwd_kernel", fixed)
    monkeypatch.setattr(prefill_attention, "_is_cuda", False)
    monkeypatch.setattr(prefill_attention, "_is_hip", False)

    prefill_attention.context_attention_fwd(
        *_attention_args(), max_input_len=17, is_causal=True
    )

    assert not autotuned.calls
    assert len(fixed.calls) == 1
    grid, _, kwargs = fixed.calls[0]
    assert grid == (1, 2, 1)
    assert kwargs["BLOCK_M"] == 64
    assert kwargs["BLOCK_N"] == 64
    assert kwargs["IS_CAUSAL"] is True


def test_vision_autotuner_persists_results(monkeypatch):
    captured = {}

    def fake_autotune(*, configs, key, cache_results):
        captured.update(
            configs=configs,
            key=key,
            cache_results=cache_results,
        )
        return lambda kernel: kernel

    monkeypatch.setattr(prefill_attention.triton, "autotune", fake_autotune)

    assert (
        prefill_attention._make_vision_context_attention_autotuner()
        is prefill_attention._fwd_kernel
    )
    assert captured["cache_results"] is True
    assert captured["key"] == ["Lk", "kv_group_num", "IS_CAUSAL"]
    assert len(captured["configs"]) == 16
