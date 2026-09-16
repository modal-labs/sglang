"""Exercise the actual NIXL dispatch with a recording transport, no GPU needed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("memory_kind,device", [("VRAM", 3), ("DRAM", 0)])
def test_descriptor_fallback_preserves_memory_registration(
    monkeypatch, memory_kind, device
):
    monkeypatch.setenv("SGLANG_NIXL_DISABLE_PREPPED", "1")
    source_ptrs = [1000, 3000, 6000]
    dest_ptrs = [10000, 13000, 16000]
    agent = MagicMock()
    agent.initialize_xfer.return_value = "handle"
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(kv_data_ptrs=source_ptrs, gpu_id=3),
        prep_handles={"": "src", "peer": "dst"},
        is_mla_backend=False,
        is_hybrid_mla_backend=True,
        agent=agent,
        get_mla_kv_ptrs_with_pp=lambda src, dst, st: (src, dst, 3),
    )
    result = NixlKVManager._send_kvcache_generic(
        manager,
        peer_name="peer",
        src_data_ptrs=source_ptrs,
        dst_data_ptrs=dest_ptrs,
        item_lens=[10, 20, 30],
        prefill_data_indices=np.array([1, 2], dtype=np.int32),
        dst_data_indices=np.array([3, 4], dtype=np.int32),
        dst_gpu_id=3,
        notif="done",
        src_mem_kind=memory_kind,
        dst_mem_kind=memory_kind,
    )
    assert result == "handle"
    agent.make_prepped_xfer.assert_not_called()
    assert agent.get_xfer_descs.call_count == 2
    src, kind = agent.get_xfer_descs.call_args_list[0].args
    np.testing.assert_array_equal(
        src, [[1010, 20, device], [3020, 40, device], [6030, 60, device]]
    )
    assert kind == memory_kind
    dst, kind = agent.get_xfer_descs.call_args_list[1].args
    np.testing.assert_array_equal(
        dst, [[10030, 20, device], [13060, 40, device], [16090, 60, device]]
    )
    assert kind == memory_kind


def test_prepared_path_remains_default(monkeypatch):
    monkeypatch.delenv("SGLANG_NIXL_DISABLE_PREPPED", raising=False)
    ptrs = [1000]
    agent = MagicMock()
    agent.make_prepped_xfer.return_value = "prepared"
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(kv_data_ptrs=ptrs),
        prep_handles={"": "src", "peer": "dst"},
        agent=agent,
        _num_slots_src=16,
        decode_kv_args_table={"peer": SimpleNamespace(dst_num_slots=16)},
    )
    result = NixlKVManager._send_kvcache_generic(
        manager,
        peer_name="peer",
        src_data_ptrs=ptrs,
        dst_data_ptrs=[10000],
        item_lens=[10],
        prefill_data_indices=np.array([1]),
        dst_data_indices=np.array([3]),
        dst_gpu_id=0,
        notif="done",
    )
    assert result == "prepared"
    agent.make_prepped_xfer.assert_called_once()
    agent.get_xfer_descs.assert_not_called()
