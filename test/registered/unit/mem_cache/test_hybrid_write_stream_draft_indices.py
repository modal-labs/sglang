"""CPU regression test: hybrid start_writing records the draft D2H indices on
the write stream even when they differ from the recorded target indices."""

from unittest.mock import MagicMock, patch

import torch

import sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller as hybrid_controller_module
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridWriteStreamDraftIndices(CustomTestCase):
    def _controller(self, target_write_back_jit: bool) -> HybridCacheController:
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.has_draft = True
        # Non-src dedup rank: dummy target host pool (no write-back JIT).
        controller.mla_broadcaster = MagicMock(is_src=target_write_back_jit)
        controller.io_backend = "kernel"
        controller.mem_pool_host = MagicMock(
            layout="page_first", can_use_write_back_jit=target_write_back_jit
        )
        controller.mem_pool_host_draft = MagicMock(
            layout="page_first", can_use_write_back_jit=True
        )
        controller.mem_pool_device = MagicMock()
        controller.mem_pool_device_draft = MagicMock()
        controller.write_stream = MagicMock()
        controller.ack_write_queue = []
        controller._record_transfer_indices_on_stream = MagicMock()
        # Model a target index path that yields tensors distinct from op.*.
        controller.move_indices = MagicMock(
            side_effect=lambda host, device: (host.clone(), device.clone())
        )
        return controller

    def _run(self, controller: HybridCacheController):
        ops = [
            hybrid_controller_module.CacheOperation(
                torch.arange(0, 8, dtype=torch.int64),
                torch.arange(100, 108, dtype=torch.int64),
                1,
            ),
            hybrid_controller_module.CacheOperation(
                torch.arange(8, 16, dtype=torch.int64),
                torch.arange(108, 116, dtype=torch.int64),
                2,
            ),
        ]
        controller.write_queue = list(ops)
        with (
            patch.object(
                hybrid_controller_module.device_module,
                "Event",
                return_value=MagicMock(),
            ),
            patch.object(
                hybrid_controller_module.device_module,
                "stream",
                return_value=MagicMock(),
            ),
        ):
            controller.start_writing()
        return ops

    def test_dummy_rank_records_merged_draft_indices(self):
        controller = self._controller(target_write_back_jit=False)
        ops = self._run(controller)

        draft_call = controller.mem_pool_host_draft.backup_from_device_all_layer
        draft_call.assert_called_once()
        draft_host, draft_device = draft_call.call_args.args[1:3]
        # Merged op: draft uses the fresh concatenated op tensors verbatim.
        assert draft_host is not ops[0].host_indices
        assert torch.equal(draft_host, torch.arange(0, 16, dtype=torch.int64))
        assert torch.equal(draft_device, torch.arange(100, 116, dtype=torch.int64))
        # Target D2H is skipped on the dummy rank.
        controller.mem_pool_host.backup_from_device_all_layer.assert_not_called()

        recorded = controller._record_transfer_indices_on_stream.call_args_list
        assert len(recorded) == 2
        # Target record covers only the moved (cloned) tensors ...
        assert recorded[0].args[1] is not draft_host
        assert recorded[0].args[2] is not draft_device
        # ... so the draft tensors must be recorded separately.
        assert recorded[1].args[0] is controller.write_stream
        assert recorded[1].args[1] is draft_host
        assert recorded[1].args[2] is draft_device

    def test_src_rank_records_draft_indices_too(self):
        controller = self._controller(target_write_back_jit=True)
        self._run(controller)
        draft_call = controller.mem_pool_host_draft.backup_from_device_all_layer
        draft_host, draft_device = draft_call.call_args.args[1:3]
        recorded = controller._record_transfer_indices_on_stream.call_args_list
        assert len(recorded) == 2
        assert recorded[1].args[1] is draft_host
        assert recorded[1].args[2] is draft_device


if __name__ == "__main__":
    import unittest

    unittest.main()
