"""Bounded, opt-in verification of actual GPU retraction restore boundaries.

This adds synchronization and device-to-host copies and is for isolated forced
retraction qualification only. It never logs tensor values or request bodies.
"""

import json
import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_MAX_RESTORES = 32
_MAX_TOKENS = 4096


def _assert_snapshot_equal(expected, actual, path="snapshot"):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: missing restored tensor")
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(f"{path}: restored tensor shape or dtype changed")
        # Byte equality also handles fp8 and NaN payloads, without numerical
        # tolerances masking a failed copy. Snapshots are already on the CPU.
        left = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        right = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left, right):
            raise AssertionError(f"{path}: restored bytes differ")
        return {"tensors": 1, "bytes": expected.numel() * expected.element_size()}
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{path}: restored snapshot structure changed")
        counts = [
            _assert_snapshot_equal(left, right, f"{path}[{i}]")
            for i, (left, right) in enumerate(zip(expected, actual))
        ]
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"{path}: restored snapshot keys changed")
        counts = [
            _assert_snapshot_equal(value, actual[key], f"{path}.{key}")
            for key, value in expected.items()
        ]
    else:
        if expected != actual:
            raise AssertionError(f"{path}: restored metadata changed")
        return {"tensors": 0, "bytes": 0}
    return {key: sum(count[key] for count in counts) for key in ("tensors", "bytes")}


def verify_retraction_restore(allocator, snapshot, indices, mamba_indices, rid):
    """Recapture through the real pool's DCP translation after slot reassignment."""
    if not envs.SGLANG_TEST_RETRACT.get():
        raise RuntimeError("SGLANG_TEST_RETRACT_VERIFY requires SGLANG_TEST_RETRACT")

    checked = getattr(allocator, "_retraction_verify_count", 0)
    event = {"rid": rid, "tokens": len(indices), "checked": checked}
    if checked >= _MAX_RESTORES or len(indices) > _MAX_TOKENS:
        event.update(
            status="skipped",
            reason="restore_limit" if checked >= _MAX_RESTORES else "token_limit",
        )
        logger.warning("PD_RETRACTION_VERIFY %s", json.dumps(event))
        return event

    allocator._retraction_verify_count = checked + 1
    actual = allocator.get_cpu_copy(indices, mamba_indices=mamba_indices)
    try:
        components = {}
        if isinstance(snapshot, tuple) and len(snapshot) in (2, 3):
            # HybridLinearKVPool snapshot: target, Mamba, optional full draft.
            for i, name in enumerate(("target", "mamba", "draft")[: len(snapshot)]):
                components[name] = _assert_snapshot_equal(snapshot[i], actual[i], name)
        else:
            components["target"] = _assert_snapshot_equal(snapshot, actual)
    except (AssertionError, IndexError, TypeError) as exc:
        event.update(status="failed", error=str(exc))
        logger.error("PD_RETRACTION_VERIFY %s", json.dumps(event))
        raise RuntimeError("Retraction restore byte verification failed") from exc
    event.update(status="passed", components=components, checked=checked + 1)
    logger.warning("PD_RETRACTION_VERIFY %s", json.dumps(event))
    return event
