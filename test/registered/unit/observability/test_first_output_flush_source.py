"""Exercise real output-streamer code without importing the GPU runtime.

The helpers compile the checked-in class/method unchanged; only dependent
transport/data types are replaced. Full serving qualification remains separate.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_node(path, class_name, method=None, **symbols):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    if method:
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == method
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"__name__": __name__, **symbols}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method or class_name]


Accumulator = load_node(
    SRT / "managers/scheduler_components/output_streamer.py",
    "_GenerationStreamAccumulator",
    dataclass=dataclass,
    field=field,
    DisaggregationMode=NS(NULL="null", PREFILL="prefill", DECODE="decode"),
    BatchTokenIDOutput=NS,
    wrap_as_pickle=lambda x: x,
)
FakeReq = load_node(
    ROOT / "test/registered/unit/managers/test_output_streamer_customized_info.py",
    "_FakeReq",
    SimpleNamespace=NS,
)


def emit(req):
    acc = Accumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=NS(is_none=lambda: True),
        disaggregation_mode="null",
        default_stream_interval=1,
        default_force_stream_interval=50,
        get_cached_tokens_details=lambda req: None,
    )
    acc.accept(req=req)
    return acc.to_payload(dp_rank=0, is_idle_batch=False)


@pytest.mark.parametrize("count", [1, 8, 50])
def test_first_output_flushes_including_speculative_blocks(count):
    req = FakeReq("r", list(range(count)))
    assert emit(req).output_ids == [list(range(count))]
    assert req.send_token_offset == count
    assert req.stream is False


def test_subsequent_output_remains_buffered_without_duplicate_tokens():
    req = FakeReq("r", list(range(8)))
    assert emit(req).output_ids == [list(range(8))]
    req.output_ids = req.output_ids_through_stop = list(range(9))
    assert emit(req) is None
    req.output_ids = req.output_ids_through_stop = list(range(50))
    assert emit(req).output_ids == [list(range(8, 50))]
    req.output_ids = req.output_ids_through_stop = list(range(53))
    req.finished = lambda: True
    req.finished_reason = NS(to_json=lambda: {"type": "length"})
    assert emit(req).output_ids == [list(range(50, 53))]


@pytest.mark.parametrize("count", [1, 50])
def test_partial_stop_prefix_is_not_flushed(count):
    req = FakeReq("r", list(range(count)))
    req.check_match_stop_str_prefix = Mock(return_value=True)
    assert emit(req) is None
    assert req.send_token_offset == 0
    req.check_match_stop_str_prefix.return_value = False
    assert emit(req).output_ids == [list(range(count))]


def test_finished_output_bypasses_partial_stop_prefix():
    req = FakeReq("r", [1])
    req.finished = lambda: True
    req.check_match_stop_str_prefix = Mock(return_value=True)
    assert emit(req).output_ids == [[1]]
    req.check_match_stop_str_prefix.assert_not_called()


def test_stream_interval_is_unchanged():
    req = FakeReq("r", [1])
    req.stream = True
    req.sampling_params.stream_interval = 3
    assert emit(req).output_ids == [[1]]
    req.output_ids = req.output_ids_through_stop = [1, 2]
    assert emit(req) is None
    req.output_ids = req.output_ids_through_stop = [1, 2, 3, 4]
    assert emit(req).output_ids == [[2, 3, 4]]


def test_nonstream_first_internal_output_does_not_reach_client():
    wait_one = load_node(
        SRT / "managers/tokenizer_manager.py",
        "TokenizerManager",
        "_wait_one_response",
        asyncio=asyncio,
        _REQUEST_STATE_WAIT_TIMEOUT=30,
    )

    async def scenario():
        obj = NS(rid="r", stream=False)
        state = NS(
            event=asyncio.Event(),
            out_list=[{"text": None, "meta_info": {}}],
            finished=False,
            time_stats=NS(response_sent_to_client_time=1),
        )
        state.event.set()
        manager = NS(
            incremental_streaming_output=False,
            rid_to_state={"r": state},
            request_logger=Mock(),
            request_metrics=NS(enabled=False),
            request_metrics_exporter_manager=NS(exporter_enabled=lambda: False),
        )
        response = wait_one(manager, obj)
        pending = asyncio.create_task(response.__anext__())
        try:

            async def consumed():
                while state.event.is_set():
                    await asyncio.sleep(0)

            await asyncio.wait_for(consumed(), 1)
            assert not pending.done()
            final = {"text": "complete", "meta_info": {}}
            state.out_list.append(final)
            state.finished = True
            state.event.set()
            assert await asyncio.wait_for(pending, 1) == final
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await response.aclose()

    asyncio.run(scenario())
