# ruff: noqa: E402
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import json
import unittest
from unittest.mock import Mock

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _MockTokenizerManager:
    def __init__(self):
        self.server_args = Mock(
            tokenizer_metrics_allowed_custom_labels=None,
            tool_call_parser="qwen3_coder",
            reasoning_parser=None,
        )

        mock_hf_config = Mock()
        mock_hf_config.model_type = "qwen3"

        self.model_config = Mock()
        self.model_config.hf_config = mock_hf_config
        self.model_config.get_default_sampling_params.return_value = {}

        self.tokenizer = Mock()


class _MockTemplateManager:
    pass


def _decode_sse_chunks(chunks):
    events = []
    for chunk in chunks:
        event_type = None
        data = None
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if event_type is not None:
            events.append((event_type, data))
    return events


async def _collect_stream(generator):
    return [chunk async for chunk in generator]


class ServingResponsesStreamTestCase(unittest.TestCase):
    def setUp(self):
        self.serving = OpenAIServingResponses(
            _MockTokenizerManager(), _MockTemplateManager()
        )
        self.fake_full_usage = UsageInfo(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )

    async def _fake_full_generator(
        self,
        request,
        sampling_params,
        result_generator,
        context,
        model_name,
        tokenizer,
        request_metadata,
        created_time=None,
    ):
        return ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time or 0,
            output=[
                ResponseFunctionToolCall(
                    type="function_call",
                    id="fc_wrong",
                    call_id="call_wrong",
                    name="echo",
                    arguments="{}",
                    status="completed",
                )
            ],
            status="completed",
            usage=self.fake_full_usage,
        )

    def test_simple_context_function_call_stream_emits_done_events_with_stable_ids(
        self,
    ):
        request = ResponsesRequest(
            model="x",
            input="Use echo.",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "echo",
                    "description": "Echo text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ],
            tool_choice="auto",
            request_id="resp_test",
        )
        context = SimpleContext()

        async def result_generator():
            context.append_output(
                {
                    "text": (
                        "<tool_call><function=echo>"
                        "<parameter=text>hello</parameter>"
                        "</function></tool_call>"
                    ),
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
            )
            yield context

        self.serving.responses_full_generator = self._fake_full_generator
        chunks = asyncio.run(
            _collect_stream(
                self.serving.responses_stream_generator(
                    request,
                    {},
                    result_generator(),
                    context,
                    "x",
                    Mock(),
                    RequestResponseMetadata(request_id=request.request_id),
                    created_time=123,
                )
            )
        )

        events = _decode_sse_chunks(chunks)
        event_types = [event_type for event_type, _ in events]

        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertIn("response.output_item.done", event_types)
        self.assertLess(
            event_types.index("response.function_call_arguments.done"),
            event_types.index("response.completed"),
        )
        self.assertLess(
            event_types.index("response.output_item.done"),
            event_types.index("response.completed"),
        )

        added = next(
            data
            for event_type, data in events
            if event_type == "response.output_item.added"
        )
        args_done = next(
            data
            for event_type, data in events
            if event_type == "response.function_call_arguments.done"
        )
        item_done = next(
            data
            for event_type, data in events
            if event_type == "response.output_item.done"
        )
        completed = next(
            data for event_type, data in events if event_type == "response.completed"
        )

        added_item = added["item"]
        done_item = item_done["item"]
        completed_item = completed["response"]["output"][0]

        self.assertEqual(args_done["arguments"], '{"text": "hello"}')
        self.assertEqual(args_done["item_id"], added_item["id"])
        self.assertEqual(done_item["id"], added_item["id"])
        self.assertEqual(done_item["call_id"], added_item["call_id"])
        self.assertEqual(completed_item["id"], added_item["id"])
        self.assertEqual(completed_item["call_id"], added_item["call_id"])
        self.assertEqual(completed_item["arguments"], '{"text": "hello"}')

    def test_simple_context_stream_preserves_cached_token_usage(self):
        request = ResponsesRequest(
            model="x",
            input="Say hi.",
            stream=True,
            tool_choice="auto",
            request_id="resp_cached",
        )
        context = SimpleContext()
        self.fake_full_usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=6),
        )

        async def result_generator():
            context.append_output(
                {
                    "text": "hi",
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
            )
            yield context

        self.serving.responses_full_generator = self._fake_full_generator
        chunks = asyncio.run(
            _collect_stream(
                self.serving.responses_stream_generator(
                    request,
                    {},
                    result_generator(),
                    context,
                    "x",
                    Mock(),
                    RequestResponseMetadata(request_id=request.request_id),
                    created_time=123,
                )
            )
        )

        events = _decode_sse_chunks(chunks)
        completed = next(
            data for event_type, data in events if event_type == "response.completed"
        )

        self.assertEqual(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            6,
        )

    def test_simple_context_length_finish_emits_response_incomplete(self):
        request = ResponsesRequest(
            model="x",
            input="Write a long answer.",
            stream=True,
            tool_choice="auto",
            max_output_tokens=1,
            request_id="resp_incomplete",
        )
        context = SimpleContext()

        async def result_generator():
            context.append_output(
                {
                    "text": "Partial",
                    "meta_info": {"finish_reason": {"type": "length"}},
                }
            )
            yield context

        self.serving.responses_full_generator = self._fake_full_generator
        chunks = asyncio.run(
            _collect_stream(
                self.serving.responses_stream_generator(
                    request,
                    {},
                    result_generator(),
                    context,
                    "x",
                    Mock(),
                    RequestResponseMetadata(request_id=request.request_id),
                    created_time=123,
                )
            )
        )

        events = _decode_sse_chunks(chunks)
        event_types = [event_type for event_type, _ in events]
        item_done = next(
            data for event_type, data in events if event_type == "response.output_item.done"
        )
        incomplete = next(
            data for event_type, data in events if event_type == "response.incomplete"
        )

        self.assertIn("response.incomplete", event_types)
        self.assertNotIn("response.completed", event_types)
        self.assertEqual(item_done["item"]["status"], "incomplete")
        self.assertEqual(incomplete["response"]["status"], "incomplete")
        self.assertEqual(incomplete["response"]["output"][0]["status"], "incomplete")
        self.assertEqual(
            incomplete["response"]["incomplete_details"],
            {"reason": "max_output_tokens"},
        )

    def test_responses_assistant_output_text_content_converts_to_string(self):
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": "Remember the code word CERULEAN. Reply only OK.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_previous",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
                {
                    "role": "user",
                    "content": "What code word did I ask you to remember?",
                },
            ],
        )

        converted = self.serving._convert_responses_input_item(request.input[1])

        self.assertEqual(converted["role"], "assistant")
        self.assertEqual(converted["content"], "OK")
        self.assertNotIn("type", converted)
        self.assertNotIn("id", converted)
        self.assertNotIn("status", converted)


if __name__ == "__main__":
    unittest.main()
