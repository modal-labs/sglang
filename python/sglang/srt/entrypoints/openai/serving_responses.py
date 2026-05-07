# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's OpenAIServingResponses
"""Handler for /v1/responses requests"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from contextlib import AsyncExitStack
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Optional, Union

import jinja2
import openai.types.responses as openai_responses_types
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai_harmony import Message as OpenAIMessage

from sglang.srt.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    Function,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    Tool,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, ToolServer
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.utils import convert_json_schema_to_str
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import random_uuid

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _normalize_function_tools_for_chat_parser(tools: list[Any] | None) -> list[Tool]:
    """Convert Responses function tools to chat-style Tool objects."""
    normalized_tools: list[Tool] = []
    for tool in tools or []:
        if getattr(tool, "type", None) != "function":
            continue

        if hasattr(tool, "function"):
            normalized_tools.append(tool)
            continue

        name = getattr(tool, "name", None)
        if not name:
            continue

        normalized_tools.append(
            Tool(
                type="function",
                function=Function(
                    name=name,
                    description=getattr(tool, "description", None),
                    parameters=getattr(tool, "parameters", None),
                    strict=bool(getattr(tool, "strict", False)),
                ),
            )
        )
    return normalized_tools


def _convert_usage_info_to_response_usage(usage_info: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens_details = usage_info.get("prompt_tokens_details") or {}
    if not isinstance(prompt_tokens_details, dict):
        prompt_tokens_details = {}

    cached_tokens = prompt_tokens_details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage_info.get("cached_tokens", 0)

    return {
        "input_tokens": usage_info.get("prompt_tokens") or 0,
        "input_tokens_details": {"cached_tokens": cached_tokens or 0},
        "output_tokens": usage_info.get("completion_tokens") or 0,
        "output_tokens_details": {
            "reasoning_tokens": usage_info.get("reasoning_tokens") or 0
        },
        "total_tokens": usage_info.get("total_tokens") or 0,
    }


class OpenAIServingResponses(OpenAIServingChat):
    """Handler for /v1/responses requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        enable_prompt_tokens_details: bool = False,
        tool_server: Optional[ToolServer] = None,
    ) -> None:
        super().__init__(tokenizer_manager, template_manager)

        # template_manager is already set by parent class
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.enable_prompt_tokens_details = enable_prompt_tokens_details

        # Get default sampling params from model config if available
        self.default_sampling_params = {}

        self.supports_browsing = (
            tool_server.has_tool("browser") if tool_server else False
        )
        self.supports_code_interpreter = (
            tool_server.has_tool("python") if tool_server else False
        )
        self.tool_server = tool_server
        # Get from model config
        self.use_harmony = (
            self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        if self.use_harmony:
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Response storage for background and retrieval operations
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # Message storage for conversation continuity
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.msg_store: dict[
            str, Union[list[ChatCompletionMessageParam], list["OpenAIMessage"]]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

    # error helpers dedicated for v1/responses
    def create_error_response(
        self,
        message: str,
        err_type: str = "invalid_request_error",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        nested_error = {
            "message": message,
            "type": err_type,
            "param": param,
            "code": status_code,
        }
        return ORJSONResponse(content={"error": nested_error}, status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        return json.dumps(
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "param": None,
                    "code": status_code,
                }
            }
        )

    def _request_id_prefix(self) -> str:
        return "resp_"

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ORJSONResponse]:
        # Validate model
        if not self.tokenizer_manager:
            return self.create_error_response("Model not loaded")

        # FIXME: If the engine is dead, raise an error
        # This is required for the streaming case

        # Handle the previous response ID
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            if not prev_response_id.startswith("resp_"):
                return self._make_invalid_id_error(prev_response_id)
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        try:
            model_name = request.model
            tokenizer = self.tokenizer_manager.tokenizer

            if self.use_harmony:
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                messages, request_prompts, engine_prompts, processed_messages, tool_call_constraint = await self._make_request(
                    request, prev_response, tokenizer
                )

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if (
            self.tool_server is not None
            and isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and request.tools
            and any(
                tool.type in ["web_search_preview", "code_interpreter"]
                for tool in request.tools
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background mode and "
                "streaming mode"
            )

        # Schedule the request and get the result generator
        generators: list[AsyncGenerator[Any, None]] = []
        tool_list = []
        if self.use_harmony:
            if self.supports_browsing:
                tool_list.append("browser")
            if self.supports_code_interpreter:
                tool_list.append("python")
        async with AsyncExitStack() as exit_stack:
            try:
                if self.tool_server is not None:
                    tool_session_ctxs: dict[str, Any] = {
                        tool_name: exit_stack.enter_async_context(
                            self.tool_server.get_tool_session(tool_name)
                        )
                        for tool_name in tool_list
                    }
                    tool_sessions = {}
                    for tool_name in tool_list:
                        tool_sessions[tool_name] = await tool_session_ctxs[tool_name]
                else:
                    assert len(tool_list) == 0
                    tool_sessions = {}
                for i, engine_prompt in enumerate(engine_prompts):
                    # Calculate default max tokens from actual prompt tokens.
                    if self.use_harmony:
                        # Harmony render_for_completion already returns token ids.
                        prompt_length = (
                            len(engine_prompt)
                            if hasattr(engine_prompt, "__len__")
                            else 0
                        )
                    else:
                        prompt_ids = getattr(processed_messages, "prompt_ids", None)
                        if isinstance(prompt_ids, str):
                            prompt_length = len(tokenizer.encode(prompt_ids))
                        elif (
                            prompt_ids is not None
                            and hasattr(prompt_ids, "__len__")
                            and len(prompt_ids) > 0
                        ):
                            prompt_length = len(prompt_ids)
                        elif isinstance(engine_prompt, str):
                            prompt_length = len(tokenizer.encode(engine_prompt))
                        elif hasattr(engine_prompt, "__len__"):
                            prompt_length = len(engine_prompt)
                        else:
                            prompt_length = 0

                    context_len = (
                        self.tokenizer_manager.model_config.context_len
                        if hasattr(self.tokenizer_manager.model_config, "context_len")
                        else 4096
                    )
                    # Account for reserved tokens (e.g., EAGLE speculative decoding slots)
                    # that the tokenizer_manager adds during validation
                    num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                    default_max_tokens = max(
                        context_len - prompt_length - num_reserved_tokens, 512
                    )  # Ensure minimum 512 tokens
                    sampling_params = request.to_sampling_params(
                        default_max_tokens, self.default_sampling_params
                    )

                    has_function_tools = any(
                        getattr(t, "type", None) == "function"
                        for t in (request.tools or [])
                    )
                    if has_function_tools and self.tool_call_parser:
                        sampling_params["skip_special_tokens"] = False

                    if not self.use_harmony and tool_call_constraint:
                        constraint_type, constraint_value = tool_call_constraint
                        if constraint_type == "structural_tag":
                            sampling_params[constraint_type] = convert_json_schema_to_str(
                                constraint_value.model_dump(by_alias=True)
                            )
                        elif constraint_type == "json_schema":
                            sampling_params[constraint_type] = convert_json_schema_to_str(
                                constraint_value
                            )
                        else:
                            sampling_params[constraint_type] = constraint_value

                    context: ConversationContext
                    if self.use_harmony:
                        if request.stream:
                            context = StreamingHarmonyContext(messages, tool_sessions)
                        else:
                            context = HarmonyContext(messages, tool_sessions)
                    else:
                        context = SimpleContext()

                    # Create GenerateReqInput for SGLang
                    prompt_kwargs = (
                        {"text": engine_prompt}
                        if isinstance(engine_prompt, str)
                        else {"input_ids": engine_prompt}
                    )
                    adapted_request = GenerateReqInput(
                        **prompt_kwargs,
                        image_data=processed_messages.image_data,
                        video_data=processed_messages.video_data,
                        audio_data=processed_messages.audio_data,
                        sampling_params=sampling_params,
                        stream=request.stream,
                        modalities=processed_messages.modalities,
                        rid=request.request_id,
                        extra_key=self._compute_extra_key(request),
                        background=request.background,
                    )

                    generator = self._generate_with_builtin_tools(
                        request.request_id,
                        request_prompts[i],
                        adapted_request,
                        sampling_params,
                        context,
                        raw_request=raw_request,
                        priority=request.priority,
                    )
                    generators.append(generator)
            except ValueError as e:
                return self.create_error_response(str(e))

            assert len(generators) == 1
            (result_generator,) = generators

            # Store the input messages
            if request.store:
                self.msg_store[request.request_id] = messages

            if request.background:
                created_time = int(time.time())
                response = ResponsesResponse.from_request(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                    output=[],
                    status="queued",
                    usage=None,
                )
                async with self.response_store_lock:
                    self.response_store[response.id] = response

                # Run the request in the background
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    ),
                    name=f"create_{response.id}",
                )

                # For cleanup
                self.background_tasks[response.id] = task
                task.add_done_callback(
                    lambda _: self.background_tasks.pop(response.id, None)
                )
                return response

            if request.stream:
                return self.responses_stream_generator(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                )
            try:
                result: Union[ORJSONResponse, ResponsesResponse] = (
                    await self.responses_full_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                    )
                )
                return result
            except Exception as e:
                return self.create_error_response(str(e))
        return self.create_error_response("Unknown error")

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        # Construct the input messages
        messages = self._construct_input_messages(request, prev_response)

        # Follow SGLang's pattern: create a ChatCompletionRequest and process messages
        try:
            # Convert ResponsesRequest to ChatCompletionRequest for processing
            # Convert ResponsesRequest tools to ChatCompletionRequest format
            chat_tools = _normalize_function_tools_for_chat_parser(request.tools) or None
            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=messages,
                stream=request.stream,
                chat_template_kwargs=request.chat_template_kwargs,
                tools=chat_tools,
                tool_choice=request.tool_choice if chat_tools else "none",
            )

            # Follow SGLang's _process_messages pattern
            is_multimodal = self.tokenizer_manager.model_config.is_multimodal
            processed_messages = self._process_messages(chat_request, is_multimodal)

            # Extract the results
            if is_multimodal:
                request_prompts = [processed_messages.prompt]
                engine_prompts = [processed_messages.prompt]
            else:
                request_prompts = [processed_messages.prompt_ids]
                engine_prompts = [processed_messages.prompt_ids]

            tool_call_constraint = processed_messages.tool_call_constraint

        except Exception as e:
            logger.warning(f"Chat processing failed, rejecting request: {e}")
            raise ValueError(
                "Unable to convert /v1/responses input into chat messages. "
                "Stateless multi-turn input arrays must use supported "
                "message/function_call/function_call_output items."
            ) from e

        return (
            messages,
            request_prompts,
            engine_prompts,
            processed_messages,
            tool_call_constraint,
        )

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in " "response API"
            )
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = prompt_token_ids
        return messages, [prompt_token_ids], [engine_prompt]

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if created_time is None:
            created_time = int(time.time())

        try:
            async for _ in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            # TODO: these are all 0 for now!
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None

            output = self._make_response_output_items(
                request, final_res["text"], tokenizer
            )

            # Calculate usage from actual output
            if isinstance(final_res, dict) and "meta_info" in final_res:
                num_prompt_tokens = final_res["meta_info"].get("prompt_tokens", 0)
                num_generated_tokens = final_res["meta_info"].get("completion_tokens", 0)
                num_cached_tokens = final_res["meta_info"].get("cached_tokens", 0)
                num_reasoning_tokens = final_res["meta_info"].get("reasoning_tokens", 0)
            elif hasattr(final_res, "prompt_token_ids") and hasattr(
                final_res, "outputs"
            ):
                # Fallback calculation if meta_info not available
                num_prompt_tokens = (
                    len(final_res.prompt_token_ids) if final_res.prompt_token_ids else 0
                )
                num_generated_tokens = (
                    len(final_res.outputs[0].token_ids)
                    if final_res.outputs and final_res.outputs[0].token_ids
                    else 0
                )
                num_cached_tokens = getattr(final_res, "num_cached_tokens", 0)
                num_reasoning_tokens = 0
            else:
                # Final fallback
                num_prompt_tokens = 0
                num_generated_tokens = 0
                num_cached_tokens = 0
                num_reasoning_tokens = 0

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            reasoning_tokens=num_reasoning_tokens,
        )
        if self.enable_prompt_tokens_details and num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=num_cached_tokens
            )
        request_metadata.final_usage_info = usage

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status="completed",
            usage=usage,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response

        return response

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        tokenizer: Any,
    ):
        # Handle reasoning parsing if enabled, respecting chat_template_kwargs
        use_reasoning = bool(self.reasoning_parser)
        if use_reasoning and request.chat_template_kwargs:
            ctk = request.chat_template_kwargs
            if ctk.get("enable_thinking") is False or ctk.get("thinking") is False:
                use_reasoning = False
        if use_reasoning:
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                request=request,
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
        else:
            reasoning_content = None
            content = final_output

        # Parse tool calls from model output
        tool_calls = []
        function_tools = _normalize_function_tools_for_chat_parser(request.tools)
        if function_tools and self.tool_call_parser and content:
            parser = FunctionCallParser(function_tools, self.tool_call_parser)
            if parser.has_tool_call(content):
                try:
                    content, call_info_list = parser.parse_non_stream(content)
                    for call_info in call_info_list:
                        tool_calls.append(call_info)
                except Exception as e:
                    logger.warning(f"Tool call parsing failed: {e}")

        output_items = []
        if reasoning_content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=[],
                content=[
                    ResponseReasoningTextContent(
                        type="reasoning_text", text=reasoning_content
                    ),
                ],
                status=None,
            )
            output_items.append(reasoning_item)
        if tool_calls:
            import uuid
            for call_info in tool_calls:
                output_items.append(ResponseFunctionToolCall(
                    type="function_call",
                    id=f"fc_{random_uuid()}",
                    call_id=f"call_{uuid.uuid4().hex[:24]}",
                    name=call_info.name,
                    arguments=call_info.parameters,
                    status="completed",
                ))
        if content and content.strip():
            output_text = ResponseOutputText(
                text=content,
                annotations=[],
                type="output_text",
                logprobs=None,
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
        return output_items

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ):
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    def _construct_input_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse] = None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": request.instructions,
                }
            )

        # Prepend the conversation history
        if prev_response is not None:
            # Add the previous messages
            prev_msg = self.msg_store[prev_response.id]
            messages.extend(prev_msg)

            # Add the previous output
            for output_item in prev_response.output:
                # NOTE: We skip the reasoning output of the previous response
                if isinstance(output_item, ResponseReasoningItem):
                    continue
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "system",
                            "content": request.instructions,
                        }
                    )

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for item in request.input:
                converted = self._convert_responses_input_item(item)
                # Merge function_call tool_calls into a preceding assistant
                # message so the chat template sees one turn with content +
                # tool_calls, matching the Chat Completions format.
                if (
                    converted.get("role") == "assistant"
                    and converted.get("tool_calls")
                    and not converted.get("content")
                    and messages
                    and messages[-1].get("role") == "assistant"
                    and not messages[-1].get("tool_calls")
                ):
                    messages[-1]["tool_calls"] = converted["tool_calls"]
                else:
                    messages.append(converted)
        return messages

    @staticmethod
    def _convert_responses_content_part(part: dict) -> dict:
        """Convert Responses API content parts to Chat Completions format."""
        t = part.get("type", "")
        if t == "input_image":
            url = part.get("image_url", "")
            return {"type": "image_url", "image_url": {"url": url}}
        if t in {"input_text", "output_text", "text"}:
            return {"type": "text", "text": part.get("text", "")}
        if t == "input_audio":
            return {"type": "audio_url", "audio_url": {"url": part.get("audio_url", "")}}
        if t == "input_video":
            return {"type": "video_url", "video_url": {"url": part.get("video_url", "")}}
        return part

    @staticmethod
    def _convert_responses_text_parts_to_string(content: list[Any]) -> Optional[str]:
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                return None
            if part.get("type") not in {"input_text", "output_text", "text"}:
                return None
            text_parts.append(part.get("text", ""))
        return "".join(text_parts)

    @classmethod
    def _convert_responses_input_item(cls, item: Any) -> ChatCompletionMessageParam:
        """Convert a Responses API input item into Chat Completions message format."""
        msg = (
            item
            if isinstance(item, dict)
            else item.model_dump(exclude_none=True)
            if hasattr(item, "model_dump")
            else dict(item)
        )
        msg_type = msg.get("type") or "message"

        if msg_type == "function_call":
            arguments = msg.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": msg["call_id"],
                        "type": "function",
                        "function": {
                            "name": msg["name"],
                            "arguments": arguments,
                        },
                    }
                ],
            }

        if msg_type == "function_call_output":
            output = msg.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            return {
                "role": "tool",
                "content": output,
                "tool_call_id": msg["call_id"],
            }

        if msg_type == "reasoning":
            reasoning_text = ""
            content = msg.get("content") or []
            if content:
                first_part = content[0]
                if isinstance(first_part, dict):
                    reasoning_text = first_part.get("text", "")
            elif msg.get("summary"):
                first_summary = msg["summary"][0]
                if isinstance(first_summary, dict):
                    reasoning_text = first_summary.get("text", "")
            return {
                "role": "assistant",
                "reasoning_content": reasoning_text,
                "content": None,
            }

        if msg_type == "message":
            msg.pop("type", None)
            msg.pop("id", None)
            msg.pop("status", None)

        role = msg.get("role")
        content = msg.get("content")
        if role in {"assistant", "system", "developer"} and isinstance(content, list):
            text = cls._convert_responses_text_parts_to_string(content)
            if text is not None:
                msg["content"] = text
                return msg

        if isinstance(content, list):
            msg["content"] = [
                cls._convert_responses_content_part(part)
                for part in content
            ]
        return msg

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list["OpenAIMessage"]:
        messages: list["OpenAIMessage"] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browser = (
                "web_search_preview" in tool_types and self.tool_server is not None
            )
            enable_code_interpreter = (
                "code_interpreter" in tool_types and self.tool_server is not None
            )
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                browser_description=(
                    self.tool_server.get_tool_description("browser")
                    if self.tool_server and enable_browser
                    else None
                ),
                python_description=(
                    self.tool_server.get_tool_description("python")
                    if self.tool_server and enable_code_interpreter
                    else None
                ),
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions, request.tools)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation.
            # FIXME: Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message.
            if (
                len(prev_msgs) > 0
                and hasattr(prev_msgs[-1], "channel")
                and prev_msgs[-1].channel == "final"
            ):  # type: ignore[union-attr]
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    if (
                        hasattr(prev_msgs[i], "channel")
                        and prev_msgs[i].channel == "final"
                    ):  # type: ignore[union-attr]
                        prev_final_msg_idx = i
                        break
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                del prev_msgs[prev_final_msg_idx + 1 :]
                for msg in recent_turn_msgs:
                    if (
                        hasattr(msg, "channel") and msg.channel != "analysis"
                    ):  # type: ignore[union-attr]
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                messages.append(parse_response_input(response_msg, prev_outputs))
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *args,
        **kwargs,
    ):
        try:
            # Update the status to "in_progress"
            async with self.response_store_lock:
                stored_response = self.response_store.get(request.request_id)
                assert stored_response is not None
                stored_response.status = "in_progress"

            response = await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time,
                *args,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(str(e))

        if isinstance(response, ORJSONResponse):
            # If the request has failed, update the status to "failed"
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message="Cannot cancel a synchronous response.",
                )

            # Update the status to "cancelled"
            response.status = "cancelled"

        # The response_id is the same as the rid used when submitting the request
        self.tokenizer_manager.abort_request(rid=response_id)

        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_invalid_id_error(self, response_id: str):
        return self.create_error_response(
            message=(
                f"Invalid 'response_id': '{response_id}'. "
                "Expected an ID that begins with 'resp'."
            ),
            err_type="invalid_request_error",
            param="response_id",
        )

    def _make_not_found_error(self, response_id: str):
        return self.create_error_response(
            message=f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[StreamingHarmonyContext],
        context: StreamingHarmonyContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            # Get event type from the event's type field if it exists
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n"
            )

        current_content_index = 0
        current_output_index = 0
        current_item_id = f"item_{random_uuid()}"
        sent_output_item_added = False

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )

        # State for non-Harmony (SimpleContext) streaming
        _prev_text = ""
        _simple_stream_started = False
        _simple_stream_finalized = False
        _simple_text = ""
        _simple_text_item_id = ""
        _simple_text_output_index = -1
        _simple_completed_output_items: dict[int, Any] = {}
        _simple_tool_call_states: dict[int, dict[str, Any]] = {}
        _simple_finish_reason = None
        _tool_parser = None
        function_tools = _normalize_function_tools_for_chat_parser(request.tools)
        if function_tools and self.tool_call_parser:
            _tool_parser = FunctionCallParser(function_tools, self.tool_call_parser)

        def _get_finish_reason_type(finish_reason):
            return (
                finish_reason.get("type")
                if isinstance(finish_reason, dict)
                else finish_reason
            )

        def _make_simple_text_content(text: str):
            return openai_responses_types.ResponseOutputText(
                type="output_text",
                text=text,
                annotations=[],
                logprobs=None,
            )

        def _start_simple_text_item():
            nonlocal current_output_index
            nonlocal _simple_stream_started, _simple_text_item_id
            nonlocal _simple_text_output_index

            if _simple_stream_started:
                return []

            _simple_stream_started = True
            _simple_text_item_id = f"msg_{random_uuid()}"
            _simple_text_output_index = current_output_index
            current_output_index += 1

            return [
                _send_event(
                    openai_responses_types.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=-1,
                        output_index=_simple_text_output_index,
                        item=openai_responses_types.ResponseOutputMessage(
                            id=_simple_text_item_id,
                            type="message",
                            role="assistant",
                            content=[],
                            status="in_progress",
                        ),
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseContentPartAddedEvent(
                        type="response.content_part.added",
                        sequence_number=-1,
                        output_index=_simple_text_output_index,
                        item_id=_simple_text_item_id,
                        content_index=current_content_index,
                        part=_make_simple_text_content(""),
                    )
                ),
            ]

        def _finish_simple_text_item(status: str = "completed"):
            nonlocal _simple_text
            nonlocal _simple_stream_started, _simple_text_item_id
            nonlocal _simple_text_output_index

            if not _simple_stream_started:
                return []

            text_content = _make_simple_text_content(_simple_text)
            message = openai_responses_types.ResponseOutputMessage(
                id=_simple_text_item_id,
                type="message",
                role="assistant",
                content=[text_content],
                status=status,
            )
            _simple_completed_output_items[_simple_text_output_index] = message

            events = [
                _send_event(
                    openai_responses_types.ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=-1,
                        output_index=_simple_text_output_index,
                        content_index=current_content_index,
                        text=_simple_text,
                        logprobs=[],
                        item_id=_simple_text_item_id,
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=-1,
                        item_id=_simple_text_item_id,
                        output_index=_simple_text_output_index,
                        content_index=current_content_index,
                        part=text_content,
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=_simple_text_output_index,
                        item=message,
                    )
                ),
            ]

            _simple_stream_started = False
            _simple_text = ""
            _simple_text_item_id = ""
            _simple_text_output_index = -1
            return events

        def _start_simple_tool_call(call_item):
            nonlocal current_output_index

            state = _simple_tool_call_states.get(call_item.tool_index)
            if state is not None:
                if call_item.name:
                    state["name"] = call_item.name
                return []

            state = {
                "id": f"fc_{random_uuid()}",
                "call_id": f"call_{uuid.uuid4().hex[:24]}",
                "name": call_item.name or "",
                "arguments": "",
                "output_index": current_output_index,
                "done": False,
            }
            current_output_index += 1
            _simple_tool_call_states[call_item.tool_index] = state

            return [
                _send_event(
                    openai_responses_types.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=-1,
                        output_index=state["output_index"],
                        item=ResponseFunctionToolCall(
                            type="function_call",
                            id=state["id"],
                            call_id=state["call_id"],
                            name=state["name"],
                            arguments="",
                            status="in_progress",
                        ),
                    )
                )
            ]

        def _append_simple_tool_call_arguments(call_item):
            if call_item.parameters is None or call_item.parameters == "":
                return []

            state = _simple_tool_call_states.get(call_item.tool_index)
            if state is None:
                events = _start_simple_tool_call(call_item)
                state = _simple_tool_call_states[call_item.tool_index]
            else:
                events = []

            state["arguments"] += call_item.parameters
            events.append(
                _send_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                        type="response.function_call_arguments.delta",
                        sequence_number=-1,
                        item_id=state["id"],
                        output_index=state["output_index"],
                        delta=call_item.parameters,
                    )
                )
            )
            return events

        def _finish_simple_tool_call_state(
            state: dict[str, Any], status: str = "completed"
        ):
            if state["done"]:
                return []

            state["done"] = True
            item = ResponseFunctionToolCall(
                type="function_call",
                id=state["id"],
                call_id=state["call_id"],
                name=state["name"],
                arguments=state["arguments"],
                status=status,
            )
            _simple_completed_output_items[state["output_index"]] = item

            return [
                _send_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=-1,
                        item_id=state["id"],
                        output_index=state["output_index"],
                        arguments=state["arguments"],
                        name=state["name"],
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=state["output_index"],
                        item=item,
                    )
                ),
            ]

        def _finish_simple_tool_calls(
            exclude_tool_index: Optional[int] = None, status: str = "completed"
        ):
            events = []
            for tool_index, state in sorted(
                _simple_tool_call_states.items(),
                key=lambda item: item[1]["output_index"],
            ):
                if tool_index == exclude_tool_index:
                    continue
                events.extend(_finish_simple_tool_call_state(state, status=status))
            return events

        def _finish_simple_stream_items(status: str = "completed"):
            events = []
            events.extend(_finish_simple_text_item(status=status))
            events.extend(_finish_simple_tool_calls(status=status))
            return events

        async for ctx in result_generator:
            # Non-Harmony path: SimpleContext streams token-by-token
            if not hasattr(ctx, "is_expecting_start"):
                if ctx.last_output is None:
                    continue
                text = ctx.last_output.get("text", "")
                delta = text[len(_prev_text) :]
                _prev_text = text
                finish_reason = ctx.last_output.get("meta_info", {}).get(
                    "finish_reason"
                )
                if finish_reason:
                    _simple_finish_reason = finish_reason

                if delta:
                    if _tool_parser:
                        normal_text, calls = _tool_parser.parse_stream_chunk(delta)
                    else:
                        normal_text, calls = delta, []

                    if normal_text:
                        for event in _finish_simple_tool_calls():
                            yield event
                        for event in _start_simple_text_item():
                            yield event
                        _simple_text += normal_text
                        yield _send_event(
                            openai_responses_types.ResponseTextDeltaEvent(
                                type="response.output_text.delta",
                                sequence_number=-1,
                                content_index=current_content_index,
                                output_index=_simple_text_output_index,
                                item_id=_simple_text_item_id,
                                delta=normal_text,
                                logprobs=[],
                            )
                        )
                    for call_item in calls:
                        if call_item.name:
                            for event in _finish_simple_text_item():
                                yield event
                            if call_item.tool_index not in _simple_tool_call_states:
                                for event in _finish_simple_tool_calls():
                                    yield event
                            for event in _start_simple_tool_call(call_item):
                                yield event
                        for event in _append_simple_tool_call_arguments(call_item):
                            yield event

                if finish_reason and not _simple_stream_finalized:
                    _simple_stream_finalized = True
                    status = (
                        "incomplete"
                        if _get_finish_reason_type(finish_reason) == "length"
                        else "completed"
                    )
                    for event in _finish_simple_stream_items(status=status):
                        yield event
                continue

            if ctx.is_expecting_start():
                current_output_index += 1
                sent_output_item_added = False

                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    if previous_item.recipient is not None:
                        # Deal with tool call here
                        pass
                    elif previous_item.channel == "analysis":
                        reasoning_item = ResponseReasoningItem(
                            id=f"rs_{random_uuid()}",
                            type="reasoning",
                            summary=[],
                            content=[
                                ResponseReasoningTextContent(
                                    text=previous_item.content[0].text,
                                    type="reasoning_text",
                                ),
                            ],
                            status="completed",
                        )
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDoneEvent(
                                type="response.reasoning_text.done",
                                item_id=current_item_id,
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=reasoning_item,
                            )
                        )
                    elif previous_item.channel == "final":
                        text_content = openai_responses_types.ResponseOutputText(
                            type="output_text",
                            text=previous_item.content[0].text,
                            annotations=[],
                        )
                        yield _send_event(
                            openai_responses_types.ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                                logprobs=[],
                                item_id=current_item_id,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=text_content,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                    status="completed",
                                ),
                            )
                        )

            if ctx.parser.last_content_delta:
                if (
                    ctx.parser.current_channel == "final"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=ctx.parser.last_content_delta,
                            # TODO, use logprobs from ctx.last_request_output
                            logprobs=[],
                        )
                    )
                elif (
                    ctx.parser.current_channel == "analysis"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                # TODO: migrate this to
                                # ResponseReasoningTextContent for now
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    # TODO: migrate to OpenAI types once updated.
                    yield _send_event(
                        openai_responses_types.ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=ctx.parser.last_content_delta,
                            sequence_number=-1,
                        )
                    )

            if ctx.is_assistant_action_turn() and len(ctx.parser.messages) > 0:
                previous_item = ctx.parser.messages[-1]
                if (
                    self.supports_browsing
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("browser.")
                ):
                    function_name = previous_item.recipient[len("browser.") :]
                    action = None
                    parsed_args = orjson.loads(previous_item.content[0].text)
                    if function_name == "search":
                        action = openai_responses_types.response_function_web_search.ActionSearch(
                            type="search",
                            query=parsed_args["query"],
                        )
                    elif function_name == "open":
                        action = openai_responses_types.response_function_web_search.ActionOpenPage(
                            type="open_page",
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    elif function_name == "find":
                        action = openai_responses_types.response_function_web_search.ActionFind(
                            type="find",
                            pattern=parsed_args["pattern"],
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    else:
                        raise ValueError(f"Unknown function name: {function_name}")

                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.response_function_web_search.ResponseFunctionWebSearch(
                                # TODO: generate a unique id for web search call
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallInProgressEvent(
                            type="response.web_search_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallSearchingEvent(
                            type="response.web_search_call.searching",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )

                    # enqueue
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallCompletedEvent(
                            type="response.web_search_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseFunctionWebSearch(
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="completed",
                            ),
                        )
                    )

                if (
                    self.supports_code_interpreter
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("python")
                ):
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code="",
                                container_id="auto",
                                outputs=[],
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInProgressEvent(
                            type="response.code_interpreter_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    # TODO: do we need to add delta event here?
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCodeDoneEvent(
                            type="response.code_interpreter_call_code.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            code=previous_item.content[0].text,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInterpretingEvent(
                            type="response.code_interpreter_call.interpreting",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCompletedEvent(
                            type="response.code_interpreter_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code=previous_item.content[0].text,
                                container_id="auto",
                                # TODO: add outputs here
                                outputs=[],
                                status="completed",
                            ),
                        )
                    )

        async def empty_async_generator():
            if False:
                yield

        final_response = await self.responses_full_generator(
            request,
            sampling_params,
            empty_async_generator(),
            context,
            model_name,
            tokenizer,
            request_metadata,
            created_time=created_time,
        )
        if _simple_completed_output_items:
            final_response.output = [
                item for _, item in sorted(_simple_completed_output_items.items())
            ]
            if request.store:
                async with self.response_store_lock:
                    self.response_store[final_response.id] = final_response

        # Convert final_response to the format expected by ResponseCompletedEvent
        response_dict = final_response.model_dump()

        # Convert UsageInfo to ResponseUsage format
        if response_dict.get("usage"):
            response_dict["usage"] = _convert_usage_info_to_response_usage(
                response_dict["usage"]
            )

        finish_reason_type = _get_finish_reason_type(_simple_finish_reason)
        if finish_reason_type == "length":
            response_dict["status"] = "incomplete"
            response_dict["incomplete_details"] = {"reason": "max_output_tokens"}
            final_response.status = "incomplete"
            final_response.incomplete_details = response_dict["incomplete_details"]
            if request.store:
                async with self.response_store_lock:
                    self.response_store[final_response.id] = final_response

            yield _send_event(
                openai_responses_types.ResponseIncompleteEvent(
                    type="response.incomplete",
                    sequence_number=-1,
                    response=response_dict,
                )
            )
            return

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: ConversationContext,
        raw_request: Optional[Request] = None,
        priority: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate with builtin tool support for harmony-based models."""
        orig_priority = priority or 0

        while True:
            # Generate using SGLang's tokenizer manager
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )

            async for res in generator:
                context.append_output(res)
                # NOTE(woosuk): The stop condition is handled by the engine.
                yield context

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_output(tool_output)

            # Prepare for the next generation turn
            # Render the updated conversation for the next completion
            prompt_token_ids = context.render_for_completion()

            # Update the adapted request with new prompt
            adapted_request = GenerateReqInput(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                stream=adapted_request.stream,
                rid=request_id,
                extra_key=adapted_request.extra_key,
                return_logprob=adapted_request.return_logprob,
                logprob_start_len=adapted_request.logprob_start_len,
                top_logprobs_num=adapted_request.top_logprobs_num,
                return_text_in_logprobs=adapted_request.return_text_in_logprobs,
                return_hidden_states=adapted_request.return_hidden_states,
                background=adapted_request.background,
            )

            # Update sampling params with reduced max_tokens
            if hasattr(sampling_params, "max_new_tokens") or isinstance(
                sampling_params, dict
            ):
                context_len = getattr(
                    self.tokenizer_manager.model_config, "context_len", 4096
                )
                num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                remaining_tokens = (
                    context_len - len(prompt_token_ids) - num_reserved_tokens
                )

                if isinstance(sampling_params, dict):
                    sampling_params["max_new_tokens"] = max(remaining_tokens, 1)
                else:
                    sampling_params.max_new_tokens = max(remaining_tokens, 1)

            # Slightly reduce priority for subsequent tool calls
            priority = orig_priority - 1
