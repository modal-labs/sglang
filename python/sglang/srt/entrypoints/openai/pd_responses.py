"""Coordinate native Responses turns with a stateless prefill worker.

The decode process remains the sole owner of Responses storage, background
jobs and tool execution. Only raw generation inputs travel to prefill.
"""

import asyncio
import copy
import dataclasses
import json
import secrets
import uuid
from urllib.parse import urlsplit

import aiohttp

from sglang.srt.utils import ImageData, VideoData


class PDResponsesError(ValueError):
    """Keep a worker failure distinct from invalid client input."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def _media_reference_json(value):
    """Encode API media references, preserving their preprocessing options."""
    if isinstance(value, (ImageData, VideoData)):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_media_reference_json(item) for item in value]
    return value


def prepare_prefill_turn(request, prefill_url, bootstrap_port):
    parsed = urlsplit(prefill_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "PD Responses prefill URL must be a trusted HTTP worker origin"
        )
    request.bootstrap_host = parsed.hostname
    request.bootstrap_port = bootstrap_port
    request.bootstrap_room = secrets.randbits(63)
    payload = {
        field.name: copy.deepcopy(getattr(request, field.name))
        for field in dataclasses.fields(request)
        if field.init
    }
    # Keep D rid == response_id for native retrieve/cancel semantics. A unique P
    # id prevents late cleanup of an old tool turn from aborting its successor.
    payload["rid"] = f"{request.rid or 'responses'}-prefill-{uuid.uuid4().hex}"
    payload["stream"] = False
    payload["background"] = False
    # These are API media references before TokenizerManager preprocessing.
    # Validate serialization before either engine admits the turn; process-local
    # tensors are not a supported cross-node media transport.
    for field in ("image_data", "video_data"):
        if field in payload:
            payload[field] = _media_reference_json(payload[field])
    json.dumps(payload)
    return payload


async def run_prefill(prefill_url, payload, timeout):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as client:
        try:
            async with client.post(
                prefill_url.rstrip("/") + "/generate", json=payload
            ) as response:
                if response.status >= 400:
                    # Do not log raw request bodies or model output.
                    raise PDResponsesError(
                        f"PD prefill failed with HTTP {response.status}",
                        response.status,
                    )
                async for _ in response.content.iter_chunked(65536):
                    pass
        except asyncio.CancelledError:
            try:
                async with client.post(
                    prefill_url.rstrip("/") + "/abort_request",
                    json={"rid": payload["rid"], "abort_all": False},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    await response.read()
            except Exception:
                # Cleanup must not mask the consumer cancellation.
                pass
            raise
        except TimeoutError as exc:
            raise PDResponsesError("PD prefill worker timed out", 504) from exc
        except aiohttp.ClientError as exc:
            raise PDResponsesError("PD prefill worker transport failed", 502) from exc


async def coordinated_turn(request, decode_generator, prefill_task, abort_decode):
    """Fail D promptly if P fails, and cancel both legs when the consumer leaves."""
    iterator = decode_generator.__aiter__()
    next_output = None
    completed = False
    decode_aborted = False
    decode_started = False
    try:
        while True:
            next_output = asyncio.create_task(anext(iterator))
            pending = {next_output}
            if not decode_started:
                if prefill_task.done():
                    prefill_task.result()
                else:
                    pending.add(prefill_task)
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # A real D output proves transfer success. Prefer it over a
            # simultaneous P HTTP drain failure; after commit, only D owns
            # the client result. Before commit, P failure must release D.
            if not decode_started and prefill_task.done() and not next_output.done():
                prefill_task.result()
            try:
                output = await next_output
            except StopAsyncIteration:
                if not decode_started and prefill_task.done():
                    prefill_task.result()
                completed = decode_started and not decode_aborted
                break
            next_output = None
            # Streaming scheduler errors are yielded as terminal abort outputs,
            # not raised. An exhausted D iterator therefore does not by itself
            # prove P transferred successfully.
            finish_reason = (
                (output.get("meta_info") or {}).get("finish_reason") or {}
                if isinstance(output, dict)
                else {}
            )
            decode_aborted |= (
                isinstance(finish_reason, dict) and finish_reason.get("type") == "abort"
            )
            if not decode_aborted:
                decode_started = True
            yield output
    finally:
        if next_output is not None and not next_output.done():
            next_output.cancel()
            await asyncio.gather(next_output, return_exceptions=True)
        try:
            await iterator.aclose()
        finally:
            if not completed:
                abort_decode(request.rid)
                prefill_task.cancel()
                await asyncio.gather(prefill_task, return_exceptions=True)
        if completed:
            # D completion proves the transfer finished. P may still be draining
            # its HTTP response; do not add that drain to client latency.
            def consume(task):
                if not task.cancelled():
                    task.exception()

            prefill_task.add_done_callback(consume)


async def _routed_turn_stream(request, router_url):
    """Send an expanded Responses turn through the ordinary PD generation router.

    The frontend retains API state; the selected P/D pair owns generation.
    Use an internal streaming leg even for nonstreaming Responses so cancellation
    closes the worker streams promptly. The frontend still formats the public API.
    """
    payload = {
        field.name: copy.deepcopy(getattr(request, field.name))
        for field in dataclasses.fields(request)
        if field.init
    }
    payload["rid"] = f"{request.rid or 'responses'}-turn-{uuid.uuid4().hex}"
    payload["stream"] = True
    payload["background"] = False
    for name in ("bootstrap_host", "bootstrap_port", "bootstrap_room"):
        payload.pop(name, None)
    for name in ("image_data", "video_data"):
        if name in payload:
            payload[name] = _media_reference_json(payload[name])
    # No request bodies are logged or persisted by this transport.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)
    ) as client:
        try:
            async with client.post(
                router_url.rstrip("/") + "/generate", json=payload
            ) as response:
                if response.status >= 400:
                    raise PDResponsesError(
                        f"PD generation router returned HTTP {response.status}",
                        response.status,
                    )
                pending = b""
                last_output = None
                async for chunk in response.content.iter_chunked(65536):
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        line = line.rstrip(b"\r")
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:]
                        if data == b"[DONE]":
                            if last_output is None:
                                raise PDResponsesError("PD generation returned no output", 502)
                            if not request.stream:
                                yield last_output
                            return
                        output = json.loads(data)
                        if "error" in output:
                            raise PDResponsesError("PD generation stream failed", 502)
                        last_output = output
                        if request.stream:
                            yield output
                raise PDResponsesError("PD generation stream ended without DONE", 502)
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise PDResponsesError(
                "PD generation router transport failed", 502
            ) from exc


async def routed_turn(request, router_url, raw_request=None):
    """Keep frontend disconnection attached to the routed generation lifetime."""
    consumer = asyncio.current_task()

    async def watch_disconnect():
        while True:
            if await raw_request.is_disconnected():
                consumer.cancel()
                return
            await asyncio.sleep(0.25)

    watcher = (
        asyncio.create_task(watch_disconnect())
        if raw_request is not None and not getattr(request, "background", False)
        else None
    )
    stream = _routed_turn_stream(request, router_url)
    try:
        async for output in stream:
            yield output
    finally:
        try:
            await stream.aclose()
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
