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


class PDResponsesError(ValueError):
    """Keep a worker failure distinct from invalid client input."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


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
