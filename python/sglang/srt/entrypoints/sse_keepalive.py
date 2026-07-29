"""Opt-in SSE comment keep-alives for long-running streaming requests."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from sglang.srt.environ import envs

_T = TypeVar("_T")
_MISSING = object()

# A comment is valid SSE, carries no application event, and is ignored by
# compliant clients. Keep the payload constant to avoid per-request allocation.
SSE_KEEPALIVE_COMMENT = ": keep-alive\n\n"


def sse_keepalive_interval() -> float:
    """Return the configured interval, or zero when keep-alives are disabled."""
    return _normalize_interval(envs.SGLANG_SSE_KEEPALIVE_INTERVAL.get())


def _normalize_interval(interval: float) -> float:
    return interval if math.isfinite(interval) and interval > 0 else 0.0


@dataclass(slots=True)
class PrimedSSEStream(Generic[_T]):
    """An iterator whose first item is ready or already being awaited."""

    iterator: AsyncIterator[_T]
    interval: float
    first_item: object = _MISSING
    pending_next: asyncio.Task[_T] | None = None


async def prime_sse_stream(
    source: AsyncIterator[_T],
    *,
    interval: float | None = None,
) -> PrimedSSEStream[_T]:
    """Wait for an immediate first item while retaining a long-running await.

    With keep-alives disabled this preserves the historical behavior: the
    caller receives the first item (or exception) before committing HTTP 200.
    With keep-alives enabled, immediate validation errors still retain their
    HTTP status, while requests that take longer than one interval can start
    an SSE response with a comment instead of waiting indefinitely.
    """
    iterator = source.__aiter__()
    resolved_interval = (
        sse_keepalive_interval() if interval is None else _normalize_interval(interval)
    )
    if resolved_interval == 0:
        return PrimedSSEStream(
            iterator=iterator,
            interval=0.0,
            first_item=await iterator.__anext__(),
        )

    pending_next = asyncio.create_task(iterator.__anext__())
    try:
        done, _ = await asyncio.wait({pending_next}, timeout=resolved_interval)
        if done:
            return PrimedSSEStream(
                iterator=iterator,
                interval=resolved_interval,
                first_item=pending_next.result(),
            )
        return PrimedSSEStream(
            iterator=iterator,
            interval=resolved_interval,
            pending_next=pending_next,
        )
    except BaseException:
        # The pending ``anext`` belongs to this priming call until ownership is
        # transferred in the returned ``PrimedSSEStream``. In particular, an
        # ASGI request cancellation here must not leave generation running.
        await _cancel_pending(pending_next)
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
        raise


async def _cancel_pending(task: asyncio.Task[object] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    # Await completed tasks too: a StopAsyncIteration (or another exception)
    # that races with downstream cancellation must still be retrieved.
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def stream_sse_with_keepalives(
    source: AsyncIterator[_T],
    *,
    interval: float | None = None,
    primed: PrimedSSEStream[_T] | None = None,
) -> AsyncGenerator[_T | str, None]:
    """Yield source items immediately and an SSE comment during each silence.

    Only one ``anext`` call is live at a time. A timeout never cancels that
    call, so the real chunk wins as soon as it is available and comments do
    not perturb or delay the upstream generator.
    """
    if primed is not None:
        iterator = primed.iterator
        resolved_interval = primed.interval
        first_item = primed.first_item
        pending_next = primed.pending_next
    else:
        iterator = source.__aiter__()
        resolved_interval = (
            sse_keepalive_interval()
            if interval is None
            else _normalize_interval(interval)
        )
        first_item = _MISSING
        pending_next = None

    try:
        if resolved_interval == 0:
            if first_item is not _MISSING:
                yield first_item  # type: ignore[misc]
            async for item in iterator:
                yield item
            return

        if first_item is not _MISSING:
            yield first_item  # type: ignore[misc]
        elif pending_next is not None:
            # ``prime_sse_stream`` already waited a full interval.
            if not pending_next.done():
                yield SSE_KEEPALIVE_COMMENT

        while True:
            if pending_next is None:
                pending_next = asyncio.create_task(iterator.__anext__())

            done, _ = await asyncio.wait(
                {pending_next},
                timeout=resolved_interval,
            )
            if not done:
                yield SSE_KEEPALIVE_COMMENT
                continue

            try:
                item = pending_next.result()
            except StopAsyncIteration:
                return
            pending_next = None
            yield item
    finally:
        await _cancel_pending(pending_next)
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
