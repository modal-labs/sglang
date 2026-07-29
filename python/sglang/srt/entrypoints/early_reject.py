"""Opt-in HTTP admission control for generation API endpoints."""

import asyncio
from http import HTTPStatus
from typing import Optional

from fastapi.responses import ORJSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

GENERATION_API_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/responses",
    }
)

_STATE_ATTRIBUTE = "api_early_reject_gate"


class APIEarlyRejectGate:
    """A non-blocking, per-HTTP-worker concurrency gate."""

    def __init__(self, max_concurrent_requests: int):
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        self.max_concurrent_requests = max_concurrent_requests
        self._active_requests = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active_requests >= self.max_concurrent_requests:
                return False
            self._active_requests += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active_requests <= 0:
                raise RuntimeError("API concurrency gate released without a permit")
            self._active_requests -= 1

    @property
    def active_requests(self) -> int:
        return self._active_requests


def configure_api_early_reject(app, max_concurrent_requests: Optional[int]) -> None:
    """Configure the gate before the HTTP server starts accepting requests."""
    gate = (
        APIEarlyRejectGate(max_concurrent_requests)
        if max_concurrent_requests is not None
        else None
    )
    setattr(app.state, _STATE_ATTRIBUTE, gate)


def _overload_response(path: str) -> ORJSONResponse:
    message = (
        "The server is at its configured concurrent request limit. "
        "Please retry shortly."
    )
    if path == "/v1/messages":
        content = {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": message,
            },
        }
    else:
        content = {
            "error": {
                "message": message,
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        }
    return ORJSONResponse(
        status_code=HTTPStatus.TOO_MANY_REQUESTS,
        headers={"Retry-After": "1"},
        content=content,
    )


class APIEarlyRejectMiddleware:
    """Reject excess generation requests before parsing or scheduler enqueue.

    A permit is held until the ASGI application finishes sending the response
    body. This includes the complete lifetime of an SSE stream, so an overload
    response is a real HTTP 429 rather than an error event inside an HTTP 200
    stream.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in GENERATION_API_PATHS
        ):
            await self.app(scope, receive, send)
            return

        fastapi_app = scope.get("app")
        gate = (
            getattr(fastapi_app.state, _STATE_ATTRIBUTE, None)
            if fastapi_app is not None
            else None
        )
        if gate is None:
            await self.app(scope, receive, send)
            return

        if not await gate.try_acquire():
            response = _overload_response(scope["path"])
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await gate.release()
