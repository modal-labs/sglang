import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from sglang.srt.entrypoints.early_reject import (
    APIEarlyRejectGate,
    APIEarlyRejectMiddleware,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.auth import add_api_key_middleware
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

GENERATION_PATHS = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
]


class _BlockingResponseApp:
    def __init__(self, block_after_response_start: bool):
        self.block_after_response_start = block_after_response_start
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        if not self.block_after_response_start:
            self.entered.set()
            await self.release.wait()

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )

        if self.block_after_response_start:
            self.entered.set()
            await self.release.wait()

        await send({"type": "http.response.body", "body": b"done"})


def _scope(path: str, gate, *, authorization: str | None = None):
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
        "app": SimpleNamespace(state=SimpleNamespace(api_early_reject_gate=gate)),
    }


async def _receive():
    return {"type": "http.disconnect"}


async def _invoke(middleware, scope):
    messages = []

    async def send(message):
        messages.append(message)

    await middleware(scope, _receive, send)
    return messages


def _status(messages):
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def _body(messages):
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def _headers(messages):
    return dict(
        next(
            message["headers"]
            for message in messages
            if message["type"] == "http.response.start"
        )
    )


@pytest.mark.parametrize("path", GENERATION_PATHS)
def test_rejects_before_streaming_http_200(path):
    asyncio.run(_test_rejects_before_streaming_http_200(path))


async def _test_rejects_before_streaming_http_200(path):
    app = _BlockingResponseApp(block_after_response_start=True)
    middleware = APIEarlyRejectMiddleware(app)
    gate = APIEarlyRejectGate(1)
    scope = _scope(path, gate)

    first = asyncio.create_task(_invoke(middleware, scope))
    await app.entered.wait()

    rejected = await _invoke(middleware, scope)
    assert _status(rejected) == 503
    assert _headers(rejected)[b"retry-after"] == b"1"
    payload = json.loads(_body(rejected))
    if path == "/v1/messages":
        assert payload == {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": (
                    "The server is at its configured concurrent request limit. "
                    "Please retry shortly."
                ),
            },
        }
    else:
        assert payload["error"]["type"] == "server_error"
        assert payload["error"]["code"] == "service_unavailable"
    assert app.calls == 1

    app.release.set()
    assert _status(await first) == 200
    assert gate.active_requests == 0

    assert _status(await _invoke(middleware, scope)) == 200


@pytest.mark.parametrize("path", GENERATION_PATHS)
def test_rejects_while_non_streaming_request_is_active(path):
    asyncio.run(_test_rejects_while_non_streaming_request_is_active(path))


async def _test_rejects_while_non_streaming_request_is_active(path):
    app = _BlockingResponseApp(block_after_response_start=False)
    middleware = APIEarlyRejectMiddleware(app)
    gate = APIEarlyRejectGate(1)
    scope = _scope(path, gate)

    first = asyncio.create_task(_invoke(middleware, scope))
    await app.entered.wait()
    assert _status(await _invoke(middleware, scope)) == 503

    app.release.set()
    assert _status(await first) == 200
    assert gate.active_requests == 0


@pytest.mark.parametrize(
    ("active_path", "rejected_path"),
    [
        ("/v1/messages", "/v1/responses"),
        ("/v1/chat/completions", "/v1/messages"),
    ],
)
def test_openai_and_anthropic_share_one_budget(active_path, rejected_path):
    asyncio.run(_test_openai_and_anthropic_share_one_budget(active_path, rejected_path))


async def _test_openai_and_anthropic_share_one_budget(active_path, rejected_path):
    app = _BlockingResponseApp(block_after_response_start=True)
    middleware = APIEarlyRejectMiddleware(app)
    gate = APIEarlyRejectGate(1)

    active = asyncio.create_task(_invoke(middleware, _scope(active_path, gate)))
    await app.entered.wait()

    rejected = await _invoke(middleware, _scope(rejected_path, gate))
    assert _status(rejected) == 503
    assert app.calls == 1

    app.release.set()
    assert _status(await active) == 200
    assert gate.active_requests == 0


def test_disabled_gate_and_non_generation_paths_bypass_admission():
    asyncio.run(_test_disabled_gate_and_non_generation_paths_bypass_admission())


async def _test_disabled_gate_and_non_generation_paths_bypass_admission():
    app = _BlockingResponseApp(block_after_response_start=False)
    app.release.set()
    middleware = APIEarlyRejectMiddleware(app)

    disabled_scope = _scope("/v1/chat/completions", None)
    assert _status(await _invoke(middleware, disabled_scope)) == 200

    saturated_gate = APIEarlyRejectGate(1)
    assert await saturated_gate.try_acquire()
    for path in ("/v1/models", "/v1/messages/count_tokens"):
        assert _status(await _invoke(middleware, _scope(path, saturated_gate))) == 200
    await saturated_gate.release()


def test_releases_permit_when_downstream_raises():
    async def run():
        async def broken_app(scope, receive, send):
            raise RuntimeError("boom")

        gate = APIEarlyRejectGate(1)
        middleware = APIEarlyRejectMiddleware(broken_app)
        with pytest.raises(RuntimeError, match="boom"):
            await _invoke(middleware, _scope("/v1/responses", gate))
        assert gate.active_requests == 0

    asyncio.run(run())


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
def test_releases_permit_when_streaming_client_disconnects(path):
    async def run(path):
        first_body = asyncio.Event()
        disconnect = asyncio.Event()

        async def stream_body():
            yield b"data: first\n\n"
            await asyncio.Event().wait()

        async def streaming_app(scope, receive, send):
            await StreamingResponse(stream_body(), media_type="text/event-stream")(
                scope, receive, send
            )

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body.set()

        gate = APIEarlyRejectGate(1)
        middleware = APIEarlyRejectMiddleware(streaming_app)
        task = asyncio.create_task(middleware(_scope(path, gate), receive, send))

        await asyncio.wait_for(first_body.wait(), timeout=1)
        assert gate.active_requests == 1

        disconnect.set()
        await asyncio.wait_for(task, timeout=1)
        assert gate.active_requests == 0

    asyncio.run(run(path))


def test_authentication_runs_before_the_saturated_gate():
    async def run():
        app = FastAPI()
        gate = APIEarlyRejectGate(1)
        app.state.api_early_reject_gate = gate

        @app.post("/v1/messages")
        async def messages():
            return {"ok": True}

        # Production installs admission first and authentication later, making
        # authentication the outer ASGI middleware.
        app.add_middleware(APIEarlyRejectMiddleware)
        add_api_key_middleware(app, api_key="secret", admin_api_key=None)

        assert await gate.try_acquire()
        unauthorized = await _invoke(app, _scope("/v1/messages", gate))
        assert _status(unauthorized) == 401
        assert gate.active_requests == 1

        authorized = await _invoke(
            app,
            _scope("/v1/messages", gate, authorization="Bearer secret"),
        )
        assert _status(authorized) == 503
        assert gate.active_requests == 1
        await gate.release()

    asyncio.run(run())


def test_server_arg_requires_positive_limit():
    with pytest.raises(
        ValueError, match="api-early-reject-max-concurrency must be positive"
    ):
        ServerArgs(
            model_path="dummy",
            api_early_reject_max_concurrency=0,
        )


def test_api_early_reject_limit_is_exposed_on_cli():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model-path", "dummy", "--api-early-reject-max-concurrency", "16"]
    )
    assert args.api_early_reject_max_concurrency == 16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
