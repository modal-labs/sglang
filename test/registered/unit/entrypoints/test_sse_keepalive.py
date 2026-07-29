import asyncio
import unittest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.sse_keepalive import (  # noqa: E402
    SSE_KEEPALIVE_COMMENT,
    prime_sse_stream,
    stream_sse_with_keepalives,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class SSEKeepaliveTest(unittest.TestCase):
    def test_disabled_is_a_transparent_passthrough(self):
        async def run():
            async def source():
                yield "data: one\n\n"
                yield "data: two\n\n"

            return [
                item async for item in stream_sse_with_keepalives(source(), interval=0)
            ]

        self.assertEqual(
            asyncio.run(run()),
            ["data: one\n\n", "data: two\n\n"],
        )

    def test_disabled_stream_close_closes_source(self):
        async def run():
            source_closed = asyncio.Event()

            async def source():
                try:
                    yield "data: one\n\n"
                    await asyncio.Event().wait()
                finally:
                    source_closed.set()

            stream = stream_sse_with_keepalives(source(), interval=0)
            self.assertEqual(await stream.__anext__(), "data: one\n\n")
            await stream.aclose()
            return source_closed.is_set()

        self.assertTrue(asyncio.run(run()))

    def test_immediate_chunk_precedes_keepalive(self):
        async def run():
            release = asyncio.Event()

            async def source():
                yield "data: ready\n\n"
                await release.wait()

            stream = stream_sse_with_keepalives(source(), interval=0.01)
            first = await stream.__anext__()
            second = await asyncio.wait_for(stream.__anext__(), timeout=0.1)
            await stream.aclose()
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first, "data: ready\n\n")
        self.assertEqual(second, SSE_KEEPALIVE_COMMENT)

    def test_silent_primed_stream_starts_with_keepalive(self):
        async def run():
            release = asyncio.Event()

            async def source():
                await release.wait()
                yield "data: delayed\n\n"

            generator = source()
            primed = await prime_sse_stream(generator, interval=0.01)
            stream = stream_sse_with_keepalives(generator, primed=primed)
            comment = await asyncio.wait_for(stream.__anext__(), timeout=0.1)
            release.set()
            chunk = await asyncio.wait_for(stream.__anext__(), timeout=0.1)
            await stream.aclose()
            return comment, chunk

        comment, chunk = asyncio.run(run())
        self.assertEqual(comment, SSE_KEEPALIVE_COMMENT)
        self.assertEqual(chunk, "data: delayed\n\n")

    def test_fast_priming_surfaces_exception_before_streaming(self):
        async def run():
            async def source():
                raise ValueError("invalid request")
                yield  # pragma: no cover

            await prime_sse_stream(source(), interval=1)

        with self.assertRaisesRegex(ValueError, "invalid request"):
            asyncio.run(run())

    def test_chunk_that_wins_after_priming_is_not_delayed_by_comment(self):
        async def run():
            release = asyncio.Event()

            async def source():
                await release.wait()
                yield "data: won-race\n\n"

            generator = source()
            primed = await prime_sse_stream(generator, interval=0.01)
            release.set()
            await asyncio.sleep(0)
            stream = stream_sse_with_keepalives(generator, primed=primed)
            first = await asyncio.wait_for(stream.__anext__(), timeout=0.1)
            await stream.aclose()
            return first

        self.assertEqual(asyncio.run(run()), "data: won-race\n\n")

    def test_closing_stream_cancels_pending_source_read(self):
        async def run():
            release = asyncio.Event()
            source_closed = asyncio.Event()

            async def source():
                try:
                    await release.wait()
                    yield "unreachable"
                finally:
                    source_closed.set()

            stream = stream_sse_with_keepalives(source(), interval=0.01)
            self.assertEqual(await stream.__anext__(), SSE_KEEPALIVE_COMMENT)
            await stream.aclose()
            return source_closed.is_set()

        self.assertTrue(asyncio.run(run()))

    def test_cancelling_prime_cancels_and_closes_source(self):
        async def run():
            source_started = asyncio.Event()
            source_closed = asyncio.Event()

            async def source():
                try:
                    source_started.set()
                    await asyncio.Event().wait()
                    yield "unreachable"
                finally:
                    source_closed.set()

            prime_task = asyncio.create_task(prime_sse_stream(source(), interval=60))
            await source_started.wait()
            prime_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await prime_task
            return source_closed.is_set()

        self.assertTrue(asyncio.run(run()))

    def test_comment_is_a_legal_payload_free_sse_frame(self):
        self.assertTrue(SSE_KEEPALIVE_COMMENT.startswith(":"))
        self.assertTrue(SSE_KEEPALIVE_COMMENT.endswith("\n\n"))
        self.assertNotIn("data:", SSE_KEEPALIVE_COMMENT)
        self.assertNotIn("event:", SSE_KEEPALIVE_COMMENT)


if __name__ == "__main__":
    unittest.main()
