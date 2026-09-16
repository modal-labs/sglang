"""CPU lifecycle tests for native Responses PD turns."""

import asyncio
import importlib.util
import unittest
from dataclasses import dataclass, field
from pathlib import Path

_path = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/entrypoints/openai/pd_responses.py"
)
_spec = importlib.util.spec_from_file_location("pd_responses_under_test", _path)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


@dataclass
class Request:
    rid: str = "resp_test"
    input_ids: list = field(default_factory=lambda: [1, 2, 3])
    sampling_params: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 64,
            "min_new_tokens": 8,
            "top_p": 0.95,
        }
    )
    image_data: list = field(default_factory=lambda: ["data:image/png;base64,abc"])
    stream: bool = True
    background: bool = True
    bootstrap_host: str = None
    bootstrap_port: int = None
    bootstrap_room: int = None


class PrepareTest(unittest.TestCase):
    def test_native_identity_sampling_and_media_preserved(self):
        request = Request()
        payload = module.prepare_prefill_turn(request, "http://10.0.0.1:8000", 8998)
        self.assertEqual(request.rid, "resp_test")
        self.assertNotEqual(payload["rid"], request.rid)
        self.assertEqual(payload["sampling_params"], request.sampling_params)
        self.assertEqual(payload["image_data"], request.image_data)
        self.assertEqual(payload["input_ids"], request.input_ids)
        self.assertEqual(payload["bootstrap_room"], request.bootstrap_room)
        self.assertEqual(payload["bootstrap_host"], "10.0.0.1")
        self.assertEqual(payload["bootstrap_port"], 8998)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["background"])
        payload["input_ids"].append(4)
        self.assertEqual(request.input_ids, [1, 2, 3])

    def test_tool_turns_get_distinct_prefill_identity_and_room(self):
        request = Request()
        first = module.prepare_prefill_turn(request, "http://worker:8000", 8998)
        second = module.prepare_prefill_turn(request, "http://worker:8000", 8998)
        self.assertNotEqual(first["rid"], second["rid"])
        self.assertNotEqual(first["bootstrap_room"], second["bootstrap_room"])
        self.assertEqual(request.rid, "resp_test")

    def test_url_must_be_trusted_origin(self):
        for url in (
            "file:///tmp/foo",
            "http://user:pass@host",
            "http://host/generate",
            "http://host?x=1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                module.prepare_prefill_turn(Request(), url, 8998)

    def test_process_local_media_fails_before_admission(self):
        request = Request(image_data=[object()])
        with self.assertRaises(TypeError):
            module.prepare_prefill_turn(request, "http://worker", 8998)


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_prefill_failure_aborts_waiting_decode(self):
        cleaned = asyncio.Event()
        aborted = []

        async def decode():
            try:
                await asyncio.Event().wait()
                yield 1
            finally:
                cleaned.set()

        async def prefill():
            await asyncio.sleep(0)
            raise ValueError("prefill failed")

        gen = module.coordinated_turn(
            Request(), decode(), asyncio.create_task(prefill()), aborted.append
        )
        with self.assertRaisesRegex(ValueError, "prefill failed"):
            await asyncio.wait_for(anext(gen), 1)
        self.assertTrue(cleaned.is_set())
        self.assertEqual(aborted, ["resp_test"])

    async def test_disconnect_cancels_both_legs(self):
        p_cleaned = asyncio.Event()
        d_cleaned = asyncio.Event()
        aborted = []

        async def prefill():
            try:
                await asyncio.Event().wait()
            finally:
                p_cleaned.set()

        async def decode():
            try:
                yield "first"
                await asyncio.Event().wait()
            finally:
                d_cleaned.set()

        p = asyncio.create_task(prefill())
        gen = module.coordinated_turn(Request(), decode(), p, aborted.append)
        self.assertEqual(await anext(gen), "first")
        await gen.aclose()
        self.assertTrue(p_cleaned.is_set())
        self.assertTrue(d_cleaned.is_set())
        self.assertEqual(aborted, ["resp_test"])

    async def test_decode_completion_does_not_wait_for_prefill_http_drain(self):
        drain = asyncio.Event()
        aborted = []

        async def decode():
            yield "done"

        p = asyncio.create_task(drain.wait())
        gen = module.coordinated_turn(Request(), decode(), p, aborted.append)
        outputs = await asyncio.wait_for(self.collect(gen), 1)
        self.assertEqual(outputs, ["done"])
        self.assertEqual(aborted, [])
        self.assertFalse(p.done())
        drain.set()
        await p

    async def test_cancellation_before_first_output_aborts_both(self):
        aborted = []

        async def decode():
            await asyncio.Event().wait()
            yield 1

        p = asyncio.create_task(asyncio.Event().wait())
        gen = module.coordinated_turn(Request(), decode(), p, aborted.append)
        waiter = asyncio.create_task(anext(gen))
        await asyncio.sleep(0.01)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertTrue(p.cancelled())
        self.assertEqual(aborted, ["resp_test"])

    async def test_prefill_success_before_decode_is_valid(self):
        async def decode():
            await asyncio.sleep(0.01)
            yield 1
            yield 2

        p = asyncio.create_task(asyncio.sleep(0))
        outputs = await self.collect(
            module.coordinated_turn(
                Request(), decode(), p, lambda rid: self.fail("unexpected abort")
            )
        )
        self.assertEqual(outputs, [1, 2])

    async def collect(self, gen):
        return [item async for item in gen]


if __name__ == "__main__":
    unittest.main()
