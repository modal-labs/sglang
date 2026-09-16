import ast
import asyncio
import importlib.util
import threading
import unittest
from time import monotonic
from types import SimpleNamespace
from pathlib import Path

path = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/entrypoints/openai/request_conversion.py"
)
spec = importlib.util.spec_from_file_location("request_conversion", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConversionTest(unittest.IsolatedAsyncioTestCase):
    async def test_result_and_error_propagation(self):
        executor = module.RequestConversionExecutor(1)
        self.assertEqual(await executor.run(lambda a, b: a + b, 2, 3), 5)

        def fail():
            raise ValueError("invalid request")

        with self.assertRaisesRegex(ValueError, "invalid request"):
            await executor.run(fail)
        self.assertEqual(await executor.run(lambda: 7), 7)

    async def test_cancelled_waiter_keeps_worker_slot_until_completion(self):
        executor = module.RequestConversionExecutor(1)
        release = threading.Event()
        started = asyncio.Event()
        second_started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def first():
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            raise ValueError("detached failure")

        def second():
            loop.call_soon_threadsafe(second_started.set)
            return 2

        first_task = asyncio.create_task(executor.run(first))
        await asyncio.wait_for(started.wait(), 2)
        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task
        second_task = asyncio.create_task(executor.run(second))
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(second_started.wait(), 0.05)
        finally:
            release.set()
        self.assertEqual(await asyncio.wait_for(second_task, 2), 2)

    async def test_cancellation_while_waiting_does_not_leak_capacity(self):
        executor = module.RequestConversionExecutor(1)
        await executor._slots.acquire()
        task = asyncio.create_task(executor.run(lambda: 1))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        executor._slots.release()
        self.assertEqual(await asyncio.wait_for(executor.run(lambda: 3), 1), 3)

    async def test_handler_offloads_schema_validation_and_preserves_error(self):
        # Execute the actual frontend method without loading GPU dependencies.
        source = path.with_name("serving_base.py")
        tree = ast.parse(source.read_text())
        klass = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingBase"
        )
        method = next(
            n
            for n in klass.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_request"
        )
        unit = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        scope = {"monotonic_time": monotonic}
        exec(compile(ast.fix_missing_locations(unit), str(source), "exec"), scope)
        main_thread = threading.get_ident()
        visited = []

        def validate(request):
            visited.append(threading.get_ident())
            return "invalid schema"

        handler = SimpleNamespace(
            request_conversion_executor=module.RequestConversionExecutor(1),
            _validate_request=validate,
            create_error_response=lambda message: {"error": message},
        )
        result = await scope["handle_request"](handler, object(), None)
        self.assertEqual(result, {"error": "invalid schema"})
        self.assertEqual(len(visited), 1)
        self.assertNotEqual(visited[0], main_thread)

    def test_invalid_concurrency(self):
        with self.assertRaises(ValueError):
            module.RequestConversionExecutor(0)


if __name__ == "__main__":
    unittest.main()
