"""Exercise the deploy recipes' heartbeat-vs-drain lifecycle without Modal.

The helpers AST-load the container state machine and
terminate_unhealthy_container from the checked-in serve files and run them
against stub os/time/subprocess/modal namespaces, so a planned drain can be
injected at every point of the heartbeat teardown path.
"""

from __future__ import annotations

import ast
import threading
import types
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEPLOY_FILES = [
    REPO_ROOT / "deploy/instinct/serve.py",
    REPO_ROOT / "deploy/instinct/serve_probe.py",
]
LIFECYCLE_NAMES = (
    "_CONTAINER_STATE_LOCK",
    "_CONTAINER_STATE",
    "_reset_container_state",
    "_container_running",
    "_container_is_draining",
    "_set_container_stopping",
    "_begin_heartbeat_termination",
    "_exit_unless_draining",
    "terminate_unhealthy_container",
)


class _ClientClosed(Exception):
    pass


class _Harness:
    """Exec the lifecycle definitions with recording stubs for side effects."""

    def __init__(self, path: Path):
        self.path = path
        self.exits: list[int] = []
        self.sleeps: list[float] = []
        self.forensics_calls = 0
        self.container_stop_calls = 0
        self.on_forensics = lambda: None
        self.on_container_stop = lambda: None
        self.on_sleep = lambda: None

        def _exit(code: int) -> None:
            self.exits.append(code)
            raise _Exited()

        def _sleep(seconds: float) -> None:
            self.sleeps.append(seconds)
            self.on_sleep()

        def _run(*_args, **_kwargs) -> None:
            self.forensics_calls += 1
            self.on_forensics()

        def _stop_current_container() -> None:
            self.container_stop_calls += 1
            self.on_container_stop()

        self.ns = {
            "threading": threading,
            "os": types.SimpleNamespace(_exit=_exit),
            "time": types.SimpleNamespace(sleep=_sleep),
            "subprocess": types.SimpleNamespace(run=_run, SubprocessError=Exception),
            "modal": types.SimpleNamespace(
                exception=types.SimpleNamespace(ClientClosed=_ClientClosed)
            ),
            "stop_current_container": _stop_current_container,
            "print": lambda *_a, **_k: None,
        }
        tree = ast.parse(path.read_text())
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") in LIFECYCLE_NAMES
            )
            or (isinstance(node, ast.FunctionDef) and node.name in LIFECYCLE_NAMES)
        ]
        found = {
            n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes
        }
        missing = set(LIFECYCLE_NAMES) - found
        assert not missing, f"{path}: missing lifecycle definitions {sorted(missing)}"
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            self.ns,
        )

    def __getattr__(self, name):
        try:
            return self.ns[name]
        except KeyError:
            raise AttributeError(name) from None

    def terminate(self) -> None:
        try:
            self.ns["terminate_unhealthy_container"]()
        except _Exited:
            pass

    def state(self) -> str:
        return self.ns["_CONTAINER_STATE"]


class _Exited(BaseException):
    """Raised by the os._exit stub so control flow stops like the real call."""


class TestDeployContainerLifecycle(CustomTestCase):
    def _each(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                yield _Harness(path)

    def test_heartbeat_termination_begins_once_and_never_during_drain(self):
        for h in self._each():
            h._reset_container_state()
            self.assertTrue(h._container_running())
            self.assertTrue(h._begin_heartbeat_termination())
            self.assertFalse(h._begin_heartbeat_termination())
            self.assertFalse(h._container_running())
            self.assertEqual(h.state(), "heartbeat_terminating")

            h._reset_container_state()
            h._set_container_stopping()
            self.assertTrue(h._container_is_draining())
            self.assertFalse(h._begin_heartbeat_termination())
            self.assertEqual(h.state(), "stopping")

    def test_unhealthy_container_is_force_exited_without_drain(self):
        for h in self._each():
            h._reset_container_state()
            self.assertTrue(h._begin_heartbeat_termination())
            h.terminate()
            self.assertEqual(h.forensics_calls, 1)
            self.assertEqual(h.container_stop_calls, 1)
            self.assertEqual(len(h.sleeps), 60)
            self.assertEqual(h.exits, [1])

    def test_client_closed_exits_immediately_without_drain(self):
        for h in self._each():
            h._reset_container_state()
            self.assertTrue(h._begin_heartbeat_termination())

            def _closed():
                raise _ClientClosed()

            h.on_container_stop = _closed
            h.terminate()
            self.assertEqual(h.sleeps, [])
            self.assertEqual(h.exits, [1])

    def test_drain_during_forensics_skips_container_stop_and_exit(self):
        for h in self._each():
            h._reset_container_state()
            self.assertTrue(h._begin_heartbeat_termination())
            h.on_forensics = h._set_container_stopping
            h.terminate()
            self.assertEqual(h.container_stop_calls, 0)
            self.assertEqual(h.exits, [])
            self.assertEqual(h.state(), "stopping")

    def test_drain_during_container_stop_skips_exit(self):
        for template in self._each():
            for raise_client_closed in (False, True):
                with self.subTest(client_closed=raise_client_closed):
                    h = _Harness(template.path)
                    h._reset_container_state()
                    self.assertTrue(h._begin_heartbeat_termination())

                    def _stop_then_drain():
                        h._set_container_stopping()
                        if raise_client_closed:
                            raise _ClientClosed()

                    h.on_container_stop = _stop_then_drain
                    h.terminate()
                    self.assertEqual(h.container_stop_calls, 1)
                    self.assertEqual(h.exits, [])
                    self.assertEqual(h.state(), "stopping")

    def test_drain_during_post_stop_wait_skips_exit(self):
        for h in self._each():
            h._reset_container_state()
            self.assertTrue(h._begin_heartbeat_termination())
            calls = {"n": 0}

            def _drain_on_third_tick():
                calls["n"] += 1
                if calls["n"] == 3:
                    h._set_container_stopping()

            h.on_sleep = _drain_on_third_tick
            h.terminate()
            self.assertEqual(h.container_stop_calls, 1)
            self.assertEqual(len(h.sleeps), 3)
            self.assertEqual(h.exits, [])

    def test_exit_unless_draining_holds_lock_across_exit(self):
        for h in self._each():
            h._reset_container_state()
            self.assertTrue(h._begin_heartbeat_termination())
            observed_locked = []

            def _exit(code: int) -> None:
                observed_locked.append(h._CONTAINER_STATE_LOCK.locked())
                h.exits.append(code)
                raise _Exited()

            h.ns["os"] = types.SimpleNamespace(_exit=_exit)
            try:
                h._exit_unless_draining("test")
            except _Exited:
                pass
            self.assertEqual(observed_locked, [True])
            self.assertEqual(h.exits, [1])
            self.assertFalse(h._CONTAINER_STATE_LOCK.locked())

            h._set_container_stopping()
            h._exit_unless_draining("test")
            self.assertEqual(h.exits, [1])


if __name__ == "__main__":
    import unittest

    unittest.main()
