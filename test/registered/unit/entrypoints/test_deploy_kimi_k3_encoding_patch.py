"""Exercise the deploy recipes' encoding_k3.py rewrite without Modal or the HF hub.

The helpers AST-load the patch machinery and constants from the checked-in
serve files and run them against synthetic sources, including one built from
the exact vulnerable block the pinned model revision ships.
"""

from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEPLOY_FILES = [
    REPO_ROOT / "deploy/instinct/serve.py",
    REPO_ROOT / "deploy/instinct/serve_probe.py",
]
CONSTANT_NAMES = (
    "_KIMI_K3_VULNERABLE_APPEND_TEXT",
    "_KIMI_K3_SAFE_APPEND_TEXT",
)


def _load_rewrite(path: Path):
    """Exec the file's _rewrite_kimi_k3_encoding FunctionDef with its constants."""
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") in CONSTANT_NAMES
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_rewrite_kimi_k3_encoding"
        )
    ]
    assert len(nodes) == 3, f"expected 2 constants + rewrite fn in {path}"
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


SNAPSHOT_REQUIRED = (
    "config.json",
    "encoding_k3.py",
    "kimi_k3_processor.py",
    "media_utils.py",
    "tokenizer_config.json",
)


def _load_prepare(path: Path, runtime_dir: Path):
    """Exec prepare_model_snapshot with the rewrite machinery and a temp runtime dir."""
    ns = _load_rewrite(path)
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_model_snapshot"
    )
    ns.update(
        {
            "Path": Path,
            "shutil": shutil,
            "print": lambda *_a, **_k: None,
            "MODEL_NAME": "moonshotai/Kimi-K3",
            "MODEL_REVISION": "test-revision",
            "RUNTIME_MODEL_PATH": str(runtime_dir),
        }
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns


def _fake_snapshot(root: Path, vulnerable: str) -> Path:
    """Lay out a minimal HF snapshot: custom code, configs, one shard, a subdir."""
    root.mkdir(parents=True)
    for name in SNAPSHOT_REQUIRED:
        (root / name).write_text("{}\n" if name.endswith(".json") else "# stub\n")
    (root / "encoding_k3.py").write_text(
        "import re\n\n\n" + vulnerable + "\n\ndef _render():\n    pass\n"
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 16)
    (root / "assets").mkdir()
    (root / "assets" / "chat_template.jinja").write_text("{{ messages }}\n")
    return root


class TestDeployKimiK3EncodingPatch(CustomTestCase):
    def test_rewrite_swaps_vulnerable_block_for_safe_one(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                ns = _load_rewrite(path)
                source = (
                    "def _helper():\n    pass\n\n\n"
                    + ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"]
                    + "\ndef _after():\n    pass\n"
                ).encode()
                patched = ns["_rewrite_kimi_k3_encoding"](source).decode()
                self.assertIn(ns["_KIMI_K3_SAFE_APPEND_TEXT"], patched)
                self.assertNotIn(ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"], patched)
                self.assertTrue(patched.startswith("def _helper():"))
                self.assertTrue(patched.endswith("def _after():\n    pass\n"))
                # Round-trip: reinserting the vulnerable block restores the source.
                restored = patched.replace(
                    ns["_KIMI_K3_SAFE_APPEND_TEXT"],
                    ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"],
                )
                self.assertEqual(restored, source.decode())

    def test_rewrite_rejects_missing_or_duplicated_block(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                ns = _load_rewrite(path)
                with self.assertRaises(RuntimeError):
                    ns["_rewrite_kimi_k3_encoding"](b"def unrelated():\n    pass\n")
                doubled = (
                    ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"]
                    + ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"]
                ).encode()
                with self.assertRaises(RuntimeError):
                    ns["_rewrite_kimi_k3_encoding"](doubled)

    def test_safe_block_only_emits_plain_text(self):
        # The safe _append_text body must never consult image_state or the
        # placeholder; otherwise untrusted strings could still bind images.
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                ns = _load_rewrite(path)
                safe = ns["_KIMI_K3_SAFE_APPEND_TEXT"]
                self.assertNotIn("IMAGE_PLACEHOLDER", safe)
                self.assertNotIn("next_prompt", safe)
                self.assertIn("segments.extend(_text(text))", safe)


class TestDeployPrepareModelSnapshot(CustomTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, path: Path, snapshot: Path, runtime: Path):
        ns = _load_prepare(path, runtime)
        calls = []

        def snapshot_download(repo_id, revision):
            calls.append((repo_id, revision))
            return str(snapshot)

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = ns["prepare_model_snapshot"]()
        self.assertEqual(calls, [("moonshotai/Kimi-K3", "test-revision")])
        return ns, result

    def test_materializes_copies_symlinks_and_patches_encoder(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                ns = _load_rewrite(path)
                snapshot = _fake_snapshot(
                    self.tmp / path.stem / "snapshot",
                    ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"],
                )
                runtime = self.tmp / path.stem / "runtime"
                ns, result = self._run(path, snapshot, runtime)

                self.assertEqual(result, str(runtime))
                shard = runtime / "model-00001-of-00001.safetensors"
                self.assertTrue(shard.is_symlink())
                self.assertEqual(
                    shard.resolve(),
                    (snapshot / "model-00001-of-00001.safetensors").resolve(),
                )
                for name in SNAPSHOT_REQUIRED:
                    with self.subTest(copied=name):
                        self.assertTrue((runtime / name).is_file())
                        self.assertFalse((runtime / name).is_symlink())
                self.assertTrue((runtime / "assets" / "chat_template.jinja").is_file())

                patched = (runtime / "encoding_k3.py").read_text()
                self.assertIn(ns["_KIMI_K3_SAFE_APPEND_TEXT"], patched)
                self.assertNotIn(ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"], patched)
                self.assertTrue(patched.startswith("import re\n"))
                self.assertTrue(patched.endswith("def _render():\n    pass\n"))
                # The shared HF cache must never be mutated.
                self.assertIn(
                    ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"],
                    (snapshot / "encoding_k3.py").read_text(),
                )

    def test_rerun_replaces_stale_runtime_dir(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                ns = _load_rewrite(path)
                snapshot = _fake_snapshot(
                    self.tmp / path.stem / "snapshot",
                    ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"],
                )
                runtime = self.tmp / path.stem / "runtime"
                runtime.mkdir(parents=True)
                (runtime / "stale.bin").write_text("old")
                self._run(path, snapshot, runtime)
                self.assertFalse((runtime / "stale.bin").exists())
                self.assertTrue((runtime / "encoding_k3.py").is_file())

    def test_incomplete_snapshot_is_rejected_before_writing(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                ns = _load_rewrite(path)
                snapshot = _fake_snapshot(
                    self.tmp / path.stem / "snapshot",
                    ns["_KIMI_K3_VULNERABLE_APPEND_TEXT"],
                )
                (snapshot / "media_utils.py").unlink()
                runtime = self.tmp / path.stem / "runtime"
                with self.assertRaisesRegex(RuntimeError, "media_utils.py"):
                    self._run(path, snapshot, runtime)
                self.assertFalse(runtime.exists())

    def test_drifted_encoder_is_rejected(self):
        for path in DEPLOY_FILES:
            with self.subTest(file=path.name):
                snapshot = _fake_snapshot(
                    self.tmp / path.stem / "snapshot",
                    "def _append_text(segments, text, image_state):\n    pass\n",
                )
                runtime = self.tmp / path.stem / "runtime"
                with self.assertRaisesRegex(RuntimeError, "image-placeholder block"):
                    self._run(path, snapshot, runtime)


if __name__ == "__main__":
    import unittest

    unittest.main()
