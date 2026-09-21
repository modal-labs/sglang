import ast
import os
import shutil
import tempfile
import time
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


def _load_seed_fn(path: Path, image_dir: Path, module: str):
    """Exec the file's seed_prebuilt_jit FunctionDef in a minimal namespace."""
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "seed_prebuilt_jit"
    )
    ns = {
        "os": os,
        "shutil": shutil,
        "tempfile": tempfile,
        "Path": Path,
        "PREBUILT_JIT_IMAGE_DIR": str(image_dir),
        "PREBUILT_JIT_MODULE": module,
        "PREBUILT_JIT_STALE_SECONDS": 3600,
        "time": time,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["seed_prebuilt_jit"]


class TestDeploySeedPrebuiltJit(CustomTestCase):
    def _fresh_layout(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.module = "mod_x"
        self.image_dir = self.tmp / "image"
        (self.image_dir / self.module).mkdir(parents=True)
        (self.image_dir / self.module / "a.so").write_text("a")
        self.jit_cache = self.tmp / "cache"
        self.jit_cache.mkdir()
        self.dst = self.jit_cache / "tvm-ffi" / self.module

    def _staging_dirs(self, ignore=()):
        return [
            p
            for p in (self.jit_cache / "tvm-ffi").glob(f".{self.module}.*")
            if p.name not in ignore
        ]

    def _for_each_file(self):
        for path in DEPLOY_FILES:
            with self.subTest(path=path.name):
                self._fresh_layout()
                yield path

    def test_seed_prebuilt_jit(self):
        for path in self._for_each_file():
            seed = _load_seed_fn(path, self.image_dir, self.module)

            # (a) fresh seed publishes the whole module
            seed(str(self.jit_cache))
            self.assertTrue((self.dst / "a.so").exists())
            self.assertEqual(self._staging_dirs(), [])

            # reset layout for the remaining cases
            shutil.rmtree(self.dst)
            self.assertEqual(list(self.dst.parent.iterdir()), [])

            # (b) skip-if-present: existing .so is left alone
            self.dst.mkdir(parents=True)
            (self.dst / "b.so").write_text("b")
            seed(str(self.jit_cache))
            self.assertFalse((self.dst / "a.so").exists())
            self.assertEqual(self._staging_dirs(), [])

            # (c) a failed copy must not publish a partial module
            shutil.rmtree(self.dst)
            real_copytree = shutil.copytree

            def copy_then_fail(src, dst, **kwargs):
                real_copytree(src, dst, **kwargs)
                raise OSError("simulated copy failure")

            with patch.object(shutil, "copytree", copy_then_fail):
                with self.assertRaises(OSError):
                    seed(str(self.jit_cache))
            self.assertFalse(self.dst.exists())
            self.assertEqual(self._staging_dirs(), [])

            # (d) a concurrent winner keeps its own contents
            real_rename = os.rename

            def loser_rename(src, dst):
                Path(dst).mkdir(exist_ok=True)
                (Path(dst) / "b.so").write_text("b")
                real_rename(src, dst)

            with patch.object(os, "rename", loser_rename):
                seed(str(self.jit_cache))
            self.assertEqual([p.name for p in self.dst.iterdir()], ["b.so"])
            self.assertEqual(self._staging_dirs(), [])

            # (e) a partial module left by an interrupted seed is replaced
            shutil.rmtree(self.dst)
            self.dst.mkdir(parents=True)
            (self.dst / "meta.json").write_text("{}")
            seed(str(self.jit_cache))
            self.assertEqual([p.name for p in self.dst.iterdir()], ["a.so"])
            self.assertEqual(self._staging_dirs(), [])

            # (f) a staging dir whose inode change time is older than
            # STALE_SECONDS is reclaimed even on the fast path
            stale = self.jit_cache / "tvm-ffi" / f".{self.module}.old"
            stale.mkdir()
            (self.dst / "b.so").write_text("b")
            (self.dst / "a.so").unlink()
            future = time.time() + 2 * 3600 + 1
            with patch.object(time, "time", return_value=future):
                seed(str(self.jit_cache))
            self.assertFalse(stale.exists())
            self.assertEqual(self._staging_dirs(), [])

            # (f2) a leftover with backdated mtime but fresh ctime is NOT
            # reclaimed (a copytree'd staging keeps the image's old mtime)
            backdated = self.jit_cache / "tvm-ffi" / f".{self.module}.old2"
            backdated.mkdir()
            old = time.time() - 2 * 3600
            os.utime(backdated, (old, old))
            seed(str(self.jit_cache))
            self.assertTrue(backdated.exists())
            shutil.rmtree(backdated)

            # (g) a fresh staging dir (plausibly a live seeder) is kept
            fresh = self.jit_cache / "tvm-ffi" / f".{self.module}.old"
            fresh.mkdir()
            seed(str(self.jit_cache))
            self.assertTrue(fresh.exists())
            self.assertEqual(self._staging_dirs(ignore={f".{self.module}.old"}), [])
            shutil.rmtree(fresh)

            # (h) quarantine is always dropped, even if the publish rename
            # loses to a concurrent seeder
            shutil.rmtree(self.dst)
            real_rename2 = os.rename
            calls = {"n": 0}

            def third_rename_loses(src, dst):
                calls["n"] += 1
                if calls["n"] == 3:
                    # the publish rename loses: winner's module appears
                    Path(dst).mkdir(exist_ok=True)
                    (Path(dst) / "b.so").write_text("b")
                    raise OSError("ENOTEMPTY")
                real_rename2(src, dst)

            self.dst.mkdir(parents=True)
            (self.dst / "meta.json").write_text("{}")
            # dst has only meta.json (no .so) -> partial-module path:
            # call1 staging->dst ENOTEMPTY, call2 dst->quarantine ok,
            # call3 staging->dst loses to the winner
            with patch.object(os, "rename", third_rename_loses):
                seed(str(self.jit_cache))
            self.assertEqual([p.name for p in self.dst.iterdir()], ["b.so"])
            self.assertEqual(self._staging_dirs(ignore={f".{self.module}.old"}), [])


if __name__ == "__main__":
    import unittest

    unittest.main()
