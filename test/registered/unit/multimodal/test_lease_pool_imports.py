"""CPU tests for imports required by the CUDA-IPC lease pool."""

import ast
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.utils import cuda_vmm_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _sglang_imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sglang."):
                    yield alias.name, ()
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("sglang."):
                yield node.module, tuple(alias.name for alias in node.names)


def test_lease_pool_imports_resolve():
    root = Path(__file__).parents[4]
    paths = sorted((root / "python/sglang/srt/multimodal/transport").glob("*.py"))
    paths.append(root / "python/sglang/srt/utils/cuda_ipc_transport_utils.py")

    for path in paths:
        for module_name, names in _sglang_imports(path):
            assert importlib.util.find_spec(module_name) is not None, (
                path,
                module_name,
            )
            module = importlib.import_module(module_name)
            for name in names:
                if hasattr(module, name):
                    continue
                assert importlib.util.find_spec(f"{module_name}.{name}") is not None, (
                    path,
                    module_name,
                    name,
                )


def test_check_drv():
    success = object()
    driver = SimpleNamespace(CUresult=SimpleNamespace(CUDA_SUCCESS=success))

    with patch.object(cuda_vmm_utils, "_get_cuda_driver", return_value=driver):
        assert cuda_vmm_utils.check_drv((success, 42), "x") == 42
        assert cuda_vmm_utils.check_drv((success,), "x") is None
        with pytest.raises(RuntimeError, match="x"):
            cuda_vmm_utils.check_drv((object(),), "x")
