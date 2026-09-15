import functools
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.kernels.jit import cute_aot_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-test-cpu")


@pytest.fixture(autouse=True)
def reset_cache_state(monkeypatch):
    monkeypatch.delenv("SGLANG_CUTE_AOT_CACHE_DIR", raising=False)
    cute_aot_cache._flashinfer_installed = False
    cute_aot_cache._loaded_modules.clear()
    cute_aot_cache._runtime_library_handles.clear()
    cute_aot_cache._environment_fingerprint.cache_clear()
    cute_aot_cache._preload_runtime_libraries.cache_clear()
    cute_aot_cache._source_fingerprint.cache_clear()


def test_disabled_cache_compiles_without_cache_setup(monkeypatch):
    monkeypatch.setattr(
        cute_aot_cache,
        "_cache_key",
        lambda *args, **kwargs: pytest.fail("disabled cache computed a key"),
    )
    fresh = object()

    result = cute_aot_cache.compile_with_cute_aot_cache(
        kind="test",
        config=(),
        source_paths=(),
        enable_tvm_ffi=True,
        compile_fn=lambda: fresh,
    )

    assert result is fresh


def test_source_fingerprint_covers_all_python_sources(tmp_path):
    root = tmp_path / "cute_dsl"
    nested = root / "attention"
    nested.mkdir(parents=True)
    first = root / "__init__.py"
    second = nested / "kernel.py"
    ignored = nested / "kernel.txt"
    first.write_text("VERSION = 1\n")
    second.write_text("TILE = 128\n")
    ignored.write_text("not Python")

    original = cute_aot_cache._source_fingerprint((str(root),))
    ignored.write_text("changed")
    cute_aot_cache._source_fingerprint.cache_clear()
    assert cute_aot_cache._source_fingerprint((str(root),)) == original

    second.write_text("TILE = 256\n")
    cute_aot_cache._source_fingerprint.cache_clear()
    assert cute_aot_cache._source_fingerprint((str(root),)) != original


def test_cuda_device_ordinal_is_not_part_of_shared_key():
    TorchDevice = type("device", (), {"__module__": "torch"})
    first = TorchDevice()
    first.type = "cuda"
    first.index = 0
    second = TorchDevice()
    second.type = "cuda"
    second.index = 7

    assert cute_aot_cache._normalize(first) == cute_aot_cache._normalize(second)


def test_compile_environment_tracks_codegen_controls_and_deployment_salt(monkeypatch):
    expected = {
        "CUTE_DSL_COMPILER_OPT": "iket",
        "CUTE_DSL_ENABLE_ASSERTIONS": "1",
        "CUTE_DSL_ENABLE_TVM_FFI": "0",
        "SGLANG_CUTE_AOT_ABI_SALT": "image-sha256:test",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    compile_environment = cute_aot_cache._compile_environment()

    for name, value in expected.items():
        assert compile_environment[name] == value


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
def test_runtime_libraries_are_preloaded_globally(
    monkeypatch, tmp_path, enable_tvm_ffi
):
    import cutlass.cute as cute

    first = tmp_path / "libcuda_runtime.so"
    second = tmp_path / "libtvm_ffi_runtime.so"
    first.touch()
    second.touch()
    paths = [first, second] if enable_tvm_ffi else [first]
    monkeypatch.setattr(
        cute.runtime,
        "find_runtime_libraries",
        lambda *, enable_tvm_ffi: [str(path) for path in paths],
    )
    calls = []
    monkeypatch.setattr(
        cute_aot_cache.ctypes,
        "CDLL",
        lambda path, *, mode: calls.append((path, mode)) or object(),
    )

    assert cute_aot_cache._preload_runtime_libraries(enable_tvm_ffi) == tuple(
        str(path) for path in paths
    )
    assert calls == [(str(path), cute_aot_cache.ctypes.RTLD_GLOBAL) for path in paths]
    assert len(cute_aot_cache._runtime_library_handles) == len(paths)

    # The same ABI mode is prepared once per process.
    cute_aot_cache._preload_runtime_libraries(enable_tvm_ffi)
    assert len(calls) == len(paths)


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
def test_object_loader_uses_requested_cute_abi(monkeypatch, tmp_path, enable_tvm_ffi):
    import cutlass.cute as cute

    function = object()

    class Module:
        def __getitem__(self, prefix):
            assert prefix == "prefix"
            return function

    calls = []

    def load_module(path, *, enable_tvm_ffi):
        calls.append((path, enable_tvm_ffi))
        return Module()

    preload_modes = []
    monkeypatch.setattr(
        cute_aot_cache,
        "_preload_runtime_libraries",
        lambda mode: preload_modes.append(mode),
    )
    monkeypatch.setattr(cute.runtime, "load_module", load_module)
    object_path = tmp_path / "kernel.o"
    object_path.write_bytes(b"object")

    assert (
        cute_aot_cache._load_object(
            object_path, "prefix", enable_tvm_ffi=enable_tvm_ffi
        )
        is function
    )
    assert preload_modes == [enable_tvm_ffi]
    assert calls == [(str(object_path), enable_tvm_ffi)]
    assert len(cute_aot_cache._loaded_modules) == 1


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
def test_miss_is_exported_atomically_and_next_process_hits(
    monkeypatch, tmp_path, enable_tvm_ffi
):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    key = "a" * 64
    prefix = f"sglcute_{key[:20]}"
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: key)

    exports = []

    class FreshFunction:
        def export_to_c(self, path, name=None, **kwargs):
            exports.append((path, name, kwargs))
            if enable_tvm_ffi:
                assert name is None
                assert kwargs == {"function_name": prefix}
                Path(path).write_bytes(b"tvm-object")
            else:
                assert name == "kernel"
                assert kwargs == {"function_prefix": prefix}
                Path(path, "kernel.h").write_text("header")
                Path(path, "kernel.o").write_bytes(b"native-object")

    fresh = FreshFunction()
    loaded = object()
    load_modes = []

    def load_object(path, function_prefix, *, enable_tvm_ffi):
        load_modes.append(enable_tvm_ffi)
        assert function_prefix == prefix
        assert path.read_bytes() in (b"tvm-object", b"native-object")
        return loaded

    monkeypatch.setattr(cute_aot_cache, "_load_object", load_object)

    first = cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=("shape", 8),
        source_paths=(__file__,),
        enable_tvm_ffi=enable_tvm_ffi,
        compile_fn=lambda: fresh,
    )
    second = cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=("shape", 8),
        source_paths=(__file__,),
        enable_tvm_ffi=enable_tvm_ffi,
        compile_fn=lambda: pytest.fail("disk hit recompiled"),
    )

    assert first is fresh
    assert second is loaded
    assert load_modes == [enable_tvm_ffi]
    assert len(exports) == 1
    assert len(list((tmp_path / "kernel").glob("*.o"))) == 1
    assert not list((tmp_path / "kernel").glob("*.tmp.o"))


def test_lock_rechecks_for_object_published_by_another_process(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    key = "b" * 64
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: key)
    loaded = object()

    @contextmanager
    def publish_while_waiting(lock_path):
        object_path = tmp_path / "kernel" / f"{key}.o"
        object_path.write_bytes(b"published")
        yield

    monkeypatch.setattr(cute_aot_cache, "_exclusive_lock", publish_while_waiting)
    monkeypatch.setattr(
        cute_aot_cache,
        "_load_object",
        lambda path, prefix, *, enable_tvm_ffi: loaded,
    )

    result = cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=(),
        source_paths=(__file__,),
        enable_tvm_ffi=True,
        compile_fn=lambda: pytest.fail("racing process caused a recompile"),
    )

    assert result is loaded


def test_cache_infrastructure_failure_is_fail_open(monkeypatch, tmp_path):
    unusable = tmp_path / "not-a-directory"
    unusable.write_text("file")
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(unusable))
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: "c" * 64)
    fresh = object()

    result = cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=(),
        source_paths=(__file__,),
        enable_tvm_ffi=False,
        compile_fn=lambda: fresh,
    )

    assert result is fresh


def test_lock_failure_is_fail_open(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: "f" * 64)

    class FailingLock:
        def __enter__(self):
            raise cute_aot_cache._CacheLockError("flock unsupported")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(cute_aot_cache, "_exclusive_lock", lambda path: FailingLock())

    class FreshFunction:
        def export_to_c(self, path, name, *, function_prefix):
            Path(path, f"{name}.o").write_bytes(b"native-object")
            Path(path, f"{name}.h").write_text("header")

    fresh = FreshFunction()

    result = cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=(),
        source_paths=(__file__,),
        enable_tvm_ffi=False,
        compile_fn=lambda: fresh,
    )

    assert result is fresh
    assert len(list((tmp_path / "kernel").glob("*.o"))) == 1


def test_lock_is_container_local_not_on_persistent_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: "1" * 64)
    seen_lock_paths = []

    @contextmanager
    def capture_lock(path):
        seen_lock_paths.append(path)
        yield

    monkeypatch.setattr(cute_aot_cache, "_exclusive_lock", capture_lock)

    class FreshFunction:
        def export_to_c(self, path, *, function_name):
            Path(path).write_bytes(b"tvm-object")

    cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=(),
        source_paths=(__file__,),
        enable_tvm_ffi=True,
        compile_fn=FreshFunction,
    )

    assert len(seen_lock_paths) == 1
    assert tmp_path not in seen_lock_paths[0].parents


def test_corrupt_object_is_replaced_atomically(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    key = "2" * 64
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: key)
    object_path = tmp_path / "kernel" / f"{key}.o"
    object_path.parent.mkdir()
    object_path.write_bytes(b"")
    loaded = object()

    def load_object(path, prefix, *, enable_tvm_ffi):
        if path.read_bytes() != b"complete-object":
            raise ValueError("corrupt object")
        return loaded

    monkeypatch.setattr(cute_aot_cache, "_load_object", load_object)

    class FreshFunction:
        def export_to_c(self, path, *, function_name):
            Path(path).write_bytes(b"complete-object")

    fresh = FreshFunction()
    assert (
        cute_aot_cache.compile_with_cute_aot_cache(
            kind="kernel",
            config=(),
            source_paths=(__file__,),
            enable_tvm_ffi=True,
            compile_fn=lambda: fresh,
        )
        is fresh
    )
    assert object_path.read_bytes() == b"complete-object"
    assert (
        cute_aot_cache.compile_with_cute_aot_cache(
            kind="kernel",
            config=(),
            source_paths=(__file__,),
            enable_tvm_ffi=True,
            compile_fn=lambda: pytest.fail("recompiled repaired object"),
        )
        is loaded
    )


def test_export_failure_keeps_fresh_kernel_usable(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: "d" * 64)

    class FreshFunction:
        def export_to_c(self, *args, **kwargs):
            raise OSError("volume unavailable")

    fresh = FreshFunction()
    result = cute_aot_cache.compile_with_cute_aot_cache(
        kind="kernel",
        config=(),
        source_paths=(__file__,),
        enable_tvm_ffi=True,
        compile_fn=lambda: fresh,
    )

    assert result is fresh


def test_kernel_compile_failure_is_not_retried_or_hidden(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cute_aot_cache, "_cache_key", lambda *args, **kwargs: "e" * 64)
    calls = 0

    def fail_compile():
        nonlocal calls
        calls += 1
        raise RuntimeError("real CuTe compiler failure")

    with pytest.raises(RuntimeError, match="real CuTe compiler failure"):
        cute_aot_cache.compile_with_cute_aot_cache(
            kind="kernel",
            config=(),
            source_paths=(__file__,),
            enable_tvm_ffi=True,
            compile_fn=fail_compile,
        )

    assert calls == 1


def test_flashinfer_installer_canonicalizes_full_compile_config(monkeypatch, tmp_path):
    source = tmp_path / "flashinfer" / "cute_dsl" / "attention" / "monolithic"
    source.mkdir(parents=True)
    module_path = source / "mla_decode.py"
    module_path.write_text("# mocked FlashInfer source\n")
    calls = []

    @functools.cache
    def original(dtype, page_size=64, enable_pdl=False):
        calls.append((dtype, page_size, enable_pdl))
        return "fresh"

    class KernelFP8:
        pass

    class KernelFP16:
        pass

    fake_module = SimpleNamespace(
        __file__=str(module_path),
        _get_compiled_mla_kernel=original,
        BlackwellMultiHeadLatentAttentionForwardFP8=KernelFP8,
        BlackwellMultiHeadLatentAttentionForwardFP16=KernelFP16,
    )
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        cute_aot_cache, "_flashinfer_mla_decode_module", lambda: fake_module
    )

    captured = {}

    def cached_compile(**kwargs):
        captured.update(kwargs)
        return kwargs["compile_fn"]()

    monkeypatch.setattr(cute_aot_cache, "compile_with_cute_aot_cache", cached_compile)

    assert cute_aot_cache.install_flashinfer_mla_decode_aot_cache()
    assert cute_aot_cache.install_flashinfer_mla_decode_aot_cache()
    assert fake_module._get_compiled_mla_kernel("bf16") == "fresh"

    assert calls == [("bf16", 64, False)]
    assert captured["kind"] == "flashinfer_mla_decode"
    assert captured["config"] == (
        ("dtype", "bf16"),
        ("page_size", 64),
        ("enable_pdl", False),
    )
    assert captured["enable_tvm_ffi"] is True
    assert str(source.parents[1]) in captured["source_paths"]


def test_flashinfer_installer_failure_does_not_replace_original(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        cute_aot_cache,
        "_flashinfer_mla_decode_module",
        lambda: (_ for _ in ()).throw(ImportError("optional dependency mismatch")),
    )

    assert not cute_aot_cache.install_flashinfer_mla_decode_aot_cache()
    assert not cute_aot_cache._flashinfer_installed
