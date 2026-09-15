"""Persistent object cache for explicit ``cute.compile`` calls.

CuTeDSL 4.6 deliberately bypasses its ordinary disk cache for explicit
``cute.compile`` calls.  This module persists the two such kernels used by the
Kimi-K3 serving path:

* FlashInfer's monolithic CuTeDSL MLA decode kernel (TVM-FFI ABI).
* SGLang's KDA MTP verify kernel (native CuTe ABI).

Set ``SGLANG_CUTE_AOT_CACHE_DIR`` to a persistent directory to enable it.
Cache failures are non-fatal: compilation always remains the fallback.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import functools
import hashlib
import inspect
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

logger = logging.getLogger(__name__)

_CACHE_ENV = "SGLANG_CUTE_AOT_CACHE_DIR"
_CACHE_FORMAT = 2
_LOCK_TIMEOUT_SECONDS = 20 * 60

_flashinfer_installed = False
_install_lock = threading.Lock()

# A loaded function owns resources through its ExternalBinaryModule.  Keep the
# module alive for as long as the returned callable can be used.
_loaded_modules: list[Any] = []
# CuTe's cached objects resolve symbols from these libraries at dlopen time.
# Retain the handles so they remain globally visible for the process lifetime.
_runtime_library_handles: list[Any] = []


class _CacheLockError(RuntimeError):
    pass


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _compile_environment() -> dict[str, str | None]:
    """Return every environment control that can affect emitted host/device code."""
    names = (
        "CC",
        "CFLAGS",
        "CUDAARCHS",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDACXX",
        "CUTE_DSL_ARCH",
        "CUTE_DSL_COMPILER_OPT",
        "CUTE_DSL_DEBUG",
        "CUTE_DSL_ENABLE_ASSERTIONS",
        "CUTE_DSL_ENABLE_OPTIMIZATION_WARNINGS",
        "CUTE_DSL_ENABLE_TVM_FFI",
        "CUTE_DSL_LIBS",
        "CUTE_DSL_LINEINFO",
        "CUTE_DSL_PTXAS_PATH",
        "CUTE_DSL_WARNINGS_AS_ERRORS",
        "CUTE_DSL_WARNINGS_IGNORE",
        "CUTLASS_PTXAS_PATH",
        "CXX",
        "CXXFLAGS",
        "LDFLAGS",
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
        "PTXAS_OPTIONS",
        "SGLANG_CUTE_AOT_ABI_SALT",
        "TRITON_PTXAS_PATH",
    )
    return {name: os.environ.get(name) for name in names}


def _host_cpu_fingerprint() -> str:
    """Fingerprint native host-code features used by CuTe's default AOT target."""
    fields: dict[str, str] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if not line.strip():
                break
            key, separator, value = line.partition(":")
            if separator and key.strip() in {
                "cpu family",
                "flags",
                "model",
                "model name",
                "stepping",
                "vendor_id",
            }:
                fields[key.strip()] = value.strip()
    except OSError:
        pass
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _executable_identity(requested: str | None) -> dict[str, Any]:
    """Identify one effective external compiler tool without failing cache setup."""
    if not requested:
        return {"requested": requested, "resolved": None}
    resolved = shutil.which(requested)
    if resolved is None and Path(requested).is_file():
        resolved = str(Path(requested).resolve())
    if resolved is None:
        return {"requested": requested, "resolved": None}

    path = Path(resolved).resolve()
    try:
        stat = path.stat()
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        version_text = (completed.stdout or completed.stderr).strip()[:4096]
        return {
            "requested": requested,
            "resolved": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "version": version_text,
        }
    except (OSError, subprocess.SubprocessError):
        return {"requested": requested, "resolved": str(path), "version": "unknown"}


@functools.lru_cache(maxsize=1)
def _environment_fingerprint() -> str:
    import cutlass
    import torch
    import tvm_ffi

    try:
        from cutlass.cute.export.export import object_file_version
    except ImportError:
        object_file_version = "unknown"

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    compile_env = _compile_environment()
    ptxas = (
        compile_env["CUTE_DSL_PTXAS_PATH"]
        or compile_env["CUTLASS_PTXAS_PATH"]
        or compile_env["TRITON_PTXAS_PATH"]
        or "ptxas"
    )
    nvcc = compile_env["CUDACXX"] or "nvcc"
    cxx = compile_env["CXX"] or "c++"
    return json.dumps(
        {
            "cache_format": _CACHE_FORMAT,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "host": platform.machine(),
            "host_cpu": _host_cpu_fingerprint(),
            "libc": platform.libc_ver(),
            "flashinfer": _package_version("flashinfer-python"),
            "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
            "cutlass": getattr(cutlass, "__version__", "unknown"),
            "cutlass_cuda": str(getattr(cutlass, "CUDA_VERSION", "unknown")),
            "object_file": str(object_file_version),
            "tvm_ffi": getattr(tvm_ffi, "__version__", "unknown"),
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda": getattr(torch.version, "cuda", None),
            "gpu": {
                "name": props.name,
                "cc": [props.major, props.minor],
                "sm_count": props.multi_processor_count,
            },
            "compile_env": compile_env,
            "toolchain": {
                "cxx": _executable_identity(cxx),
                "nvcc": _executable_identity(nvcc),
                "ptxas": _executable_identity(ptxas),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@functools.lru_cache(maxsize=None)
def _source_fingerprint(source_paths: tuple[str, ...]) -> str:
    """Hash every Python source under the supplied files/directories."""
    digest = hashlib.sha256()
    for index, raw_path in enumerate(source_paths):
        path = Path(raw_path).resolve()
        if path.is_dir():
            files = sorted(
                candidate for candidate in path.rglob("*.py") if candidate.is_file()
            )
            base = path
        elif path.is_file():
            files = [path]
            base = path.parent
        else:
            raise FileNotFoundError(path)

        digest.update(f"root:{index}:{path.name}".encode())
        for source in files:
            relative = source.relative_to(base).as_posix()
            content = source.read_bytes()
            digest.update(relative.encode())
            digest.update(len(content).to_bytes(8, "little"))
            digest.update(content)
    return digest.hexdigest()


def _normalize(value: Any) -> Any:
    """Convert a kernel configuration to deterministic JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    value_type = type(value)
    if value_type.__module__.startswith("torch"):
        # A CUDA device ordinal differs across TP ranks but does not affect the
        # generated kernel.  GPU model/CC/SM count are keyed separately.
        if value_type.__name__ == "device":
            return {"type": "torch.device", "device_type": value.type}
        return {
            "type": f"{value_type.__module__}.{value_type.__qualname__}",
            "value": str(value),
        }
    return {
        "type": f"{value_type.__module__}.{value_type.__qualname__}",
        "value": repr(value),
    }


def _cache_key(
    kind: str,
    config: Any,
    source_paths: tuple[str, ...],
    *,
    enable_tvm_ffi: bool,
) -> str:
    payload = {
        "kind": kind,
        "abi": "tvm_ffi" if enable_tvm_ffi else "cute_native",
        "environment": _environment_fingerprint(),
        "source": _source_fingerprint(source_paths),
        "config": _normalize(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Acquire a bounded advisory lock for one cache key."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as error:
        raise _CacheLockError(f"could not open CuTe AOT lock {path}") from error
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise _CacheLockError(
                        f"could not acquire CuTe AOT lock {path}"
                    ) from error
                if time.monotonic() >= deadline:
                    raise _CacheLockError(f"timed out waiting for CuTe AOT lock {path}")
                time.sleep(0.1)
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            logger.warning("[cute-aot] failed to release lock %s", path, exc_info=True)
        finally:
            os.close(fd)


def _load_object(path: Path, prefix: str, *, enable_tvm_ffi: bool) -> Any:
    import cutlass.cute as cute

    _preload_runtime_libraries(enable_tvm_ffi)
    module = cute.runtime.load_module(str(path), enable_tvm_ffi=enable_tvm_ffi)
    function = module[prefix]
    _loaded_modules.append(module)
    return function


@functools.lru_cache(maxsize=2)
def _preload_runtime_libraries(enable_tvm_ffi: bool) -> tuple[str, ...]:
    """Make CuTe runtime symbols globally visible before loading cached objects."""
    import cutlass.cute as cute

    loaded: list[str] = []
    for raw_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=enable_tvm_ffi):
        path = Path(raw_path)
        if not path.is_file():
            continue
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        _runtime_library_handles.append(handle)
        loaded.append(str(path))
    return tuple(loaded)


def _atomic_export(
    function: Any,
    object_path: Path,
    prefix: str,
    *,
    enable_tvm_ffi: bool,
) -> int:
    """Export to a sibling temporary object, then atomically publish it."""
    temporary_object = object_path.with_name(
        f".{object_path.stem}.{uuid.uuid4().hex}.tmp.o"
    )
    temporary_dir: str | None = None
    try:
        if enable_tvm_ffi:
            function.export_to_c(str(temporary_object), function_name=prefix)
        else:
            temporary_dir = tempfile.mkdtemp(
                prefix=f".{object_path.stem}.", dir=object_path.parent
            )
            function.export_to_c(
                temporary_dir,
                "kernel",
                function_prefix=prefix,
            )
            os.replace(Path(temporary_dir) / "kernel.o", temporary_object)

        size = temporary_object.stat().st_size
        if size == 0:
            raise OSError(
                f"CuTe AOT export produced an empty object: {temporary_object}"
            )
        os.replace(temporary_object, object_path)
        return size
    finally:
        temporary_object.unlink(missing_ok=True)
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)


def _compile_and_export(
    *,
    compile_fn: Callable[[], Any],
    object_path: Path,
    prefix: str,
    enable_tvm_ffi: bool,
    kind: str,
    key: str,
) -> Any:
    started = time.perf_counter()
    function = compile_fn()
    compile_seconds = time.perf_counter() - started
    try:
        size = _atomic_export(
            function,
            object_path,
            prefix,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        logger.info(
            "[cute-aot] MISS %s compile=%.1fs size=%.2fMB key=%s",
            kind,
            compile_seconds,
            size / 1e6,
            key[:16],
        )
    except Exception:
        logger.exception(
            "[cute-aot] export failed for %s key=%s; using fresh kernel",
            kind,
            key[:16],
        )
    return function


def compile_with_cute_aot_cache(
    *,
    kind: str,
    config: Any,
    source_paths: Sequence[str | os.PathLike[str]],
    enable_tvm_ffi: bool,
    compile_fn: Callable[[], Any],
) -> Any:
    """Load one explicit CuTe compile from disk, or compile and persist it."""
    cache_root = os.environ.get(_CACHE_ENV)
    if not cache_root:
        return compile_fn()

    paths = (
        os.path.abspath(__file__),
        *(os.path.abspath(os.fspath(path)) for path in source_paths),
    )
    try:
        key = _cache_key(kind, config, paths, enable_tvm_ffi=enable_tvm_ffi)
        prefix = f"sglcute_{key[:20]}"
        cache_dir = Path(cache_root) / kind
        cache_dir.mkdir(parents=True, exist_ok=True)
        object_path = cache_dir / f"{key}.o"
    except Exception:
        logger.exception("[cute-aot] key/setup failed for %s; compiling normally", kind)
        return compile_fn()

    # Modal's persistent volume need only provide atomic same-directory
    # renames. Keep the advisory lock on the container-local filesystem, where
    # it serializes TP workers without depending on DirectFS/9p flock support.
    # Separate containers may compile the same key concurrently; their complete
    # sibling objects race safely through os.replace.
    lock_dir = Path(tempfile.gettempdir()) / f"sglang-cute-aot-locks-{os.getuid()}"
    lock_path = lock_dir / f"{key}.lock"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            "[cute-aot] local lock setup failed for %s key=%s; "
            "publishing without a lock",
            kind,
            key[:16],
            exc_info=True,
        )
        return _compile_and_export(
            compile_fn=compile_fn,
            object_path=object_path,
            prefix=prefix,
            enable_tvm_ffi=enable_tvm_ffi,
            kind=kind,
            key=key,
        )

    if object_path.is_file():
        try:
            function = _load_object(object_path, prefix, enable_tvm_ffi=enable_tvm_ffi)
            logger.info("[cute-aot] HIT %s key=%s", kind, key[:16])
            return function
        except Exception:
            logger.warning(
                "[cute-aot] load failed for %s key=%s; rebuilding",
                kind,
                key[:16],
                exc_info=True,
            )

    try:
        with _exclusive_lock(lock_path):
            # Another process may have populated the object while this one waited.
            if object_path.is_file():
                try:
                    function = _load_object(
                        object_path, prefix, enable_tvm_ffi=enable_tvm_ffi
                    )
                    logger.info("[cute-aot] HIT-after-wait %s key=%s", kind, key[:16])
                    return function
                except Exception:
                    logger.warning(
                        "[cute-aot] locked load failed for %s key=%s; rebuilding",
                        kind,
                        key[:16],
                        exc_info=True,
                    )

            return _compile_and_export(
                compile_fn=compile_fn,
                object_path=object_path,
                prefix=prefix,
                enable_tvm_ffi=enable_tvm_ffi,
                kind=kind,
                key=key,
            )
    except _CacheLockError:
        logger.exception(
            "[cute-aot] local lock failed for %s key=%s; publishing without a lock",
            kind,
            key[:16],
        )
        return _compile_and_export(
            compile_fn=compile_fn,
            object_path=object_path,
            prefix=prefix,
            enable_tvm_ffi=enable_tvm_ffi,
            kind=kind,
            key=key,
        )


def _flashinfer_mla_decode_module() -> Any:
    from flashinfer.cute_dsl.attention.monolithic import mla_decode

    return mla_decode


def install_flashinfer_mla_decode_aot_cache() -> bool:
    """Patch FlashInfer's monolithic MLA decode compile before its first use."""
    global _flashinfer_installed
    if not os.environ.get(_CACHE_ENV):
        return False

    with _install_lock:
        if _flashinfer_installed:
            return True
        try:
            module = _flashinfer_mla_decode_module()
            current = module._get_compiled_mla_kernel
            if getattr(current, "_sglang_cute_aot_cache", False):
                _flashinfer_installed = True
                return True

            original = getattr(current, "__wrapped__", current)
            signature = inspect.signature(original)
            module_path = Path(module.__file__).resolve()
            cute_source_root = module_path.parents[2]
            source_paths = [str(cute_source_root)]
            for source_object in (
                original,
                module.BlackwellMultiHeadLatentAttentionForwardFP8,
                module.BlackwellMultiHeadLatentAttentionForwardFP16,
            ):
                try:
                    source_file = inspect.getsourcefile(source_object)
                except TypeError:
                    source_file = None
                if source_file is not None:
                    resolved = Path(source_file).resolve()
                    if not resolved.is_relative_to(cute_source_root):
                        source_paths.append(str(resolved))

            @functools.cache
            @functools.wraps(original)
            def cached_compile(*args: Any, **kwargs: Any) -> Any:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return compile_with_cute_aot_cache(
                    kind="flashinfer_mla_decode",
                    config=tuple(bound.arguments.items()),
                    source_paths=source_paths,
                    enable_tvm_ffi=True,
                    compile_fn=lambda: original(*args, **kwargs),
                )

            cached_compile._sglang_cute_aot_cache = True  # type: ignore[attr-defined]
            module._get_compiled_mla_kernel = cached_compile
            _flashinfer_installed = True
            logger.info(
                "[cute-aot] enabled FlashInfer MLA decode cache at %s",
                os.environ[_CACHE_ENV],
            )
            return True
        except Exception:
            logger.exception(
                "[cute-aot] FlashInfer MLA cache installation failed; "
                "continuing without persistence"
            )
            return False
