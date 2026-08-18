# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA core dump and py-spy dump utilities."""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import signal
import subprocess
import time
from errno import ENXIO
from pathlib import Path
from typing import List

import psutil

logger = logging.getLogger(__name__)

# Signal used to request an in-process faulthandler stack dump from engine
# subprocesses. Chosen because nothing else in sglang claims it.
FAULTHANDLER_DUMP_SIGNAL = signal.SIGUSR2


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def _is_sglang_engine_subprocess(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith(("sglang::scheduler", "sglang::detokenizer"))


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


def collect_engine_subprocesses() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_engine_subprocess(proc)
    ]


def enable_faulthandler_signal_dump() -> None:
    """Arm this process to dump all thread stacks to stderr on
    FAULTHANDLER_DUMP_SIGNAL.

    faulthandler's handler is C code that walks thread states directly, so it
    produces stacks even when every Python thread is deadlocked or a C
    extension holds the GIL — the exact situations where py-spy (blocked by
    ptrace restrictions under gVisor) and normal Python signal handlers
    cannot help.
    """
    try:
        faulthandler.register(FAULTHANDLER_DUMP_SIGNAL, all_threads=True, chain=False)
    except (AttributeError, ValueError, OSError):
        # Non-main interpreter / unsupported platform: diagnostics only,
        # never fatal.
        logger.exception("Failed to register faulthandler dump signal.")


def faulthandler_dump_engine_processes(settle_secs: float = 3.0) -> None:
    """Request in-process stack dumps from every scheduler and detokenizer
    subprocess (see enable_faulthandler_signal_dump), then dump this
    process's own threads. Each target writes to its own stderr, which lands
    in the shared container log."""
    procs = collect_engine_subprocesses()
    if not procs:
        logger.error("No sglang engine subprocesses found for faulthandler dump.")
    for proc in procs:
        try:
            os.kill(proc.pid, FAULTHANDLER_DUMP_SIGNAL)
            logger.error(
                "Requested faulthandler stack dump from PID %s (signal %s).",
                proc.pid,
                FAULTHANDLER_DUMP_SIGNAL,
            )
        except OSError:
            logger.exception("Failed to signal PID %s for faulthandler dump.", proc.pid)
            continue
        # Serialize the dumps: every target writes to the same container log
        # fd, and faulthandler emits each frame as several small write()s, so
        # concurrent dumps interleave into an unreadable shuffle (observed on
        # the first prod capture). One second per process keeps each dump
        # contiguous; this path only runs on an already-terminal container.
        time.sleep(1.0)
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        logger.exception("faulthandler self-dump failed.")
    if procs and settle_secs > 0:
        # Give the last target a moment to flush before any follow-on
        # diagnostics or shutdown truncates the log.
        time.sleep(settle_secs)


def pyspy_dump_schedulers(scheduler_only=False):
    """py-spy dump on all scheduler in a local node."""
    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for py-spy dump.")
            return
        pids = [proc.pid for proc in procs]
    else:
        pids = [psutil.Process().pid]
    for pid in pids:
        for attempt, native_flag in enumerate(["--native", ""]):
            try:
                cmd = f"py-spy dump {native_flag} --pid {pid}".strip()
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, check=True
                )
                logger.error(f"Pyspy dump for PID {pid} ({cmd}):\n{result.stdout}")
                break
            except subprocess.CalledProcessError as e:
                logger.error(f"Pyspy failed ({cmd}). Error: {e.stderr}")
                if attempt == 1:
                    logger.error(f"All pyspy dump attempts failed for PID {pid}.")


def trigger_cuda_user_coredump(scheduler_only=False):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return
    else:
        procs = [psutil.Process()]

    for proc in procs:
        pipe_path = _resolve_cuda_coredump_pipe_path(proc)
        try:
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
            finally:
                os.close(fd)
            logger.error(
                "Triggered CUDA user coredump for PID %s via %s",
                proc.pid,
                pipe_path,
            )
        except FileNotFoundError:
            logger.error(
                "CUDA coredump pipe not found for PID %s: %s. Ensure "
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                "process initialized CUDA.",
                proc.pid,
                pipe_path,
            )
        except OSError as e:
            if e.errno == ENXIO:
                logger.error(
                    "CUDA coredump pipe has no reader for PID %s: %s",
                    proc.pid,
                    pipe_path,
                )
            else:
                logger.exception(
                    "Failed to trigger CUDA user coredump for PID %s via %s",
                    proc.pid,
                    pipe_path,
                )
