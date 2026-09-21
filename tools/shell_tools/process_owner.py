"""Platform-specific ownership of one local command's process tree."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache
import os
import signal
import subprocess
from typing import Any


JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_TERMINATION_EXIT_CODE = 1


class JobObjectBasicLimitInformation(ctypes.Structure):
    """Native JOBOBJECT_BASIC_LIMIT_INFORMATION layout, including alignment."""

    _fields_ = [
        ("process_time_limit", ctypes.c_int64),
        ("job_time_limit", ctypes.c_int64),
        ("limit_flags", wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class JobObjectExtendedLimitInformation(ctypes.Structure):
    """Native JOBOBJECT_EXTENDED_LIMIT_INFORMATION passed to kernel32."""

    _fields_ = [
        ("basic_limit_information", JobObjectBasicLimitInformation),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


@lru_cache(maxsize=1)
def _windows_api() -> Any:
    """Bind native signatures once; load kernel32 only on the Windows path."""
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = (
        ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
    )
    for name, argument_types, return_type in signatures:
        function = getattr(api, name)
        function.argtypes = argument_types
        function.restype = return_type
    return api


def _create_windows_job(api: Any) -> int:
    handle = api.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = JobObjectExtendedLimitInformation()
    limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = api.SetInformationJobObject(
        handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
    )
    if not configured:
        error = ctypes.WinError(ctypes.get_last_error())
        api.CloseHandle(handle)
        raise error
    return handle


class ProcessOwner:
    """负责一个命令的整个进程树生命周期：Windows Job Object 或 POSIX 进程组。"""

    def __init__(self) -> None:
        self.pid: int | None = None
        self.handle: int | None = None
        self.api = _windows_api() if os.name == "nt" else None
        if self.api is not None:
            self.handle = _create_windows_job(self.api)

    def attach(self, process: subprocess.Popen[bytes]) -> None:
        self.pid = process.pid
        if self.handle is not None:
            # Popen owns this process handle. The Job Object receives membership,
            # not ownership of Popen's handle; only our job handle is closed here.
            if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle is not None:
            error = None
            if not self.api.TerminateJobObject(self.handle, JOB_TERMINATION_EXIT_CODE):
                error = ctypes.WinError(ctypes.get_last_error())
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None
            if error is not None:
                raise error
        elif os.name != "nt" and self.pid is not None:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.pid = None
