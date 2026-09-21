"""Windows task-wide limits and accounting, independent of agent policy."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache

from .process_owner import ProcessOwner, JobObjectExtendedLimitInformation, _windows_api

JOB_BASIC_ACCOUNTING = 1
JOB_COMPLETION_PORT = 7
JOB_EXTENDED_LIMITS = 9
JOB_CPU_RATE = 15
LIMIT_ACTIVE_PROCESSES = 0x8
LIMIT_JOB_MEMORY = 0x200
LIMIT_KILL_ON_CLOSE = 0x2000
CPU_ENABLE_HARD_CAP = 0x5
WAIT_TIMEOUT = 258
LIMIT_MESSAGES = {3: "process_count", 9: "memory", 10: "memory"}


class CpuRate(ctypes.Structure):
    _fields_ = [("flags", wintypes.DWORD), ("rate", wintypes.DWORD)]


class CompletionPort(ctypes.Structure):
    _fields_ = [("key", ctypes.c_void_p), ("port", wintypes.HANDLE)]


class Accounting(ctypes.Structure):
    _fields_ = [("user_time", ctypes.c_int64), ("kernel_time", ctypes.c_int64),
                ("period_user", ctypes.c_int64), ("period_kernel", ctypes.c_int64),
                ("page_faults", wintypes.DWORD), ("total_processes", wintypes.DWORD),
                ("active_processes", wintypes.DWORD), ("terminated_processes", wintypes.DWORD)]


@lru_cache(maxsize=1)
def resource_api():
    api = _windows_api()
    signatures = (
        ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                       wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        ("CreateIoCompletionPort", [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t,
                                    wintypes.DWORD], wintypes.HANDLE),
        ("GetQueuedCompletionStatus", [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                                       ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.DWORD], wintypes.BOOL),
    )
    for name, args, result in signatures:
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


class ResourceJob(ProcessOwner):
    def __init__(self, *, memory_bytes, cpu_percent, max_processes):
        super().__init__()
        self.port = None
        self.api = resource_api()
        try:
            self.port = self.api.CreateIoCompletionPort(ctypes.c_void_p(-1), None, 0, 1)
            if not self.port:
                raise ctypes.WinError(ctypes.get_last_error())
            self._set(JOB_COMPLETION_PORT, CompletionPort(1, self.port))
            limits = JobObjectExtendedLimitInformation()
            limits.basic_limit_information.limit_flags = LIMIT_KILL_ON_CLOSE | LIMIT_ACTIVE_PROCESSES | LIMIT_JOB_MEMORY
            limits.basic_limit_information.active_process_limit = max_processes
            limits.job_memory_limit = memory_bytes
            self._set(JOB_EXTENDED_LIMITS, limits)
            self._set(JOB_CPU_RATE, CpuRate(CPU_ENABLE_HARD_CAP, cpu_percent * 100))
        except BaseException:
            self.close()
            raise

    def _set(self, kind, value):
        if not self.api.SetInformationJobObject(self.handle, kind, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())

    def _query(self, kind, cls):
        value = cls()
        if not self.api.QueryInformationJobObject(self.handle, kind, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def sample(self):
        accounting = self._query(JOB_BASIC_ACCOUNTING, Accounting)
        memory = self._query(JOB_EXTENDED_LIMITS, JobObjectExtendedLimitInformation)
        event = ""
        # Never block the supervisor draining process notifications.
        for _ in range(256):
            message, key, value = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
            if not self.api.GetQueuedCompletionStatus(self.port, ctypes.byref(message), ctypes.byref(key),
                                                      ctypes.byref(value), 0):
                error = ctypes.get_last_error()
                if error == WAIT_TIMEOUT:
                    break
                raise ctypes.WinError(error)
            event = LIMIT_MESSAGES.get(message.value, event)
        return {"memory": memory.peak_job_memory_used, "processes": accounting.active_processes,
                "cpu_seconds": (accounting.user_time + accounting.kernel_time) / 10_000_000,
                "event": event}

    def close(self):
        try:
            super().close()
        finally:
            if self.port is not None:
                if not self.api.CloseHandle(self.port):
                    raise ctypes.WinError(ctypes.get_last_error())
                self.port = None
