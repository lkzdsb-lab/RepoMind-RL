"""Task ownership and explicit hard/monitored resource capabilities."""
from __future__ import annotations

import os
import time

import psutil

from tools.shell_tools.resource_job import ResourceJob
from .backend import BackendUnavailable
from .contracts import ResourceLimits, ResourceUsage


class ResourceExceeded(RuntimeError):
    pass


class TaskProcessScope:
    """ 任务资源分配"""
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.usage = ResourceUsage(limits=limits)
        self.job = None
        self.roots = []
        self.known = {}
        self.started = time.monotonic()
        self.last_sample = self.started
        self.last_cpu = 0.0
        if os.name == "nt":
            try:
                self.job = ResourceJob(memory_bytes=limits.memory_limit_mb * 1024**2,
                                       cpu_percent=limits.cpu_limit_percent, max_processes=limits.max_processes)
            except OSError as exc:
                if limits.mode == "strict":
                    raise BackendUnavailable(f"Task hard limits unavailable: {exc}") from exc
        if self.job is None and limits.mode == "strict":
            raise BackendUnavailable("Task hard limits unavailable on this platform; explicitly select best_effort to use monitoring")
        capability = "hard" if self.job else "monitored"
        self.usage.capabilities = dict.fromkeys(("memory", "cpu", "process_count"), capability)

    def attach(self, process):
        # Called before the launch gate is released, not after discovery by PID.
        self.roots.append(process)
        if self.job:
            self.job.attach(process)
        else:
            item = psutil.Process(process.pid)
            self.known[(item.pid, item.create_time())] = item

    def _processes(self):
        for item in list(self.known.values()):
            try:
                if not item.is_running():
                    continue
                for child in item.children(recursive=True):
                    self.known[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                continue
        return [item for item in self.known.values() if item.is_running()]

    def sample(self):
        now = time.monotonic()
        if self.job:
            values = self.job.sample()
        else:
            memory = cpu = count = 0
            for item in self._processes():
                try:
                    memory += item.memory_info().rss
                    times = item.cpu_times()
                    cpu += times.user + times.system
                    count += 1
                except psutil.NoSuchProcess:
                    continue
            values = {"memory": memory, "cpu_seconds": cpu, "processes": count, "event": ""}
            if memory > self.limits.memory_limit_mb * 1024**2:
                values["event"] = "memory"
            elif count > self.limits.max_processes:
                values["event"] = "process_count"
        self.usage.peak_memory_bytes = max(self.usage.peak_memory_bytes, values["memory"])
        self.usage.peak_processes = max(self.usage.peak_processes, values["processes"])
        self.usage.cpu_percent = round(max(0, values["cpu_seconds"] - self.last_cpu) * 100 /
                                       max(0.001, now - self.last_sample) / (os.cpu_count() or 1), 2)
        self.last_sample, self.last_cpu = now, values["cpu_seconds"]
        self.usage.peak_cpu_percent = max(self.usage.peak_cpu_percent, self.usage.cpu_percent)
        # Best-effort CPU monitoring terminates; it does not claim to throttle.
        if not self.job and now - self.started > 1 and self.usage.cpu_percent > self.limits.cpu_limit_percent:
            values["event"] = "cpu"
        if values["event"]:
            self.usage.limit_event = values["event"]
            raise ResourceExceeded(f"Task {values['event']} limit exceeded; limits={self.limits.model_dump()}")

    def close(self, timeout=5):
        if self.job:
            if self.job.handle is None:
                return
            # Retain the job until its active count is zero, so success means observed cleanup.
            if not self.job.api.TerminateJobObject(self.job.handle, 1):
                self.job.close()
                raise RuntimeError("Could not terminate task job")
            deadline = time.monotonic() + timeout
            try:
                while self.job.sample()["processes"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Task processes did not exit within cleanup budget")
                    time.sleep(0.05)
            finally:
                self.job.close()
        else:
            processes = self._processes()
            errors = []
            for item in reversed(processes):
                try:
                    item.kill()
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as exc:
                    errors.append(str(exc))
            _, alive = psutil.wait_procs(processes, timeout=timeout)
            if alive or errors:
                raise RuntimeError(f"Task cleanup incomplete: {[p.pid for p in alive]}; {errors}")
