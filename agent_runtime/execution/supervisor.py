"""Non-tool-executing parent supervision and task-level lifecycle."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

from tools.shell_tools.process import launch_process
from .artifacts import read_json
from .resources import TaskProcessScope


class ExecutionDeadline(TimeoutError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ExecutionSupervisor:
    def __init__(self, directory, limits, resources, deadline, cancel_event=None):
        self.directory, self.limits, self.deadline = directory, limits, deadline
        self.cancel_event = cancel_event
        self.scope = TaskProcessScope(resources)
        self.processes = {}
        self.owners = []
        self.logs = []
        self.phase_deadline = 0.0
        self.last_progress = None

    def launch(self, role, environment):
        self.check()
        self.phase_deadline = min(self.deadline, time.monotonic() + self.limits.startup_timeout)
        log = (self.directory / f"{role}.log").open("wb")
        self.logs.append(log)
        try:
            process, owner = launch_process(
                [sys.executable, "-m", f"agent_runtime.execution.{role}", str(self.directory)],
                cwd=Path(__file__).resolve().parents[2], env=environment, output=log, scope=self.scope,
                startup_timeout=max(0.01, self.phase_deadline - time.monotonic()), cancel_event=self.cancel_event)
        except TimeoutError as exc:
            raise ExecutionDeadline("STARTUP_DEADLINE_EXCEEDED", str(exc)) from exc
        self.processes[role] = process
        self.owners.append(owner)
        return process

    def check(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise KeyboardInterrupt()
        now = time.monotonic()
        if now >= self.deadline:
            raise ExecutionDeadline("TASK_DEADLINE_EXCEEDED", "Execution task deadline exceeded")
        self.scope.sample()
        if self.phase_deadline and now >= self.phase_deadline:
            raise ExecutionDeadline("STARTUP_DEADLINE_EXCEEDED", "Execution process startup deadline exceeded")

    def status(self):
        self.check()
        path = self.directory / "host_status.json"
        status = read_json(path, max_bytes=8192) if path.exists() else {}
        deadline = status.get("deadline", 0)
        if deadline and time.monotonic() >= deadline:
            raise ExecutionDeadline("TOOL_DEADLINE_EXCEEDED",
                                    f"Tool Host deadline exceeded: {status}; side effects may be unknown; do not replay")
        if status.get("phase") == "failed":
            raise RuntimeError(f"Tool Host failed: {status.get('error')}")
        host = self.processes.get("host")
        if host is not None and host.poll() is not None and status.get("phase") != "done":
            raise RuntimeError(f"Tool Host exited with code {host.returncode}")
        return status

    def wait_ready(self):
        while self.status().get("phase") != "ready":
            time.sleep(0.05)
        self.phase_deadline = 0

    def wait_worker(self, on_progress):
        worker = self.processes["worker"]
        while True:
            self.status()
            path = self.directory / "progress.json"
            if path.exists():
                progress = read_json(path, max_bytes=16384)
                self.phase_deadline = 0
                if progress != self.last_progress:
                    on_progress(progress)
                    self.last_progress = progress
            if worker.poll() is not None:
                self.phase_deadline = 0
                if worker.returncode:
                    raise RuntimeError(f"Execution worker exited with code {worker.returncode}; inspect worker.log")
                return
            time.sleep(0.1)

    def wait_result(self):
        # Also cover a Host stuck between its last operation and finalization.
        stop = min(self.deadline, time.monotonic() + self.limits.operation_timeout)
        while self.status().get("phase") != "done":
            if time.monotonic() >= stop:
                raise ExecutionDeadline("TOOL_DEADLINE_EXCEEDED", "Tool Host finalization deadline exceeded")
            time.sleep(0.05)
        self.check()
        return read_json(self.directory / "host_result.json")

    def close(self):
        errors = []
        deadline = time.monotonic() + self.limits.cleanup_timeout
        try:
            self.scope.close(self.limits.cleanup_timeout)
        except Exception as exc:
            errors.append(str(exc))
        # Scope termination first; never depend on host finally or worker cooperation.
        for owner in self.owners:
            try:
                owner.close()
            except Exception as exc:
                errors.append(str(exc))
        for process in self.processes.values():
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
                process.stdin.close()
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(str(exc))
        for log in self.logs:
            log.close()
        if errors:
            raise RuntimeError("; ".join(errors))
