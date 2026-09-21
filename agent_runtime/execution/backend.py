"""Trusted local task environment and process ownership."""
from __future__ import annotations

import os
from pathlib import Path
import time

import psutil

from tools.shell_tools.command import validate_command_argv
from tools.shell_tools.process import ManagedProcess


class BackendUnavailable(RuntimeError):
    pass


class Deadline:
    def __init__(self, deadline):
        self.deadline = deadline

    def remaining(self, maximum=15):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Execution deadline exceeded")
        return min(maximum, value)


def command_environment(work: Path, extra: dict, secret_names: set[str]) -> dict:
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE",
               "HOME", "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL", "VIRTUAL_ENV",
               "GOROOT", "GOPATH", "JAVA_HOME"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment.update(extra)
    secrets = {name.upper() for name in secret_names}
    environment = {key: value for key, value in environment.items() if key.upper() not in secrets}
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                        "TEMP": str(work), "TMP": str(work), "TMPDIR": str(work)})
    return environment


class ProcessManager(Deadline):
    """  负责一次 execution task 中的全部本地命令"""
    def __init__(self, root: Path, work: Path, deadline: float, environment: dict):
        super().__init__(deadline)
        self.root, self.work, self.environment = root.resolve(), work.resolve(), environment
        self.processes: dict[str, ManagedProcess] = {}

    def start(self, process_id, argv, cwd):
        self.remaining()
        # 命令过滤
        reason = validate_command_argv(argv)
        if reason:
            raise PermissionError(reason)
        directory = self.work if cwd == "@work" else (self.root / cwd).resolve()
        if not directory.is_dir() or not directory.is_relative_to(self.root):
            raise ValueError("cwd must be an existing project directory or @work")
        process = ManagedProcess(argv, cwd=directory, env=self.environment)
        self.processes[process_id] = process
        return process

    def owns_listener(self, process_id, port):
        """  校验 HTTP 端口归属 """
        process = self.processes.get(process_id)
        if process is None or process.process.poll() is not None:
            return False
        try:
            owner = psutil.Process(process.process.pid)
            pids = {owner.pid, *(p.pid for p in owner.children(recursive=True))}
            listeners = [c for c in psutil.net_connections(kind="tcp4")
                         if c.status == psutil.CONN_LISTEN and c.laddr.port == port
                         and c.laddr.ip in {"127.0.0.1", "0.0.0.0"}]
            return bool(listeners) and all(c.pid in pids for c in listeners)
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            raise PermissionError("Cannot establish local service ownership") from exc

    def cleanup(self):
        """ 任务完成、取消、超时或 worker 崩溃时，会遍历所有 ManagedProcess 并关闭。"""
        errors = []
        for process in self.processes.values():
            try:
                process.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))
