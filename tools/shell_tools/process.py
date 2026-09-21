"""Reusable owned local processes; lifecycle control, not a security sandbox."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tools.shell_tools.process_owner import ProcessOwner


def child_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def launch_process(argv, *, cwd, env=None, merge_output=False, output=None,
                   scope=None, startup_timeout=30, cancel_event=None):
    if not argv or not argv[0] or any("\0" in part for part in argv):
        raise ValueError("Command requires nonempty argv without null bytes")
    payload = (json.dumps(argv, ensure_ascii=True) + "\n").encode("ascii")
    if len(payload) >= 128000:
        raise ValueError("Command exceeds launch frame limit")
    # Attach task ownership before releasing the command launch gate.
    owner = ProcessOwner()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("process_launcher.py"))],
            cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=output if output is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_output or output is not None else subprocess.PIPE,
            start_new_session=os.name != "nt", **child_options())
        if scope is not None:
            scope.attach(process)
        owner.attach(process)
        errors = []
        done = threading.Event()
        def deliver():
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()
        writer = threading.Thread(target=deliver, daemon=True)
        writer.start()
        deadline = time.monotonic() + startup_timeout
        while not done.wait(0.05):
            if cancel_event is not None and cancel_event.is_set():
                raise KeyboardInterrupt()
            if time.monotonic() >= deadline:
                raise TimeoutError("Process launch handshake exceeded startup deadline")
        if errors:
            raise errors[0]
        return process, owner
    except BaseException:
        # Attempt both cleanup steps even when one fails.
        try:
            owner.close()
        finally:
            if process is not None:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except OSError:
                    pass
        raise


class ManagedProcess:
    """ 它表示一个可以被 subagent 持续操作的命令"""
    def __init__(self, argv, *, cwd, env=None):
        self.process, self.owner = launch_process(argv, cwd=cwd, env=env, merge_output=True)
        self.chunks = deque()
        self.retained_bytes = self.bytes_seen = 0
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while data := self.process.stdout.read1(1024):
                with self.lock:
                    self.bytes_seen += len(data)
                    self.chunks.append(data)
                    self.retained_bytes += len(data)
                    while self.retained_bytes > 32768:
                        self.retained_bytes -= len(self.chunks.popleft())
        finally:
            self.process.stdout.close()

    def snapshot(self, cursor=0):
        code = self.process.poll()
        if code is not None:
            self.reader.join(timeout=0.2)
        with self.lock:
            output = b"".join(self.chunks)
            base = self.bytes_seen - len(output)
            start = max(min(cursor, self.bytes_seen), base, self.bytes_seen - 8000)
            return {"exit_code": code, "running": code is None,
                    "output_complete": code is not None and not self.reader.is_alive(),
                    "output": output[start-base:].decode("utf-8", errors="replace"),
                    "output_bytes": self.bytes_seen, "next_cursor": self.bytes_seen,
                    "truncated": start > cursor, "partial": start > 0 or self.reader.is_alive()}

    def close(self):
        self.owner.close()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.reader.join(timeout=1)


def run_process(argv, *, cwd, timeout, env=None):
    """Synchronous adapter with separate stdout/stderr and tree cleanup."""
    process, owner = launch_process(argv, cwd=Path(cwd), env=env)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(argv, process.returncode,
            stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        owner.close()
        stdout, stderr = process.communicate(timeout=5)
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from exc
    finally:
        owner.close()
        process.wait(timeout=5)
