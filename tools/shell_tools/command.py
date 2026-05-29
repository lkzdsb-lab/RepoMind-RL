"""Guarded command execution primitive."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


DENIED_EXECUTABLES = {
    "rm",
    "rmdir",
    "mkfs",
    "dd",
    "shutdown",
    "reboot",
    "sudo",
    "su",
    "chmod",
    "chown",
}

DENIED_TOKENS = {
    "reset --hard",
    "checkout --",
    "clean -fd",
    "clean -fx",
    "curl |",
    "wget |",
    "> /dev/",
}


def run_shell_command(repo_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    command = str(args.get("command") or "").strip()
    if not command:
        return {
            "error": "run_shell_command requires command.",
            "needs_more_context": True,
            "exit_code": -1,
        }

    allow_shell = bool(args.get("allow_shell", False))
    timeout = max(1, min(int(_safe_int(args.get("timeout"), 120)), 1800))
    purpose = _clean_purpose(args.get("purpose"))
    reason = str(args.get("reason") or "").strip()[:500]

    denied = _deny_reason(command, allow_shell=allow_shell)
    if denied:
        return {
            "error": denied,
            "needs_more_context": True,
            "command": command,
            "purpose": purpose,
            "exit_code": -1,
        }

    repo = Path(repo_path).resolve()
    started = time.perf_counter()
    try:
        if allow_shell:
            result = subprocess.run(
                command,
                cwd=repo,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            argv = shlex.split(command)
            if not argv:
                return {"error": "Command parsed to an empty argv.", "exit_code": -1}
            result = subprocess.run(
                argv,
                cwd=repo,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        return {
            "command": command,
            "purpose": purpose,
            "reason": reason,
            "exit_code": -1,
            "stdout": _tail(exc.stdout or "", 6000),
            "stderr": _tail((exc.stderr or "") + "\ncommand timeout", 6000),
            "duration_ms": round(duration_ms, 1),
            "timeout": True,
        }
    except (OSError, ValueError) as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        return {
            "command": command,
            "purpose": purpose,
            "reason": reason,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "duration_ms": round(duration_ms, 1),
            "error": str(exc),
        }

    duration_ms = (time.perf_counter() - started) * 1000
    return {
        "command": command,
        "purpose": purpose,
        "reason": reason,
        "exit_code": result.returncode,
        "stdout": _tail(result.stdout, 6000),
        "stderr": _tail(result.stderr, 6000),
        "duration_ms": round(duration_ms, 1),
        "shell": allow_shell,
    }


def _deny_reason(command: str, *, allow_shell: bool) -> str:
    """
        对 command line 的限制
    """
    lowered = " ".join(command.lower().split())
    if allow_shell and any(token in lowered for token in [";", "&&", "||", "`", "$("]):
        return "Shell control operators are not allowed in run_shell_command."
    for token in DENIED_TOKENS:
        if token in lowered:
            return f"Denied potentially destructive command pattern: {token}"
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return str(exc)
    if not argv:
        return "Command parsed to an empty argv."
    executable = Path(argv[0]).name.lower()
    if executable in DENIED_EXECUTABLES:
        return f"Denied executable: {executable}"
    if executable == "git" and len(argv) >= 3:
        sub = argv[1].lower()
        flags = {item.lower() for item in argv[2:]}
        if sub == "reset" and "--hard" in flags:
            return "Denied destructive git reset --hard."
        if sub == "checkout" and "--" in flags:
            return "Denied destructive git checkout --."
        if sub == "clean":
            return "Denied destructive git clean."
    return ""


def _clean_purpose(value: Any) -> str:
    purpose = str(value or "diagnostic").strip().lower()
    if purpose not in {"verification", "diagnostic", "search", "build"}:
        return "diagnostic"
    return purpose


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tail(value: Any, limit: int) -> str:
    text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value or "")
    return text[-limit:]
