"""Guard and normalize LLM-selected verification commands."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SHELL_OPERATORS = {"&&", "||", ";", "|", ">", ">>", "<", "`"}
DENIED_EXECUTABLES = {
    "bash",
    "cmd",
    "curl",
    "del",
    "git",
    "powershell",
    "pwsh",
    "rm",
    "rmdir",
    "wget",
}


@dataclass
class VerificationDecision:
    allowed: bool
    command: str
    reason: str
    violations: list[str] = field(default_factory=list)
    normalized_from: str = ""
    capability_id: str = ""
    scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "command": self.command,
            "reason": self.reason,
            "violations": self.violations,
            "normalized_from": self.normalized_from,
            "capability_id": self.capability_id,
            "scope": self.scope,
        }


def validate_verification_command(
    command: str,
    capabilities: dict[str, Any],
) -> VerificationDecision:
    raw = str(command or "").strip()
    if not raw:
        return VerificationDecision(False, "", "Verification command is empty.", ["empty_command"])
    argv, violation = _safe_split(raw)
    if violation:
        return VerificationDecision(False, raw, violation, ["parse_error"])
    common_violation = _common_violation(argv)
    if common_violation:
        return VerificationDecision(False, raw, common_violation, [common_violation])
    language = str(capabilities.get("language") or "").lower()
    if language == "go":
        return _validate_go(argv, raw)
    if language == "python":
        return _validate_python(argv, raw)
    if language in {"javascript", "typescript"}:
        return _validate_node(argv, raw)
    return VerificationDecision(
        False,
        raw,
        "No verification capabilities are available for this project.",
        ["unsupported_project"],
    )


def invalid_verification_resolution(
    command: str,
    decision: VerificationDecision,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    examples = [
        str(item.get("example") or item.get("command_pattern") or "").strip()
        for item in capabilities.get("allowed", []) or []
        if isinstance(item, dict) and str(item.get("example") or item.get("command_pattern") or "").strip()
    ]
    return {
        "kind": "invalid_verification_command",
        "action": "run_shell_command",
        "required_next_action": "run_shell_command",
        "reason": decision.reason,
        "details": {
            "command": command,
            "decision": decision.to_dict(),
            "allowed_examples": examples[:6],
        },
    }


def _safe_split(command: str) -> tuple[list[str], str]:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return [], str(exc)
    if not argv:
        return [], "Command parsed to an empty argv."
    return argv, ""


def _common_violation(argv: list[str]) -> str:
    for token in argv:
        if token in SHELL_OPERATORS:
            return "Shell control operators are not allowed for verification."
    executable = Path(argv[0]).name.lower()
    if executable in DENIED_EXECUTABLES:
        return f"Executable is not allowed for verification: {executable}"
    return ""


def _validate_go(argv: list[str], raw: str) -> VerificationDecision:
    if len(argv) >= 2 and argv[0] == "go" and argv[1] == "test":
        normalized = _normalize_go_test(argv)
        if normalized:
            capability_id, scope = _go_command_capability(argv, normalized)
            return VerificationDecision(
                True,
                normalized,
                "Matches Go verification capability.",
                normalized_from=raw if normalized != raw else "",
                capability_id=capability_id,
                scope=scope,
            )
    return VerificationDecision(
        False,
        raw,
        "Go verification commands must use go test with an allowed package or test scope.",
        ["unsupported_go_command"],
    )


def _normalize_go_test(argv: list[str]) -> str:
    args = argv[2:]
    if not args:
        return "go test ./..."
    allowed_flags: list[str] = []
    packages: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-run":
            if index + 1 >= len(args):
                return ""
            pattern = args[index + 1]
            if not _safe_go_test_pattern(pattern):
                return ""
            allowed_flags.extend(["-run", pattern])
            index += 2
            continue
        if arg.startswith("-run="):
            pattern = arg.split("=", 1)[1]
            if not _safe_go_test_pattern(pattern):
                return ""
            allowed_flags.append(arg)
            index += 1
            continue
        if arg.startswith("-"):
            return ""
        if _safe_go_package(arg):
            packages.append(arg)
            index += 1
            continue
        return ""
    if not packages:
        packages = ["./..."]
    return " ".join(["go", "test", *allowed_flags, *packages])


def _safe_go_package(value: str) -> bool:
    return value == "./..." or bool(re.fullmatch(r"\./[A-Za-z0-9_./-]+(?:/\.\.\.)?", value))


def _safe_go_test_pattern(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_./^$|()-]+", value))


def _go_command_capability(argv: list[str], normalized: str) -> tuple[str, str]:
    if any(arg == "-run" or arg.startswith("-run=") for arg in argv[2:]):
        return "go_test_named", "test"
    if normalized == "go test ./...":
        return "go_test_all", "repo"
    return "go_test_package", "package"


def _validate_python(argv: list[str], raw: str) -> VerificationDecision:
    if argv == ["python", "-m", "compileall", "."]:
        return VerificationDecision(
            True,
            raw,
            "Matches Python verification capability.",
            capability_id="python_compile",
            scope="repo",
        )
    if argv == ["pytest"]:
        return VerificationDecision(
            True,
            raw,
            "Matches Python verification capability.",
            capability_id="pytest",
            scope="repo",
        )
    return VerificationDecision(False, raw, "Unsupported Python verification command.", ["unsupported_python_command"])


def _validate_node(argv: list[str], raw: str) -> VerificationDecision:
    if argv in (["npm", "test"], ["npm", "run", "test"]):
        return VerificationDecision(
            True,
            raw,
            "Matches Node verification capability.",
            capability_id="npm_test" if argv == ["npm", "test"] else "npm_run_test",
            scope="repo",
        )
    return VerificationDecision(False, raw, "Unsupported Node verification command.", ["unsupported_node_command"])
