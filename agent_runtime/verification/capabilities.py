"""Project-aware verification command capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model.agent.graph import AgentState


def build_verification_capabilities(state: AgentState) -> dict[str, Any]:
    repo = Path(str(state.get("repo_path") or "."))
    profile = state.get("project_profile") if isinstance(state.get("project_profile"), dict) else {}
    language = str(profile.get("primary_language") or _detect_language(repo) or "").lower()
    if language == "go" or (repo / "go.mod").exists():
        return _go_capabilities(repo)
    if language == "python":
        return _python_capabilities(repo)
    if language in {"javascript", "typescript"} or (repo / "package.json").exists():
        return _node_capabilities(repo)
    return {
        "language": language or "unknown",
        "allowed": [],
        "forbidden": _forbidden_policy(),
    }


def recommended_verification_command(state: AgentState) -> str:
    caps = build_verification_capabilities(state)
    for item in caps.get("allowed", []) or []:
        if isinstance(item, dict) and bool(item.get("recommended")):
            command = str(item.get("example") or item.get("command_pattern") or "").strip()
            if command:
                return command
    return ""


def _go_capabilities(repo: Path) -> dict[str, Any]:
    has_go_mod = (repo / "go.mod").exists()
    return {
        "language": "go",
        "framework": "go",
        "allowed": [
            {
                "id": "go_test_all",
                "command_pattern": "go test ./...",
                "example": "go test ./...",
                "scope": "repo",
                "cost": "medium",
                "recommended": True,
                "reason": "Go module or Go source detected.",
            },
            {
                "id": "go_test_package",
                "command_pattern": "go test ./<package>",
                "example": "go test ./...",
                "scope": "package",
                "cost": "low",
                "recommended": False,
            },
            {
                "id": "go_test_named",
                "command_pattern": "go test -run <TestName> ./...",
                "example": "go test -run TestName ./...",
                "scope": "test",
                "cost": "low",
                "recommended": False,
            },
        ],
        "forbidden": _forbidden_policy(),
        "metadata": {"has_go_mod": has_go_mod},
    }


def _python_capabilities(repo: Path) -> dict[str, Any]:
    has_pytest = (repo / "pytest.ini").exists() or (repo / "pyproject.toml").exists()
    example = "pytest" if has_pytest else "python -m compileall ."
    return {
        "language": "python",
        "framework": "pytest" if has_pytest else "python",
        "allowed": [
            {
                "id": "python_compile",
                "command_pattern": "python -m compileall .",
                "example": "python -m compileall .",
                "scope": "repo",
                "cost": "low",
                "recommended": not has_pytest,
            },
            {
                "id": "pytest",
                "command_pattern": "pytest",
                "example": "pytest",
                "scope": "repo",
                "cost": "medium",
                "recommended": has_pytest,
            },
        ],
        "forbidden": _forbidden_policy(),
        "metadata": {"recommended": example},
    }


def _node_capabilities(repo: Path) -> dict[str, Any]:
    return {
        "language": "javascript",
        "framework": "node",
        "allowed": [
            {
                "id": "npm_test",
                "command_pattern": "npm test",
                "example": "npm test",
                "scope": "repo",
                "cost": "medium",
                "recommended": True,
            },
            {
                "id": "npm_run_test",
                "command_pattern": "npm run test",
                "example": "npm run test",
                "scope": "repo",
                "cost": "medium",
                "recommended": False,
            },
        ],
        "forbidden": _forbidden_policy(),
    }


def _detect_language(repo: Path) -> str:
    if (repo / "go.mod").exists() or any(repo.glob("*.go")):
        return "go"
    if (repo / "package.json").exists():
        return "javascript"
    if any(repo.glob("*.py")):
        return "python"
    return ""


def _forbidden_policy() -> dict[str, bool]:
    return {
        "shell_operators": True,
        "network": True,
        "destructive": True,
        "background_process": True,
    }
