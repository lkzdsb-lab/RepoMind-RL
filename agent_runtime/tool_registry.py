"""
Tool registry and adapters.

Tools expose small dictionary contracts so executor, LangGraph nodes, and future
RL policies can share the same sandbox interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from config import FileConfig
from tools.code_tools.code import search_code
from tools.code_tools.file import list_files, read_file
from tools.git_tools.diff import git_diff
from tools.go_tools.go_test import run_command


ToolFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    runner: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.register_defaults()

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def run(self, name: str, repo_path: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        return self._tools[name].runner(repo_path, args or {})

    def names(self) -> list[str]:
        return sorted(self._tools)

    def register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="list_files",
                description="List repository files with common generated directories ignored.",
                runner=lambda repo, args: list_files(
                    repo,
                    max_files=int(args.get("max_files", FileConfig.MAX_READ_AMOUNT)),
                ),
            )
        )
        self.register(
            ToolSpec(
                name="search_code",
                description="Search code using ripgrep when available.",
                runner=lambda repo, args: search_code(
                    repo,
                    str(args.get("query", "")),
                    max_results=int(args.get("max_results", 30)),
                ),
            )
        )
        self.register(
            ToolSpec(
                name="read_file",
                description="Read a repository file by relative path.",
                runner=lambda repo, args: read_file(
                    repo,
                    str(args.get("file_path", "")),
                    max_chars=int(args.get("max_chars", 8000)),
                ),
            )
        )
        self.register(
            ToolSpec(
                name="run_tests",
                description="Run the configured verification command.",
                runner=lambda repo, args: run_command(
                    repo,
                    str(args.get("command", "pytest")),
                    timeout=int(args.get("timeout", 120)),
                ),
            )
        )
        self.register(
            ToolSpec(
                name="git_diff",
                description="Return the current git diff.",
                runner=lambda repo, args: git_diff(repo),
            )
        )

