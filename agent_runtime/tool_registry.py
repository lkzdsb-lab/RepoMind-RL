"""
Tool registry and adapters.

Tools expose small dictionary contracts so executor, LangGraph nodes, and future
RL policies can share the same sandbox interface.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping

from config import FileConfig
from model.agent.tools import ToolSpec
from tools.code_tools.code import search_code
from tools.code_tools.context import build_codebase_context, search_code_context
from tools.code_tools.file import list_files, read_file
from tools.git_tools.diff import git_diff
from tools.go_tools.go_test import run_command
from loguru import logger


def reduce_search_code_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    return {"candidate_files": _extract_files_from_search(output.get("matches", []))}


def reduce_search_code_context_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    context = output.get("selected_code_context")
    if not isinstance(context, dict):
        context = output

    files = []
    for item in context.get("files", []):
        path = item.get("path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    for item in context.get("functions", []):
        path = item.get("file_path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    for item in context.get("symbols", []):
        path = item.get("file_path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    for item in context.get("api_routes", []):
        path = item.get("file_path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    for item in context.get("db_models", []):
        path = item.get("file_path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    return {
        "candidate_files": files[:10],
        "code_context": output,
        "selected_code_context": context if context is not output else {},
        "code_context_query_plan": output.get("query_plan", {}),
        "code_context_rerank": output.get("code_context_rerank", {}),
    }


def reduce_run_tests_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    if output.get("skipped"):
        return {"status": "running"}
    return {
        "test_results": state.get("test_results", []) + [output],
        "status": "testing",
    }


def reduce_git_diff_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    diff = output.get("diff", "")
    return {
        "patch": diff or None,
        "patch_summary": summarize_diff(diff),
    }


def _extract_files_from_search(matches: list[str]) -> list[str]:
    files: list[str] = []
    for line in matches:
        path = line.split(":", 1)[0]
        if path.startswith("./"):
            path = path[2:]
        if path and path not in files:
            files.append(path)
    return files[:10]


def summarize_diff(diff: str) -> str:
    if not diff:
        return "当前工作区没有 git diff。"
    added = sum(
        1
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return f"当前补丁包含 {added} 行新增、{removed} 行删除。"

# 工具注册
class ToolRegistry:
    def __init__(self, include_defaults: bool = True) -> None:
        self._tools: dict[str, ToolSpec] = {}
        if include_defaults:
            self.register_defaults()

    # 后续新注册 tools
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            logger.warning("overriding registered tool name={}", spec.name)
        else:
            logger.debug("registering tool name={}", spec.name)
        self._tools[spec.name] = spec

    def run(self, name: str, repo_path: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if name not in self._tools:
            logger.warning("unknown tool requested name={}", name)
            return {"error": f"Unknown tool: {name}"}
        logger.debug("tool registry dispatch name={} repo_path={} args={}", name, repo_path, args or {})
        return self._tools[name].runner(repo_path, args or {})

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def items(self) -> Mapping[str, ToolSpec]:
        return MappingProxyType(dict(self._tools))

    def register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="build_codebase_context",
                description="Build or refresh the local codebase context index.",
                runner=lambda repo, args: build_codebase_context(
                    repo,
                    index_path=str(args.get("index_path", ".repomind/codebase_context/index.json")),
                    force_rebuild=bool(args.get("force_rebuild", False)),
                ),
            )
        )
        self.register(
            ToolSpec(
                name="search_code_context",
                description="Search the structured codebase context index.",
                reducer=reduce_search_code_context_output,
                runner=lambda repo, args: search_code_context(
                    repo,
                    str(args.get("query", "")),
                    limit=int(args.get("limit", args.get("max_results", 10))),
                    index_path=str(args.get("index_path", ".repomind/codebase_context/index.json")),
                    force_rebuild=bool(args.get("force_rebuild", False)),
                ),
            )
        )
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
                reducer=reduce_search_code_output,
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
                reducer=reduce_run_tests_output,
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
                reducer=reduce_git_diff_output,
                runner=lambda repo, args: git_diff(repo),
            )
        )
