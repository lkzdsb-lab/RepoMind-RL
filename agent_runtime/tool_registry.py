"""
Tool registry and adapters.

Tools expose small dictionary contracts so executor, LangGraph nodes, and future
RL policies can share the same sandbox interface.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping

from config import FileConfig
from model.agent.tools import ToolSpec, normalize_tool_result, run_tool_spec
from tools.code_tools.code import search_code
from tools.code_tools.context import build_codebase_context, search_code_context
from tools.code_tools.edit import apply_code_patch
from tools.code_tools.file import list_files, read_file
from tools.code_tools.search_text import search_text
from tools.git_tools.diff import git_diff
from tools.plan_tools.mode import enter_plan_mode, exit_plan_mode
from tools.shell_tools.command import run_shell_command
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="allow")


class BuildCodebaseContextInput(ToolInput):
    index_path: str = ".repomind/codebase_context/index.json"
    force_rebuild: bool = False


class SearchCodeContextInput(ToolInput):
    query: str = ""
    limit: int | None = Field(default=None, ge=1, le=100)
    max_results: int = Field(default=10, ge=1, le=100)
    index_path: str = ".repomind/codebase_context/index.json"
    force_rebuild: bool = False


class SearchCodeInput(ToolInput):
    query: str = ""
    max_results: int = Field(default=30, ge=1, le=200)


class SearchTextInput(ToolInput):
    pattern: str = Field(min_length=1)
    regex: bool = True
    globs: list[str] = Field(default_factory=list)
    context_lines: int = Field(default=0, ge=0, le=5)
    max_results: int = Field(default=50, ge=1, le=200)
    timeout: int = Field(default=20, ge=5, le=120)


class ReadFileInput(ToolInput):
    file_path: str = Field(min_length=1)
    max_chars: int = Field(default=8000, ge=1, le=200000)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class RunShellCommandInput(ToolInput):
    command: str = Field(min_length=1)
    purpose: Literal["verification", "diagnostic", "search", "build"] = "diagnostic"
    timeout: int = Field(default=120, ge=1, le=1800)
    reason: str = ""
    allow_shell: bool = False


class RunTestsInput(ToolInput):
    command: str = "pytest"
    timeout: int = Field(default=120, ge=1, le=1800)


class ApplyCodePatchChangeInput(ToolInput):
    file_path: str = Field(min_length=1)
    operation: Literal["replace", "create"] = "replace"
    old_text: str | None = None
    new_text: str
    expected_occurrences: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_operation_fields(self) -> "ApplyCodePatchChangeInput":
        if self.operation == "replace" and not self.old_text:
            raise ValueError("old_text is required for replace changes")
        return self


class ApplyCodePatchInput(ToolInput):
    changes: list[ApplyCodePatchChangeInput] = Field(min_length=1)
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    uncertainty_questions: list[str] = Field(default_factory=list)
    dry_run: bool = False


class EnterPlanModeInput(ToolInput):
    technical_plan: str = ""
    risks: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ExitPlanModeInput(ToolInput):
    evaluation: str = ""
    approved: bool = False
    remaining_uncertainties: list[str] = Field(default_factory=list)
    next_step: str = ""


def reduce_search_code_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    return {"candidate_files": _extract_files_from_search(output.get("matches", []))}


def reduce_search_text_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    files = list(state.get("candidate_files", []))
    for item in output.get("matches", []) or []:
        path = item.get("file_path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    return {"candidate_files": files[:20]}


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
    """
        获取简练的输出
    """
    if output.get("skipped"):
        return {"status": "running"}
    verification_commands = state.get("verification_commands", []) + [
        {
            "tool": "run_tests",
            "command": output.get("command", ""),
            "exit_code": output.get("exit_code"),
            "purpose": "verification",
        }
    ]
    updates: Dict[str, Any] = {
        "test_results": state.get("test_results", []) + [output],
        "verification_commands": verification_commands,
        "status": "testing",
    }
    if output.get("exit_code") == 0:
        updates["verification_stale"] = False
        updates["last_verified_edit_loop"] = state.get("loop_count", 0)
    return updates


def reduce_run_shell_command_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    command_results = state.get("command_results", []) + [output]
    updates: Dict[str, Any] = {"command_results": command_results}
    if output.get("purpose") == "verification":
        verification_commands = state.get("verification_commands", []) + [
            {
                "tool": "run_shell_command",
                "command": output.get("command", ""),
                "exit_code": output.get("exit_code"),
                "duration_ms": output.get("duration_ms"),
                "reason": output.get("reason", ""),
                "purpose": "verification",
            }
        ]
        updates.update(
            {
                "test_results": state.get("test_results", []) + [output],
                "verification_commands": verification_commands,
                "status": "testing",
            }
        )
        if output.get("exit_code") == 0:
            updates["verification_stale"] = False
            updates["last_verified_edit_loop"] = state.get("loop_count", 0)
    return updates


def reduce_git_diff_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    if output.get("unsupported") and output.get("reason") == "not_git_repo":
        return {
            "patch": None,
            "patch_summary": "目标目录不是 Git 仓库，已跳过 git diff。",
            "is_git_repo": False,
        }
    diff = output.get("diff", "")
    change_summary = summarize_diff_detail(diff, source="git_diff")
    return {
        "patch": diff or None,
        "patch_summary": change_summary["summary"],
    }


def reduce_apply_code_patch_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    edit_results = state.get("edit_results", []) + [output]
    edited_files = list(state.get("edited_files", []))
    if output.get("applied"):
        for path in output.get("changed_files", []) or []:
            if path and path not in edited_files:
                edited_files.append(path)
    updates: Dict[str, Any] = {
        "edit_results": edit_results,
        "edited_files": edited_files,
    }
    diff = str(output.get("diff") or "")
    change_summary = summarize_diff_detail(
        diff,
        source="apply_code_patch",
        applied=bool(output.get("applied")),
        dry_run=bool(output.get("dry_run")),
    )
    if change_summary.get("files"):
        updates["change_summaries"] = state.get("change_summaries", []) + [change_summary]
        updates["last_change_summary"] = change_summary
    if output.get("applied"):
        updates["status"] = "patching"
        updates["verification_stale"] = True
        updates["last_edit_at_loop"] = state.get("loop_count", 0)
    return updates


def reduce_enter_plan_mode_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    events = state.get("plan_mode_events", []) + [
        {
            "tool": "EnterPlanMode",
            "entered": output.get("entered"),
            "technical_plan": output.get("technical_plan", ""),
            "risks": output.get("risks", []),
            "verification_commands": output.get("verification_commands", []),
            "assumptions": output.get("assumptions", []),
        }
    ]
    if not output.get("entered"):
        return {"plan_mode_events": events}
    return {
        "plan_mode": True,
        "plan_mode_entered": True,
        "plan_mode_approved": False,
        "debug_technical_plan": output.get("technical_plan", ""),
        "plan_mode_events": events,
        "status": "planning",
    }


def reduce_exit_plan_mode_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    approved = bool(output.get("exited")) and bool(output.get("approved"))
    events = state.get("plan_mode_events", []) + [
        {
            "tool": "ExitPlanMode",
            "exited": output.get("exited"),
            "approved": output.get("approved"),
            "evaluation": output.get("evaluation", ""),
            "remaining_uncertainties": output.get("remaining_uncertainties", []),
        }
    ]
    updates: Dict[str, Any] = {
        "plan_mode_events": events,
        "plan_mode_evaluation": output.get("evaluation", ""),
    }
    if approved:
        updates.update(
            {
                "plan_mode": False,
                "plan_mode_approved": True,
                "status": "running",
            }
        )
    else:
        updates.update(
            {
                "plan_mode": True,
                "plan_mode_approved": False,
                "status": "planning",
            }
        )
    return updates


def _extract_files_from_search(matches: list[str]) -> list[str]:
    files: list[str] = []
    for line in matches:
        if isinstance(line, dict):
            path = str(line.get("file_path") or "")
        else:
            path = str(line).split(":", 1)[0]
        if path.startswith("./"):
            path = path[2:]
        if path and path not in files:
            files.append(path)
    return files[:10]


def summarize_diff(diff: str) -> str:
    return summarize_diff_detail(diff)["summary"]


def summarize_diff_detail(
    diff: str,
    *,
    source: str = "",
    applied: bool | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    if not diff:
        return {
            "source": source,
            "applied": applied,
            "dry_run": dry_run,
            "files": [],
            "total_added": 0,
            "total_removed": 0,
            "summary": "当前没有可展示的修改。",
        }

    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    old_path = ""
    for line in diff.splitlines():
        if line.startswith("--- "):
            old_path = _clean_diff_path(line[4:].strip())
            continue
        if line.startswith("+++ "):
            new_path = _clean_diff_path(line[4:].strip())
            file_path = new_path or old_path
            current = {"file_path": file_path, "added": 0, "removed": 0}
            files.append(current)
            continue
        if current is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["removed"] += 1

    files = [item for item in files if item.get("file_path")]
    total_added = sum(int(item.get("added", 0)) for item in files)
    total_removed = sum(int(item.get("removed", 0)) for item in files)
    file_text = "，".join(
        f"{item['file_path']} +{item['added']} -{item['removed']}"
        for item in files[:5]
    )
    suffix = " 等" if len(files) > 5 else ""
    scope = "本次修改" if source == "apply_code_patch" else "当前补丁"
    summary = (
        f"{scope}包含 {len(files)} 个文件，"
        f"{total_added} 行新增、{total_removed} 行删除"
        + (f"：{file_text}{suffix}。" if file_text else "。")
    )
    return {
        "source": source,
        "applied": applied,
        "dry_run": dry_run,
        "files": files,
        "total_added": total_added,
        "total_removed": total_removed,
        "summary": summary,
    }


def _clean_diff_path(value: str) -> str:
    path = value.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return ""
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path

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

    def run(
        self,
        name: str,
        repo_path: str,
        args: Dict[str, Any] | None = None,
        *,
        allowed_permissions: list[str] | None = None,
    ) -> Dict[str, Any]:
        if name not in self._tools:
            logger.warning("unknown tool requested name={}", name)
            return normalize_tool_result({"error": f"Unknown tool: {name}"}, tool_name=name)
        logger.debug("tool registry dispatch name={} repo_path={} args={}", name, repo_path, args or {})
        return run_tool_spec(
            self._tools[name],
            repo_path,
            args or {},
            allowed_permissions=allowed_permissions,
        )

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
                input_schema=BuildCodebaseContextInput,
                permissions=["repo:read"],
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
                input_schema=SearchCodeContextInput,
                permissions=["repo:read"],
            )
        )
        # self.register(
        #     ToolSpec(
        #         name="list_files",
        #         description="List repository files with common generated directories ignored.",
        #         runner=lambda repo, args: list_files(
        #             repo,
        #             max_files=int(args.get("max_files", FileConfig.MAX_READ_AMOUNT)),
        #         ),
        #     )
        # )
        self.register(
            ToolSpec(
                name="EnterPlanMode",
                description=(
                    "Enter non-mutating planning mode and record a detailed "
                    "Debug/Refactor technical plan before code edits."
                ),
                reducer=reduce_enter_plan_mode_output,
                runner=lambda repo, args: enter_plan_mode(repo, args),
                input_schema=EnterPlanModeInput,
                permissions=["agent:plan"],
            )
        )
        self.register(
            ToolSpec(
                name="ExitPlanMode",
                description=(
                    "Exit planning mode only after the technical plan is evaluated "
                    "as feasible and uncertainties are resolved."
                ),
                reducer=reduce_exit_plan_mode_output,
                runner=lambda repo, args: exit_plan_mode(repo, args),
                input_schema=ExitPlanModeInput,
                permissions=["agent:plan"],
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
                input_schema=SearchCodeInput,
                permissions=["repo:read"],
            )
        )
        self.register(
            ToolSpec(
                name="search_text",
                description="Search repository text using regex or fixed-string patterns.",
                reducer=reduce_search_text_output,
                runner=lambda repo, args: search_text(repo, args),
                input_schema=SearchTextInput,
                permissions=["repo:read"],
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
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                ),
                input_schema=ReadFileInput,
                permissions=["repo:read"],
            )
        )
        self.register(
            ToolSpec(
                name="run_tests",
                description="Compatibility alias for run_shell_command with purpose=verification.",
                reducer=reduce_run_tests_output,
                runner=lambda repo, args: run_shell_command(
                    repo,
                    {
                        "command": str(args.get("command", "pytest")),
                        "purpose": "verification",
                        "timeout": int(args.get("timeout", 120)),
                        "reason": str(args.get("reason", "configured verification command")),
                        "allow_shell": False,
                    },
                ),
                input_schema=RunTestsInput,
                permissions=["repo:command"],
            )
        )
        self.register(
            ToolSpec(
                name="run_shell_command",
                description="Run a guarded command in the target repository.",
                reducer=reduce_run_shell_command_output,
                runner=lambda repo, args: run_shell_command(repo, args),
                input_schema=RunShellCommandInput,
                permissions=["repo:command"],
            )
        )
        self.register(
            ToolSpec(
                name="apply_code_patch",
                description=(
                    "Apply guarded exact-replacement edits to files that were "
                    "already read in this run."
                ),
                reducer=reduce_apply_code_patch_output,
                runner=lambda repo, args: apply_code_patch(repo, args),
                input_schema=ApplyCodePatchInput,
                permissions=["repo:write"],
            )
        )
        self.register(
            ToolSpec(
                name="git_diff",
                description="Return the current git diff.",
                reducer=reduce_git_diff_output,
                runner=lambda repo, args: git_diff(repo),
                permissions=["repo:read"],
            )
        )
