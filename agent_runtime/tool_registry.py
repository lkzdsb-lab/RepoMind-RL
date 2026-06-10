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
from agent_runtime.verification import infer_lightweight_verification_command
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


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


def reduce_read_file_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    if output.get("error"):
        return {}
    file_path = str(output.get("file_path") or "").strip()
    if not file_path:
        return {}
    cache = dict(state.get("read_file_cache") or {})
    order = [
        str(path).strip()
        for path in state.get("read_file_order", []) or []
        if str(path).strip()
    ]
    content = str(output.get("content") or "")
    focus_excerpt, focus_ranges = _focused_file_excerpt(state, file_path, content)
    snapshot = {
        "file_path": file_path,
        "content": content,
        "total_lines": int(output.get("total_lines") or 0),
        "truncated": bool(output.get("truncated", False)),
        "start_line": output.get("start_line"),
        "end_line": output.get("end_line"),
        "line_range_requested": bool(output.get("line_range_requested", False)),
        "is_empty": int(output.get("total_lines") or 0) == 0 and not bool(content),
        "imports_excerpt": _imports_excerpt(file_path, content),
        "focus_excerpt": focus_excerpt,
        "focus_ranges": focus_ranges,
        "full_read": (
            not bool(output.get("line_range_requested", False))
            and not bool(output.get("truncated", False))
        ),
    }
    cache[file_path] = snapshot
    if file_path in order:
        order.remove(file_path)
    order.append(file_path)
    return {
        "read_file_cache": cache,
        "read_file_order": order[-50:],
    }


def _focused_file_excerpt(
    state: Dict[str, Any],
    file_path: str,
    content: str,
    *,
    padding: int = 8,
    max_chars: int = 2400,
) -> tuple[str, list[list[int]]]:
    if not content:
        return "<empty file>", []
    lines = content.splitlines()
    if not lines:
        return "<empty file>", []
    ranges = _focus_ranges_from_code_context(state, file_path, total_lines=len(lines), padding=padding)
    if not ranges:
        excerpt = "\n".join(lines[: min(len(lines), 80)])
        return excerpt[:max_chars], []

    chunks: list[str] = []
    normalized_ranges: list[list[int]] = []
    for start, end in ranges:
        normalized_ranges.append([start, end])
        chunk_lines = lines[start - 1 : end]
        header = f"# {file_path}:{start}-{end}"
        chunks.append(header)
        chunks.append("\n".join(chunk_lines))
    excerpt = "\n".join(chunks)
    return excerpt[:max_chars], normalized_ranges


def _imports_excerpt(file_path: str, content: str, *, max_chars: int = 1200) -> str:
    if not content:
        return ""
    suffix = file_path.lower()
    lines = content.splitlines()
    collected: list[str] = []
    if suffix.endswith(".py"):
        for line in lines[:80]:
            stripped = line.strip()
            if not stripped:
                if collected:
                    collected.append(line)
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                collected.append(line)
                continue
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                if not collected:
                    collected.append(line)
                continue
            if collected:
                break
        return "\n".join(collected)[:max_chars]
    if suffix.endswith(".go"):
        in_import_block = False
        for line in lines[:120]:
            stripped = line.strip()
            if not stripped:
                if collected:
                    collected.append(line)
                continue
            if stripped.startswith("package "):
                collected.append(line)
                continue
            if stripped == "import (":
                collected.append(line)
                in_import_block = True
                continue
            if stripped.startswith("import "):
                collected.append(line)
                continue
            if in_import_block:
                collected.append(line)
                if stripped == ")":
                    in_import_block = False
                    break
                continue
            if collected:
                break
        return "\n".join(collected)[:max_chars]
    return ""


def _focus_ranges_from_code_context(
    state: Dict[str, Any],
    file_path: str,
    *,
    total_lines: int,
    padding: int,
) -> list[tuple[int, int]]:
    context = state.get("selected_code_context")
    if not isinstance(context, dict) or not context:
        context = state.get("code_context")
    if not isinstance(context, dict):
        return []

    raw_ranges: list[tuple[int, int]] = []
    for item in context.get("functions", []) or []:
        if not isinstance(item, dict) or str(item.get("file_path") or "") != file_path:
            continue
        start = int(item.get("start_line") or 0)
        end = int(item.get("end_line") or start or 0)
        if start > 0:
            raw_ranges.append((max(1, start - padding), min(total_lines, max(start, end) + padding)))
    for item in context.get("symbols", []) or []:
        if not isinstance(item, dict) or str(item.get("file_path") or "") != file_path:
            continue
        line = int(item.get("line") or 0)
        if line > 0:
            raw_ranges.append((max(1, line - padding), min(total_lines, line + padding)))
    for item in context.get("api_routes", []) or []:
        if not isinstance(item, dict) or str(item.get("file_path") or "") != file_path:
            continue
        line = int(item.get("line") or 0)
        if line > 0:
            raw_ranges.append((max(1, line - padding), min(total_lines, line + padding)))

    if not raw_ranges:
        return []
    raw_ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in raw_ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged[:6]


class RunShellCommandInput(ToolInput):
    command: str = Field(min_length=1)
    purpose: Literal["verification", "diagnostic", "search", "build"] = "diagnostic"
    timeout: int = Field(default=120, ge=1, le=1800)
    reason: str = ""
    allow_shell: bool = False


class RunTestsInput(ToolInput):
    command: str = ""
    timeout: int = Field(default=120, ge=1, le=1800)


class ApplyCodePatchChangeInput(ToolInput):
    file_path: str = Field(min_length=1)
    operation: Literal["replace", "create", "append", "insert_after", "insert_before"] = "replace"
    old_text: str | None = None
    new_text: str
    expected_occurrences: int = Field(default=1, ge=1)


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
    files = _extract_files_from_search(output.get("matches", []))
    return {"candidate_files": _prioritize_candidate_files(state, files)}


def reduce_search_text_output(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    files = list(state.get("candidate_files", []))
    for item in output.get("matches", []) or []:
        path = item.get("file_path") if isinstance(item, dict) else ""
        if path and path not in files:
            files.append(path)
    return {"candidate_files": _prioritize_candidate_files(state, files)}


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
    prioritized = _prioritize_candidate_files(state, files, preferred=context)
    return {
        "candidate_files": prioritized,
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


def _prioritize_candidate_files(
    state: Dict[str, Any],
    files: list[str],
    *,
    preferred: Dict[str, Any] | None = None,
    limit: int = 12,
) -> list[str]:
    deduped: list[str] = []
    for item in files:
        path = str(item or "").strip()
        if path and path not in deduped:
            deduped.append(path)

    preferred_paths = _preferred_code_paths(preferred or state.get("selected_code_context") or {})
    scored = sorted(
        deduped,
        key=lambda path: (
            _candidate_file_priority(path, preferred_paths),
            deduped.index(path),
        ),
    )
    return scored[:limit]


def _preferred_code_paths(context: Dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    if not isinstance(context, dict):
        return paths
    for key, field in (
        ("files", "path"),
        ("functions", "file_path"),
        ("symbols", "file_path"),
        ("api_routes", "file_path"),
        ("db_models", "file_path"),
    ):
        for item in context.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get(field) or "").strip()
            if path:
                paths.add(path)
    return paths


def _candidate_file_priority(path: str, preferred_paths: set[str]) -> tuple[int, int, str]:
    normalized = path.lower()
    suffix = normalized.rsplit("/", 1)[-1]
    is_preferred = 0 if path in preferred_paths else 1
    is_source = 0 if normalized.endswith((".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".rb")) else 1
    is_low_signal = 1 if suffix in {"config.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"} else 0
    is_test = 1 if any(token in suffix for token in ("test", "_spec", ".spec", ".snap")) else 0
    return (is_preferred, is_source, is_low_signal + is_test, normalized)


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
            updates["execution_queue"] = _advance_execution_queue_for_verification(state)
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
    updates.update(_updated_read_file_cache_from_patch(state, output))
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
        updates["execution_queue"] = _advance_execution_queue_for_patch(state, output)
    return updates


def _updated_read_file_cache_from_patch(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    updated_contents = output.get("updated_contents")
    if not isinstance(updated_contents, dict) or not updated_contents:
        return {}
    cache = dict(state.get("read_file_cache") or {})
    order = [
        str(path).strip()
        for path in state.get("read_file_order", []) or []
        if str(path).strip()
    ]
    changed = False
    for file_path, raw_content in updated_contents.items():
        path = str(file_path or "").strip()
        if not path:
            continue
        content = str(raw_content or "")
        focus_excerpt, focus_ranges = _focused_file_excerpt(state, path, content)
        total_lines = len(content.splitlines()) if content else 0
        snapshot = dict(cache.get(path) or {})
        snapshot.update(
            {
                "file_path": path,
                "content": content,
                "total_lines": total_lines,
                "truncated": False,
                "start_line": None,
                "end_line": None,
                "line_range_requested": False,
                "is_empty": total_lines == 0 and not bool(content),
                "imports_excerpt": _imports_excerpt(path, content),
                "focus_excerpt": focus_excerpt,
                "focus_ranges": focus_ranges,
                "full_read": True,
            }
        )
        cache[path] = snapshot
        if path in order:
            order.remove(path)
        order.append(path)
        changed = True
    if not changed:
        return {}
    return {
        "read_file_cache": cache,
        "read_file_order": order[-50:],
    }


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
        "plan_verification_commands": output.get("verification_commands", []),
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
                "execution_queue": _build_execution_queue(state, output),
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


def _build_execution_queue(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> list[Dict[str, Any]]:
    plan_text = "\n".join(
        part
        for part in (
            str(state.get("debug_technical_plan") or ""),
            str(output.get("evaluation") or ""),
            str(output.get("next_step") or ""),
        )
        if part.strip()
    )
    candidate_files = [
        str(path).strip()
        for path in state.get("candidate_files", []) or []
        if str(path).strip()
    ]
    ranked = sorted(
        candidate_files,
        key=lambda path: (
            plan_text.find(path) if path and path in plan_text else 10**9,
            candidate_files.index(path),
        ),
    )
    patch_targets: list[str] = []
    for path in ranked:
        lowered = path.lower()
        if lowered.endswith((".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".rb")) and path in plan_text:
            if path not in patch_targets:
                patch_targets.append(path)
    if not patch_targets:
        for path in candidate_files[:3]:
            lowered = path.lower()
            if lowered.endswith((".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".rb")):
                patch_targets.append(path)
    queue: list[Dict[str, Any]] = [
        {"kind": "patch", "target_files": [path], "status": "pending"}
        for path in patch_targets[:6]
    ]
    verification_commands = _planned_verification_commands(state)
    if bool(state.get("verification_required", True)) or verification_commands:
        queue.append(
            {
                "kind": "verify",
                "target_files": patch_targets[:],
                "commands": verification_commands,
                "status": "pending",
            }
        )
    return queue


def _planned_verification_commands(state: Dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for item in state.get("plan_verification_commands", []) or []:
        command = str(item or "").strip()
        if command and command not in commands:
            commands.append(command)
    if commands:
        return commands[:5]
    for event in reversed(state.get("plan_mode_events", []) or []):
        if not isinstance(event, dict):
            continue
        if str(event.get("tool") or "") != "EnterPlanMode":
            continue
        for item in event.get("verification_commands", []) or []:
            command = str(item or "").strip()
            if command and command not in commands:
                commands.append(command)
        if commands:
            break
    return commands[:5]


def _advance_execution_queue_for_patch(
    state: Dict[str, Any],
    output: Dict[str, Any],
) -> list[Dict[str, Any]]:
    queue = [dict(item) for item in state.get("execution_queue", []) or [] if isinstance(item, dict)]
    changed_files = {
        str(path).strip()
        for path in output.get("changed_files", []) or []
        if str(path).strip()
    }
    if not changed_files:
        return queue
    for item in queue:
        if str(item.get("status") or "pending") != "pending":
            continue
        if str(item.get("kind") or "") != "patch":
            continue
        targets = {
            str(path).strip()
            for path in item.get("target_files", []) or []
            if str(path).strip()
        }
        if targets and targets.issubset(changed_files):
            item["status"] = "completed"
            break
    return queue


def _advance_execution_queue_for_verification(
    state: Dict[str, Any],
) -> list[Dict[str, Any]]:
    queue = [dict(item) for item in state.get("execution_queue", []) or [] if isinstance(item, dict)]
    for item in queue:
        if str(item.get("status") or "pending") == "pending" and str(item.get("kind") or "") == "verify":
            item["status"] = "completed"
            break
    return queue


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
                reducer=reduce_read_file_output,
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
                        "command": str(args.get("command") or infer_lightweight_verification_command(repo)),
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
                    "Apply guarded edits to files that were already read in this run. "
                    "Supported operations: replace, create, append, insert_after, insert_before."
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
