"""Attention focus derivation for LLM context shaping."""

from __future__ import annotations

from typing import Any

from agent_runtime.execution_queue import current_execution_item
from model.agent.graph import AgentState
from utils import _append_unique, _as_dict, _clean_string_list, _latest_dict, _positive_int, _truncate


MAX_FOCUS_FILES = 8
MAX_FOCUS_SYMBOLS = 12
MAX_FOCUS_EVIDENCE = 8
MAX_TEXT = 240


def build_attention_focus(state: AgentState) -> dict[str, Any]:
    """Build a bounded, serializable focus view for the next LLM decision.

    This layer is intentionally advisory. It does not decide legal actions,
    completion, queue reconciliation, or runtime phase transitions.
    """
    phase = str(state.get("phase") or "collect_context")
    runtime_decision = _as_dict(state.get("runtime_decision"))
    pending_resolution = _as_dict(state.get("pending_resolution"))
    execution = _as_dict(current_execution_item(state))

    focus_files: list[str] = []
    reasons: list[str] = []

    pending_kind = str(pending_resolution.get("kind") or "").strip()
    pending_target = str(pending_resolution.get("target_file") or "").strip()
    if pending_kind:
        reasons.append(f"pending_resolution:{pending_kind}")
    if pending_target:
        _append_unique(focus_files, [pending_target])

    execution_kind = str(execution.get("kind") or "").strip()
    execution_files = _clean_string_list(execution.get("target_files"), -1, None)
    if execution_kind:
        reasons.append(f"execution:{execution_kind}")
    _append_unique(focus_files, execution_files)

    if bool(state.get("verification_stale", False)) or phase == "verify":
        reasons.append("verification_stale" if state.get("verification_stale") else "phase:verify")
        _append_unique(focus_files, _clean_string_list(state.get("edited_files"), -1, None))

    context_files = _selected_context_files(state)
    _append_unique(focus_files, context_files)
    _append_unique(focus_files, _clean_string_list(state.get("candidate_files"), -1, None))
    focus_files = focus_files[:MAX_FOCUS_FILES]

    focus = {
        "phase": phase, # 当前处在哪个运行阶段
        "primary_goal": _primary_goal(state, phase, execution_kind, pending_kind), # 拼核心问题
        "focus_files": focus_files,
        "focus_ranges": _focus_ranges(state, focus_files),
        "focus_symbols": _focus_symbols(state, focus_files),
        "focus_actions": _focus_actions(state, phase, execution_kind, pending_resolution, runtime_decision),
        "focus_evidence": _focus_evidence(state, runtime_decision, pending_resolution),
        "suppressed_context": _suppressed_context(phase, focus_files),
        "reason": "; ".join(reasons) or "derived from current task context",
    }
    return focus


def _primary_goal(
    state: AgentState,
    phase: str,
    execution_kind: str,
    pending_kind: str,
) -> str:
    title = _truncate(str(state.get("title") or "").strip(), 120)
    qualifiers = [part for part in (phase, execution_kind, pending_kind) if part]
    prefix = " / ".join(qualifiers) if qualifiers else "current task"
    if title:
        return f"{prefix}: {title}"
    return prefix


def _focus_actions(
    state: AgentState,
    phase: str,
    execution_kind: str,
    pending_resolution: dict[str, Any],
    runtime_decision: dict[str, Any],
) -> list[str]:
    """ 这轮更可能合理的 action 候选，比如 recovery 时偏向 read_file，verify 时偏向
  run_shell_command，patch 时偏向 read_file/apply_code_patch。作用是辅助 LLM 在 legal_actions 里排
  序，但不替代 legal_actions。"""
    actions: list[str] = []
    required = str(pending_resolution.get("required_next_action") or "").strip()
    if required:
        _append_unique(actions, [required])
    runtime_required = str(runtime_decision.get("required_next_action") or "").strip()
    if runtime_required:
        _append_unique(actions, [runtime_required])
    if pending_resolution:
        _append_unique(actions, ["read_file"])
    if bool(state.get("verification_stale", False)) or phase == "verify" or execution_kind == "verify":
        _append_unique(actions, ["run_shell_command", "run_tests"])
    if execution_kind == "patch":
        _append_unique(actions, ["read_file", "apply_code_patch"])
    if phase == "plan":
        if state.get("debug_technical_plan"):
            _append_unique(actions, ["ExitPlanMode"])
        else:
            _append_unique(actions, ["EnterPlanMode"])
    if not actions:
        _append_unique(actions, ["search_code_context", "read_file"])
    return actions[:6]


def _focus_evidence(
    state: AgentState,
    runtime_decision: dict[str, Any],
    pending_resolution: dict[str, Any],
) -> list[str]:
    """ 当前最相关的证据摘要，比如 blockers、pending_resolution、最近 edit/test/command 结果、completion_judgement。
    作用是把“为什么现在关注这些”压成短证据，减少 LLM 回翻长历史"""
    evidence: list[str] = []
    for blocker in runtime_decision.get("blockers") or []:
        if isinstance(blocker, dict):
            kind = str(blocker.get("kind") or "").strip()
            message = str(blocker.get("message") or blocker.get("reason") or "").strip()
            _append_unique(evidence, [_truncate(f"blocker:{kind}:{message}", MAX_TEXT)])
        else:
            _append_unique(evidence, [_truncate(f"blocker:{blocker}", MAX_TEXT)])
    if pending_resolution:
        _append_unique(evidence, [_truncate(f"pending_resolution:{pending_resolution}", MAX_TEXT)])
    for key, label in (
        ("edit_results", "latest_edit"),
        ("test_results", "latest_test"),
        ("command_results", "latest_command"),
    ):
        latest = _latest_dict(state.get(key))
        if latest:
            _append_unique(evidence, [_truncate(f"{label}:{latest}", MAX_TEXT)])
    judgement = _as_dict(state.get("completion_judgement"))
    if judgement:
        _append_unique(evidence, [_truncate(f"completion_judgement:{judgement}", MAX_TEXT)])
    return evidence[:MAX_FOCUS_EVIDENCE]


def _suppressed_context(phase: str, focus_files: list[str]) -> list[str]:
    """ 建议弱化的上下文类别"""
    suppressed: list[str] = []
    if focus_files:
        suppressed.append("non_focus_read_files")
        suppressed.append("older_low_importance_observations")
    if phase in {"execute_patch", "patch", "verify", "recover"}:
        suppressed.append("broad_repository_context")
    return suppressed


def _selected_context_files(state: AgentState) -> list[str]:
    files: list[str] = []
    for context in (state.get("selected_code_context"), state.get("code_context")):
        if not isinstance(context, dict):
            continue
        for key, field in (
            ("files", "path"),
            ("functions", "file_path"),
            ("symbols", "file_path"),
            ("api_routes", "file_path"),
            ("db_models", "file_path"),
            ("call_graph", "file_path"),
        ):
            for item in context.get(key, []) or []:
                if isinstance(item, dict):
                    _append_unique(files, [str(item.get(field) or "").strip()])
    return files


def _focus_symbols(state: AgentState, focus_files: list[str]) -> list[str]:
    """ 重点函数、类、路由、模型等符号。作用是比文件更细一层，告诉 LLM 当前逻辑可能落在哪些函数/对象上。"""
    focus_set = set(focus_files)
    symbols: list[str] = []
    for context in (state.get("selected_code_context"), state.get("code_context")):
        if not isinstance(context, dict):
            continue
        for key in ("functions", "symbols", "api_routes", "db_models"):
            for item in context.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("file_path") or "").strip()
                if focus_set and path not in focus_set:
                    continue
                name = (
                    item.get("full_name")
                    or item.get("name")
                    or item.get("route")
                    or item.get("path")
                )
                text = str(name or "").strip()
                if text:
                    _append_unique(symbols, [text])
    return symbols[:MAX_FOCUS_SYMBOLS]


def _focus_ranges(state: AgentState, focus_files: list[str]) -> dict[str, list[dict[str, int]]]:
    """ 文件内的重点行号范围。主要来自 selected code context 或 read cache 里的 focus ranges。作用是让 LLM 不只知道文件，还知道文件里的局部位置。"""
    ranges: dict[str, list[dict[str, int]]] = {path: [] for path in focus_files}
    focus_set = set(focus_files)
    cache = state.get("read_file_cache")
    if isinstance(cache, dict):
        for path in focus_files:
            snapshot = cache.get(path)
            if not isinstance(snapshot, dict):
                continue
            for item in snapshot.get("focus_ranges") or []:
                normalized = _normalize_range(item)
                if normalized:
                    ranges[path].append(normalized)
    for context in (state.get("selected_code_context"), state.get("code_context")):
        if not isinstance(context, dict):
            continue
        for key in ("functions", "symbols", "api_routes", "db_models", "call_graph"):
            for item in context.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("file_path") or "").strip()
                if path not in focus_set:
                    continue
                normalized = _normalize_range(item)
                if normalized:
                    ranges.setdefault(path, []).append(normalized)
    return {path: _dedupe_ranges(items)[:6] for path, items in ranges.items() if items}


def _normalize_range(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    start = _positive_int(value.get("start_line") or value.get("line") or value.get("start"))
    end = _positive_int(value.get("end_line") or value.get("line") or value.get("end") or start)
    if start is None:
        return None
    if end is None or end < start:
        end = start
    return {"start_line": start, "end_line": end}


def _dedupe_ranges(items: list[dict[str, int]]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in items:
        key = (int(item["start_line"]), int(item["end_line"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
