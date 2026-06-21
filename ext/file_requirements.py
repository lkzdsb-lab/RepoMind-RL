"""Dynamic full-read requirements for candidate files."""

from __future__ import annotations

import re
from typing import Any

from agent_runtime.execution_queue import current_execution_item
from model.agent.graph import AgentState

FULL_READ_MAX_CHARS = 200000
DEFAULT_READ_MAX_CHARS = 8000
RELEVANCE_THRESHOLD = 2.0


def collect_active_queries(state: AgentState) -> list[str]:
    """ 从 state 获取所有的 query 条件"""
    # 1.从用户的自然语言中获取
    queries: list[str] = []
    _append_query(queries, state.get("title"))
    _append_query(queries, state.get("description"))
    _append_query(queries, state.get("current_step"))

    # 2.从 llm 的决策中获取
    pending = (state.get("pending_resolution") or {}).get("details")
    if isinstance(pending, dict):
        _append_query(queries, pending.get("reason"))
        _append_query(queries, pending.get("message"))
        partial_args = pending.get("partial_args")
        if isinstance(partial_args, dict):
            for key in ("pattern", "query", "file_path", "command"):
                _append_query(queries, partial_args.get(key))

    for plan_like in (
        state.get("debug_technical_plan"),
        state.get("plan_mode_evaluation"),
    ):
        _append_query(queries, plan_like)

    # 3.从 代码索引 中获取
    for query_plan_key in ("code_context_query_plan",):
        query_plan = state.get(query_plan_key)
        if isinstance(query_plan, dict):
            _append_query(queries, query_plan.get("query"))
            for item in query_plan.get("queries") or []:
                _append_query(queries, item)
            for item in query_plan.get("terms") or []:
                _append_query(queries, item)

    # 4.从工具返回中获取
    for call in reversed(state.get("tool_calls", [])[-6:]):
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        tool_input = call.get("input")
        tool_output = call.get("output")
        if not isinstance(tool_input, dict):
            tool_input = {}
        if not isinstance(tool_output, dict):
            tool_output = {}
        if name == "search_text":
            _append_query(queries, tool_input.get("pattern"))
        elif name == "search_code_context":
            _append_query(queries, tool_input.get("query"))
            _append_query(queries, tool_output.get("query"))
            for item in tool_output.get("queries") or []:
                _append_query(queries, item)

    return queries[:12]


def full_read_requirements(
    state: AgentState,
    *,
    candidate_files: list[str] | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """ 获取排名靠前的需要 全量读取的文件"""
    files = candidate_files or [
        str(path).strip()
        for path in state.get("candidate_files", []) or []
        if str(path).strip()
    ]
    requirements: list[dict[str, Any]] = []
    for path in files:
        reason = full_read_reason(state, path)
        if not reason:
            continue
        requirements.append(reason)
    requirements.sort(
        key=lambda item: (
            -float(item.get("relevance", 0.0)),
            0 if str(item.get("sufficiency") or "") == "none" else 1,
            str(item.get("file_path") or ""),
        )
    )
    return requirements[:limit]


def full_read_reason(state: AgentState, file_path: str) -> dict[str, Any] | None:
    """ 填充需要读取整个文件的理由"""
    path = str(file_path or "").strip()
    if not path:
        return None
    relevance = file_relevance_score(state, path)
    sufficiency = summary_sufficiency(state, path)
    if not _needs_full_read(state, path, relevance, sufficiency):
        return None
    reason_parts: list[str] = []
    if _is_execution_target(state, path):
        reason_parts.append("current execution target")
    if _is_selected_context_path(state, path):
        reason_parts.append("selected code context hit")
    if relevance >= RELEVANCE_THRESHOLD:
        reason_parts.append(f"query relevance {relevance:.2f}")
    if sufficiency == "none":
        reason_parts.append("no usable summary yet")
    elif sufficiency == "partial":
        reason_parts.append("only partial summary/excerpt available")
    return {
        "file_path": path,
        "relevance": round(relevance, 3),
        "sufficiency": sufficiency,
        "reason": ", ".join(reason_parts) or "full read required",
    }


def choose_read_file_target(
    state: AgentState,
    *,
    requested_path: str = "",
    default_path: str = "",
) -> str:
    requested_path = str(requested_path or "").strip()
    default_path = str(default_path or "").strip()
    required = full_read_requirements(state, limit=12)
    required_paths = [str(item.get("file_path") or "").strip() for item in required if str(item.get("file_path") or "").strip()]
    for candidate in (requested_path, default_path):
        if candidate and candidate in required_paths and not is_full_read(state, candidate):
            return candidate
    for candidate in required_paths:
        if candidate and not is_full_read(state, candidate):
            return candidate
    return requested_path or default_path


def recommended_read_file_args(
    state: AgentState,
    file_path: str,
    *,
    requested_max_chars: Any = None,
    default_max_chars: int = DEFAULT_READ_MAX_CHARS,
) -> dict[str, Any]:
    """ 获取对这个文件的读取的推荐参数"""
    path = str(file_path or "").strip()
    max_chars = _coerce_max_chars(requested_max_chars, default_max_chars)
    reason = full_read_reason(state, path)
    if reason is not None:
        max_chars = max(max_chars, FULL_READ_MAX_CHARS)
    return {
        "file_path": path,
        "max_chars": max_chars,
        "full_read_expected": reason is not None,
    }


def is_full_read(state: AgentState, file_path: str) -> bool:
    """ 过滤已经全读过的文件"""
    cache = state.get("read_file_cache")
    if not isinstance(cache, dict):
        return False
    snapshot = cache.get(file_path)
    if not isinstance(snapshot, dict):
        return False
    return bool(snapshot.get("full_read", False))


def summary_sufficiency(state: AgentState, file_path: str) -> str:
    """ 现有摘要是否足够支持“诊断/修改/验证”决策，不充足才读整个文件"""
    cache = state.get("read_file_cache")
    if isinstance(cache, dict):
        snapshot = cache.get(file_path)
        if isinstance(snapshot, dict):
            if bool(snapshot.get("full_read", False)):
                return "sufficient"
            if snapshot.get("focus_excerpt") or snapshot.get("imports_excerpt"):
                return "partial"
            return "none"
    if _is_selected_context_path(state, file_path):
        return "partial"
    return "none"


def file_relevance_score(state: AgentState, file_path: str) -> float:
    """ 这个文件和当前 query 有多相关"""
    path = str(file_path or "").strip()
    if not path:
        return 0.0
    score = 0.0
    if _is_execution_target(state, path):
        score += 3.0
    if _is_selected_context_path(state, path):
        score += 1.8
    basename = path.rsplit("/", 1)[-1].lower()
    path_terms = set(_extract_terms(path))
    for query in collect_active_queries(state):
        query_terms = _extract_terms(query)
        if not query_terms:
            continue
        if basename in query.lower():
            score += 2.8
        overlap = len(path_terms.intersection(query_terms))
        score += min(2.0, overlap * 0.5)
    return score


def _needs_full_read(
    state: AgentState,
    file_path: str,
    relevance: float,
    sufficiency: str,
) -> bool:
    if is_full_read(state, file_path):
        return False
    if _is_execution_target(state, file_path):
        return True
    if sufficiency == "none":
        return relevance > 0.0
    return relevance >= RELEVANCE_THRESHOLD and sufficiency != "sufficient"


def _is_execution_target(state: AgentState, file_path: str) -> bool:
    item = current_execution_item(state)
    if isinstance(item, dict):
        for path in item.get("target_files", []) or []:
            if str(path or "").strip() == file_path:
                return True
    return False


def _is_selected_context_path(state: AgentState, file_path: str) -> bool:
    for context in (state.get("selected_code_context"), state.get("code_context")):
        if not isinstance(context, dict):
            continue
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
                if path == file_path:
                    return True
    return False


def _append_query(queries: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in queries:
        queries.append(text[:1000])


def _extract_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]+", str(text or "")):
        token = raw.strip(".,:;()[]{}<>`'\"").lower()
        if not token:
            continue
        for part in re.split(r"[_\-.:/]+", token):
            part = part.strip()
            if len(part) >= 2:
                terms.add(part)
        if len(token) >= 2:
            terms.add(token)
    return terms


def _coerce_max_chars(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(FULL_READ_MAX_CHARS, parsed))
