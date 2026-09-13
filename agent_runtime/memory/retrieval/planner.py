"""Deterministic query planning from current task state."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.memory.retrieval.models import PlannedMemoryQuery
from agent_runtime.memory.retrieval.scope import ScopeContext
from model.agent.graph import AgentState


class RuleBasedMemoryQueryPlanner:
    def __init__(self, max_queries: int = 4, max_query_chars: int = 2000) -> None:
        self.max_queries = max(1, min(8, int(max_queries)))
        self.max_query_chars = max(200, int(max_query_chars))

    def plan(self, state: AgentState, scope_context: ScopeContext | None = None) -> list[PlannedMemoryQuery]:
        """ 查询规划器"""
        analysis = state.get("task_analysis")
        if not isinstance(analysis, dict):
            analysis = {}
        entities = _strings(analysis.get("entities"), 20)
        search_hints = _strings(analysis.get("search_hints"), 20)
        candidate_files = _strings(state.get("candidate_files"), 30)
        main_text = _join(
            state.get("title"),
            state.get("description"),
            entities,
            search_hints,
        )
        if not main_text:
            return []
        queries = [PlannedMemoryQuery("task", self._bounded(main_text))]
        queries.append(
            PlannedMemoryQuery(
                "preference",
                self._bounded(main_text),
                memory_types=("preference",),
            )
        )
        error_text = _runtime_error_text(state)
        if error_text:
            queries.append(
                PlannedMemoryQuery(
                    "failure",
                    self._bounded(_join(main_text, error_text)),
                    memory_types=("anti_pattern", "semantic", "episodic"),
                )
            )
        scope_terms = scope_context.query_terms() if scope_context else ()
        if scope_terms or candidate_files or entities or search_hints:
            queries.append(
                PlannedMemoryQuery(
                    "scope",
                    self._bounded(_join(scope_terms, entities, search_hints, candidate_files, main_text)),
                    memory_types=("semantic", "procedural", "anti_pattern"),
                )
            )
        return _dedupe(queries)[: self.max_queries]

    def fingerprint_payload(self, state: AgentState) -> dict[str, Any]:
        analysis = state.get("task_analysis")
        if not isinstance(analysis, dict):
            analysis = {}
        return {
            "title": str(state.get("title") or ""),
            "description": str(state.get("description") or ""),
            "task_type": str(state.get("task_type") or ""),
            "task_category": str(state.get("task_category") or ""),
            "entities": _strings(analysis.get("entities"), 20),
            "search_hints": _strings(analysis.get("search_hints"), 20),
            "candidate_files": _strings(state.get("candidate_files"), 30),
            "errors": _runtime_error_text(state),
            "user_inputs": [
                {
                    "answer": str(item.get("answer") or "")[:500],
                    "questions": _strings(item.get("questions"), 5),
                }
                for item in (state.get("user_inputs") or [])[-2:]
                if isinstance(item, dict)
            ],
        }

    def _bounded(self, value: str) -> str:
        text = " ".join(str(value or "").split())
        return text[: self.max_query_chars]


def _runtime_error_text(state: AgentState) -> str:
    values: list[str] = []
    error = str(state.get("error") or "").strip()
    if error:
        values.append(error)
    for group_name in ("test_results", "command_results"):
        group = state.get(group_name)
        if not isinstance(group, list):
            continue
        for item in group[-2:]:
            if not isinstance(item, dict):
                continue
            exit_code = item.get("exit_code")
            if exit_code in (None, 0):
                continue
            values.extend(
                str(item.get(key) or "").strip()
                for key in ("command", "stderr", "stdout", "error")
                if str(item.get(key) or "").strip()
            )
    return " ".join(values)[:3000]


def _join(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            parts.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
            if text:
                parts.append(text)
    return " ".join(dict.fromkeys(parts))


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(str(item).strip()[:500] for item in value if str(item).strip())
    )[:limit]


def _dedupe(values: list[PlannedMemoryQuery]) -> list[PlannedMemoryQuery]:
    result: list[PlannedMemoryQuery] = []
    seen: set[str] = set()
    for item in values:
        identity = json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
        if identity not in seen:
            result.append(item)
            seen.add(identity)
    return result
