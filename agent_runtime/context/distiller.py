"""Rule-based event distillation for prompt context.

This module deliberately avoids writing memory. It only exposes distilled facts
and memory_candidates so a future memory layer can consume them.
"""

from __future__ import annotations

from typing import Any

from model.agent.compress import ContextEvent
from model.agent.graph import AgentState
from model.agent.compress import DistilledEvent, DistillationLevel
from utils import _clean_string_list


def distill_context_events(
    events: list[ContextEvent],
    state: AgentState,
) -> tuple[list[DistilledEvent], list[dict[str, Any]]]:
    distilled = [_distill_event(event, state) for event in events]
    memory_candidates: list[dict[str, Any]] = []
    for item in distilled:
        for candidate in item.memory_candidates:
            key = (
                candidate.get("origin_kind"),
                candidate.get("content"),
                tuple(candidate.get("source_event_ids") or []),
            )
            if key not in {
                (
                    existing.get("origin_kind"),
                    existing.get("content"),
                    tuple(existing.get("source_event_ids") or []),
                )
                for existing in memory_candidates
            }:
                memory_candidates.append(candidate)
    return distilled, memory_candidates


def _distill_event(event: ContextEvent, state: AgentState) -> DistilledEvent:
    level = _level_for_event(event)
    facts: list[str] = []
    risks: list[str] = []
    open_questions: list[str] = []
    next_actions: list[str] = []
    memory_candidates: list[dict[str, Any]] = []

    if event.event_type == "task_event":
        facts.extend(_task_facts(event))
    elif event.event_type == "user_event":
        facts.append(event.summary)
    elif event.event_type == "plan_event":
        facts.extend(_plan_facts(event))
        if event.payload.get("approved") is False and event.payload.get("evaluation"):
            open_questions.append(str(event.payload.get("evaluation"))[:300])
    elif event.event_type == "search_event":
        facts.extend(_search_facts(event))
        if int(event.payload.get("candidate_count") or 0) == 0:
            next_actions.append("Try a different search term or inspect project structure.")
    elif event.event_type == "file_event":
        facts.extend(_file_facts(event))
        memory_candidates.extend(_code_fact_candidates(event, facts))
    elif event.event_type == "edit_event":
        facts.extend(_edit_facts(event))
        if event.payload.get("applied"):
            next_actions.append("Run verification for the latest edit.")
    elif event.event_type == "verification_event":
        facts.extend(_verification_facts(event))
        if event.payload.get("exit_code") not in (None, 0):
            risks.append("This command returned a nonzero exit code; inspect whether it is a reproduction or an environment failure.")
            next_actions.append(
                "Inspect the result and update the hypothesis; a reproduced bug may satisfy a diagnostic task without a fix."
                if not state.get("verification_required", True)
                else "Inspect verification stderr/stdout and update the hypothesis."
            )
            memory_candidates.append(
                _memory_candidate(
                    event,
                    "verification_failure",
                    event.summary,
                    commands=(str(event.payload.get("command") or ""),),
                )
            )
    elif event.event_type == "error_event":
        facts.append(event.summary)
        risks.append(event.summary)
        next_actions.append("Recover from the error before continuing the same path.")
        memory_candidates.append(
            _memory_candidate(
                event,
                "error_pattern",
                event.summary,
                files=(str(event.payload.get("file_path") or ""),),
                commands=(str(event.payload.get("command") or ""),),
            )
        )
    elif event.event_type == "llm_event":
        facts.append(event.summary)
        if str(event.payload.get("category") or "") in {"billing_or_quota", "auth_or_access"}:
            risks.append("LLM provider error may block LLM-dependent steps.")
    elif event.event_type == "progress_event":
        facts.append(event.summary)
        facts.extend(_clean_list(event.payload.get("new_findings"), 5, 260))
        open_questions.extend(_clean_list(event.payload.get("missing_context"), 5, 260))
        memory_candidates.extend(_observation_memory_candidates(event))
    else:
        facts.append(event.summary)

    if not facts:
        facts.append(event.summary)

    return DistilledEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        source=event.source,
        level=level,
        importance=event.importance,
        retention=event.retention,
        summary=event.summary,
        facts=_clean_string_list(facts, 10, 360),
        risks=_clean_string_list(risks, 6, 360),
        open_questions=_clean_string_list(open_questions, 6, 360),
        next_actions=_clean_string_list(next_actions, 6, 360),
        memory_candidates=memory_candidates[:6],
        raw_ref=event.raw_ref,
    )


def _level_for_event(event: ContextEvent) -> DistillationLevel:
    if event.importance == "critical" or event.retention == "working":
        return "pinned"
    if event.retention == "archive" or event.importance == "low":
        return "archive"
    return "semantic"


def _task_facts(event: ContextEvent) -> list[str]:
    facts = [event.summary]
    if "verification_required" in event.payload:
        facts.append(f"verification_required={bool(event.payload.get('verification_required'))}")
    reason = str(event.payload.get("verification_reason") or "").strip()
    if reason:
        facts.append(f"verification_reason={reason[:260]}")
    return facts


def _plan_facts(event: ContextEvent) -> list[str]:
    facts = [event.summary]
    if "approved" in event.payload:
        facts.append(f"plan_approved={bool(event.payload.get('approved'))}")
    evaluation = str(event.payload.get("evaluation") or "").strip()
    if evaluation:
        facts.append(f"plan_evaluation={evaluation[:360]}")
    return facts


def _search_facts(event: ContextEvent) -> list[str]:
    count = int(event.payload.get("candidate_count") or 0)
    facts = [f"{event.source} found {count} candidates."]
    files = _clean_list(event.payload.get("files"), 10, 220)
    if files:
        facts.append(f"candidate_files={files}")
    query = str(event.payload.get("query") or event.payload.get("pattern") or "").strip()
    if query:
        facts.append(f"search_query={query[:260]}")
    return facts


def _file_facts(event: ContextEvent) -> list[str]:
    path = str(event.payload.get("file_path") or "").strip()
    facts = [f"Read file {path}." if path else event.summary]
    facts.append(
        f"revision={event.payload.get('file_revision', '')}; "
        f"returned_lines={event.payload.get('start_line')}-{event.payload.get('end_line')}; "
        f"source_truncated={bool(event.payload.get('source_truncated'))}; "
        f"excerpt_truncated={bool(event.payload.get('excerpt_truncated'))}. "
        "This is a read observation, not an analysis-completion record."
    )
    excerpt = str(event.payload.get("content_excerpt") or "").strip()
    if excerpt:
        facts.append(f"relevant_excerpt={excerpt[:700]}")
    return facts


def _edit_facts(event: ContextEvent) -> list[str]:
    changed = _clean_list(event.payload.get("changed_files"), 8, 220)
    return [
        f"edit_applied={bool(event.payload.get('applied'))}",
        f"changed_files={changed}" if changed else event.summary,
        f"changed_line_count={event.payload.get('changed_line_count', 0)}",
    ]


def _verification_facts(event: ContextEvent) -> list[str]:
    command = str(event.payload.get("command") or "").strip()
    exit_code = event.payload.get("exit_code")
    facts = [f"verification command exit_code={exit_code}."]
    if command:
        facts.append(f"command={command[:260]}")
    stderr = str(event.payload.get("stderr_excerpt") or "").strip()
    stdout = str(event.payload.get("stdout_excerpt") or "").strip()
    if exit_code not in (None, 0):
        if stderr:
            facts.append(f"stderr={stderr[:700]}")
        elif stdout:
            facts.append(f"stdout={stdout[:700]}")
    return facts


def _code_fact_candidates(event: ContextEvent, facts: list[str]) -> list[dict[str, Any]]:
    path = str(event.payload.get("file_path") or "").strip()
    if not path:
        return []
    content = " ".join(facts)[:900]
    if not content:
        return []
    return [
        _memory_candidate(
            event,
            "code_observation",
            f"{path}: {content}",
            files=(path,),
        )
    ]


def _memory_candidate(
    event: ContextEvent,
    origin_kind: str,
    content: str,
    *,
    files: tuple[str, ...] = (),
    symbols: tuple[str, ...] = (),
    commands: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "origin_kind": origin_kind,
        "source_event_ids": [event.event_id],
        "evidence_refs": [],
        "files": [value for value in files if value],
        "symbols": [value for value in symbols if value],
        "commands": [value for value in commands if value],
        "importance": event.importance,
        "content": str(content or "").strip()[:1000],
        "raw_ref": event.raw_ref,
    }


def _observation_memory_candidates(event: ContextEvent) -> list[dict[str, Any]]:
    value = event.payload.get("memory_candidates")
    if not isinstance(value, list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        candidates.append(
            {
                "schema_version": 2,
                "origin_kind": "observer_observation",
                "source_event_ids": [event.event_id],
                "evidence_refs": [],
                "files": [],
                "symbols": [],
                "commands": [],
                "importance": event.importance,
                "content": content[:1000],
                "raw_ref": event.raw_ref,
            }
        )
        if len(candidates) >= 6:
            break
    return candidates


def _clean_list(value: Any, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text[:max_chars])
        if len(result) >= limit:
            break
    return result
