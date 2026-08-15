"""Task understanding stage for the agent runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import TaskAnalysisResponse
from prompts.templates import load_prompt, render_prompt


TASK_TYPES = {"BUG_FIX", "FEATURE_IMPL", "DIAGNOSE"}


class TaskAnalyzer(Protocol):
    def analyze(self, state: AgentState) -> dict[str, Any]:
        ...


@dataclass
class DisabledTaskAnalyzer:
    """No-op analyzer used when task analysis LLM is not enabled."""

    def analyze(self, state: AgentState) -> dict[str, Any]:
        return {
            "task_type": state.get("task_type", "BUG_FIX"),
            "verification_required": True,
            "verification_reason": "Task analyzer LLM is disabled; defaulting to verification.",
            "task_category": "",
            "entities": [],
            "acceptance_criteria": [],
            "risk_notes": [],
            "search_hints": [],
            "source": "disabled",
        }


@dataclass
class LLMTaskAnalyzer:
    llm_config: LLMConfig

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="task_analyzer",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/task_analyzer.md"),
            build_prompt=_task_analysis_prompt,
            fallback=None,
            response_model=TaskAnalysisResponse,
            normalize=_normalize_task_analysis,
            raise_on_error=True,
        )

    def analyze(self, state: AgentState) -> dict[str, Any]:
        return self.node.run(state)


def _task_analysis_prompt(state: AgentState, context: dict[str, Any]) -> str:
    return render_prompt(
        "user/task_analyzer.md",
        title=state.get("title", ""),
        description=state.get("description", ""),
        current_task_type=state.get("task_type", ""),
        verify_command=state.get("verify_command", ""),
        project_profile=json.dumps(state.get("project_profile", {}), ensure_ascii=False),
        registry_snapshot=json.dumps(state.get("registry_snapshot", {}), ensure_ascii=False),
    )


def _normalize_task_analysis(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    task_type = str(data.get("task_type", "")).strip().upper()
    if task_type not in TASK_TYPES:
        raise ValueError(f"invalid task_type from LLM task analyzer: {task_type}")

    return {
        "task_type": task_type,
        "verification_required": _normalize_bool(data.get("verification_required"), default=True),
        "verification_reason": str(data.get("verification_reason") or "").strip()[:240],
        "task_category": str(data.get("task_category", "")).strip()[:120],
        "entities": _clean_list(data.get("entities"), limit=12),
        "acceptance_criteria": _clean_list(data.get("acceptance_criteria"), limit=8),
        "risk_notes": _clean_list(data.get("risk_notes"), limit=8),
        "search_hints": _clean_list(data.get("search_hints"), limit=12),
    }


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _clean_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:240])
        if len(cleaned) >= limit:
            break
    return cleaned
