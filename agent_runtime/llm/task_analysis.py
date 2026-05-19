"""Task understanding stage for the agent runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import TaskAnalysisResponse


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
            system_prompt=(
                "You classify and structure a debugging or implementation task for an agent. "
                "Return only JSON matching the requested schema. "
                "Do not invent repository facts."
            ),
            build_prompt=_task_analysis_prompt,
            fallback=None,
            response_model=TaskAnalysisResponse,
            normalize=_normalize_task_analysis,
            raise_on_error=True,
        )

    def analyze(self, state: AgentState) -> dict[str, Any]:
        return self.node.run(state)


def _task_analysis_prompt(state: AgentState, context: dict[str, Any]) -> str:
    return (
        "Return JSON with keys: task_type, task_category, entities, acceptance_criteria, "
        "risk_notes, search_hints.\n"
        "task_type must be one of BUG_FIX, FEATURE_IMPL, DIAGNOSE.\n"
        "entities and search_hints should come from the user's actual task wording only. "
        "If a field is uncertain, use an empty list/string instead of guessing.\n\n"
        f"title={state.get('title', '')}\n"
        f"description={state.get('description', '')}\n"
        f"current_task_type={state.get('task_type', '')}\n"
        f"verify_command={state.get('verify_command', '')}\n"
        f"registry_snapshot={json.dumps(state.get('registry_snapshot', {}), ensure_ascii=False)}\n"
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
        "task_category": str(data.get("task_category", "")).strip()[:120],
        "entities": _clean_list(data.get("entities"), limit=12),
        "acceptance_criteria": _clean_list(data.get("acceptance_criteria"), limit=8),
        "risk_notes": _clean_list(data.get("risk_notes"), limit=8),
        "search_hints": _clean_list(data.get("search_hints"), limit=12),
    }


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
