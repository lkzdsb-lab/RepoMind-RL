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


class TaskAnalyzer(Protocol):
    def analyze(self, state: AgentState) -> dict[str, Any]:
        ...


@dataclass
class DisabledTaskAnalyzer:
    """No-op analyzer used when task analysis LLM is not enabled."""

    def analyze(self, state: AgentState) -> dict[str, Any]:
        return {
            "intent": "diagnose",
            "task_type": state.get("task_type", "BUG_FIX"),
            "task_category": "",
            "entities": [],
            "acceptance_criteria": [],
            "risk_notes": [],
            "review_focus": [],
            "search_hints": [],
            "historical_context": [],
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
            normalize=None,
            raise_on_error=True,
        )

    def analyze(self, state: AgentState) -> dict[str, Any]:
        return self.node.run(state)


def _task_analysis_prompt(state: AgentState, context: dict[str, Any]) -> str:
    session_memory = state.get("session_memory", {})
    if isinstance(session_memory, dict):
        session_memory = {
            key: value for key, value in session_memory.items() if key != "rendered"
        }
    return render_prompt(
        "user/task_analyzer.md",
        title=state.get("title", ""),
        description=state.get("description", ""),
        current_task_type=state.get("task_type", ""),
        project_profile=json.dumps(state.get("project_profile", {}), ensure_ascii=False),
        registry_snapshot=json.dumps(state.get("registry_snapshot", {}), ensure_ascii=False),
        session_memory=json.dumps(session_memory, ensure_ascii=False),
    )
