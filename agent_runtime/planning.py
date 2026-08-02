"""Planning strategies for the debug agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.verification.capabilities import recommended_verification_command
from config import LLMConfig
from ext.tool_summaries import validated_cache_summary
from model.agent.graph import AgentState
from model.llm import PlanResponse
from prompts.templates import load_prompt, render_prompt


class Planner(Protocol):
    def make_plan(self, state: AgentState) -> list[str]:
        ...


@dataclass
class HeuristicPlanner:
    def make_plan(self, state: AgentState) -> list[str]:
        verification_example = _verification_example(state)
        verification_step = (
            "Skip command verification because the current read-only task does not require it."
            if not _verification_required(state)
            else f"Run an allowed verification command, for example: {verification_example}"
        )
        return [
            "Analyze the task and extract code search keywords.",
            "Search structured code context for candidate files.",
            "Read candidate files to build exact context.",
            verification_step,
            "Inspect git diff and summarize the current patch state.",
        ]


@dataclass
class LLMPlanner:
    llm_config: LLMConfig
    fallback: Planner | None = None

    def __post_init__(self) -> None:
        self.fallback = self.fallback or HeuristicPlanner()
        self.node = LLMJsonNode(
            name="planner",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/planner.md"),
            build_prompt=_planner_node_prompt,
            fallback=lambda state, context: {
                "plan": self.fallback.make_plan(state) if self.fallback else []
            },
            response_model=PlanResponse,
            normalize=_normalize_plan_response,
        )

    def make_plan(self, state: AgentState) -> list[str]:
        fallback_plan = self.fallback.make_plan(state) if self.fallback else []
        data = self.node.run(state, {"fallback_plan": fallback_plan})
        plan = data.get("plan")
        if isinstance(plan, list) and plan:
            return [str(item).strip() for item in plan if str(item).strip()][:8]
        return fallback_plan


def _planner_node_prompt(state: AgentState, context: dict) -> str:
    fallback_plan = context.get("fallback_plan") or HeuristicPlanner().make_plan(state)
    return render_prompt(
        "user/planner.md",
        title=state.get("title", ""),
        description=state.get("description", ""),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False),
        current_step=state.get("current_step", ""),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
        validated_file_cache=json.dumps(
            validated_cache_summary(state),
            ensure_ascii=False,
            default=str,
        ),
        memory_context=str(state.get("memory_context", ""))[:3000],
        compressed_context=str(state.get("compressed_context", ""))[:3000],
        verification_required=json.dumps(_verification_required(state)),
        verification_reason=state.get("verification_reason", ""),
        verification_capabilities=json.dumps(
            state.get("verification_capabilities", {}), ensure_ascii=False
        ),
        default_plan=json.dumps(fallback_plan, ensure_ascii=False),
    )


def _normalize_plan_response(data: dict, state: AgentState, context: dict) -> dict:
    fallback_plan = context.get("fallback_plan") or HeuristicPlanner().make_plan(state)
    raw_plan = data.get("plan")
    if not isinstance(raw_plan, list):
        raw_plan = fallback_plan
    plan = [str(item).strip() for item in raw_plan if str(item).strip()]
    if not plan:
        plan = fallback_plan
    return {"plan": plan[:8]}


def _verification_required(state: AgentState) -> bool:
    return bool(state.get("verification_required", True))


def _verification_example(state: AgentState) -> str:
    return recommended_verification_command(state)
