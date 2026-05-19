"""Planning strategies for the debug agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import PlanResponse


class Planner(Protocol):
    def make_plan(self, state: AgentState) -> list[str]:
        ...


@dataclass
class HeuristicPlanner:
    def make_plan(self, state: AgentState) -> list[str]:
        verify_command = state.get("verify_command") or "pytest"
        return [
            "解析 issue，提取代码搜索关键词",
            "读取仓库结构并搜索相关代码",
            "阅读候选文件建立上下文",
            f"运行验证命令：{verify_command}",
            "查看 git diff 并汇总当前补丁状态",
            "按 reward gate 写入分层记忆，必要时沉淀到 skill",
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
            system_prompt=(
                "You are planning the next debugging workflow for an agent. "
                "Return only JSON matching the requested schema. "
                "Do not output prose outside JSON."
            ),
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
    return (
        f"title={state.get('title', '')}\n"
        f"description={state.get('description', '')}\n"
        f"task_analysis={json.dumps(state.get('task_analysis', {}), ensure_ascii=False)}\n"
        f"current_step={state.get('current_step', '')}\n"
        f"candidate_files={state.get('candidate_files', [])}\n"
        f"memory_context={state.get('memory_context', '')[:3000]}\n"
        f"compressed_context={state.get('compressed_context', '')[:3000]}\n"
        f"verify_command={state.get('verify_command', '')}\n"
        f"default_plan={json.dumps(fallback_plan, ensure_ascii=False)}\n"
        "Return JSON like {\"plan\": [\"...\", \"...\"]}."
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
