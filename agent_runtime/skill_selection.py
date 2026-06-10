"""LLM-assisted skill selection for an agent run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import SkillSelectorResponse
from model.skill import SkillSpec
from prompts.templates import load_prompt, render_prompt
from utils import _truncate_text, _clamp_float


@dataclass
class SkillSelection:
    selected_skills: list[str] = field(default_factory=list)
    source: str = "disabled"
    rationale: str = ""
    selections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillSelector(Protocol):
    def select(
        self,
        state: AgentState,
        skills: Mapping[str, SkillSpec],
    ) -> SkillSelection:
        ...


@dataclass
class DisabledSkillSelector:
    def select(
        self,
        state: AgentState,
        skills: Mapping[str, SkillSpec],
    ) -> SkillSelection:
        selected = _dedupe_names(state.get("selected_skills", []), skills)
        return SkillSelection(
            selected_skills=selected,
            source="disabled",
            rationale="LLM skill selector is disabled.",
        )


@dataclass
class LLMSkillSelector:
    llm_config: LLMConfig
    selected_limit: int = 5

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="skill_selector",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/skill_selector.md"),
            build_prompt=_skill_selector_prompt,
            fallback=None,
            response_model=SkillSelectorResponse,
            normalize=self._normalize,
            raise_on_error=True,
        )

    def select(
        self,
        state: AgentState,
        skills: Mapping[str, SkillSpec],
    ) -> SkillSelection:
        if not skills:
            return SkillSelection(source="llm", rationale="No registered skills.")
        data = self.node.run(
            state,
            {
                "skills": skills,
                "selected_limit": self.selected_limit,
            },
        )
        selections = data.get("selected", [])
        if not isinstance(selections, list):
            raise ValueError("skill selector response missing selected list")
        selected: list[str] = []
        payloads: list[dict[str, Any]] = []
        for item in selections:
            if not isinstance(item, dict):
                continue
            skill_name = str(item.get("skill_name", "")).strip()
            if skill_name not in skills or skill_name in selected:
                continue
            selected.append(skill_name)
            payloads.append(
                {
                    "skill_name": skill_name,
                    "relevance": _clamp_float(item.get("relevance"), 0.5, "invalid skill relevance from LLM"),
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
            if len(selected) >= self.selected_limit:
                break
        return SkillSelection(
            selected_skills=selected,
            source="llm",
            rationale=str(data.get("rationale", "")).strip(),
            selections=payloads,
        )

    def _normalize(
        self,
        data: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        skills = context.get("skills") or {}
        skill_names = set(skills) if isinstance(skills, Mapping) else set()
        raw_selected = data.get("selected")
        if not isinstance(raw_selected, list):
            raise ValueError("skill selector response missing selected list")
        selected: list[dict[str, Any]] = []
        for item in raw_selected:
            if not isinstance(item, dict):
                continue
            skill_name = str(item.get("skill_name", "")).strip()
            if skill_name not in skill_names:
                continue
            selected.append(
                {
                    "skill_name": skill_name,
                    "relevance": _clamp_float(item.get("relevance"), 0.5, "invalid skill relevance from LLM"),
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
            if len(selected) >= self.selected_limit:
                break
        if raw_selected and not selected:
            raise ValueError("skill selector selected only unknown skills")
        return {
            "selected": selected,
            "rationale": str(data.get("rationale", "")).strip()[:500],
        }


def _skill_selector_prompt(state: AgentState, context: dict[str, Any]) -> str:
    skills = context.get("skills") or {}
    selected_limit = int(context.get("selected_limit") or 5)
    return render_prompt(
        "user/skill_selector.md",
        selected_limit=selected_limit,
        title=state.get("title", ""),
        description=state.get("description", ""),
        project_profile=json.dumps(state.get("project_profile", {}), ensure_ascii=False),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False),
        current_step=state.get("current_step", ""),
        memory_context=_truncate_text(str(state.get("memory_context", "")), 3000),
        code_context=_truncate_text(json.dumps(state.get("code_context", {}), ensure_ascii=False), 3000),
        available_skills=json.dumps(_skills_payload(skills), ensure_ascii=False),
    )


def _skills_payload(skills: Mapping[str, SkillSpec]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for skill in skills.values():
        payload.append(
            {
                "name": skill.name,
                "description": skill.description,
                "triggers": skill.triggers[:20],
                "entrypoints": skill.entrypoints,
                "metadata": skill.metadata,
            }
        )
    return payload


def _dedupe_names(values: Any, skills: Mapping[str, SkillSpec]) -> list[str]:
    selected: list[str] = []
    if not isinstance(values, list):
        return selected
    for value in values:
        name = str(value).strip()
        if name in skills and name not in selected:
            selected.append(name)
    return selected
