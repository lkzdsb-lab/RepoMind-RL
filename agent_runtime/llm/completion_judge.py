"""Completion judgement before an agent run is finalized."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from ext.tool_summaries import read_file_summaries, tool_call_summaries
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import CompletionJudgeResponse
from prompts.templates import load_prompt, render_prompt
from utils import _truncate_text, _safe_float, _clean_string_list


class CompletionJudge(Protocol):
    def judge(self, state: AgentState) -> dict[str, Any]:
        ...


@dataclass
class RuleBasedCompletionJudge:
    """Deterministic fallback that preserves the historical finish behavior."""

    def judge(self, state: AgentState) -> dict[str, Any]:
        return {
            "decision": "complete",
            "reason": "Rule-based completion judge preserves existing finish behavior.",
            "questions": [],
            "suggested_next_action": "",
            "confidence": 0.5,
            "source": "rule_based",
        }


@dataclass
class LLMCompletionJudge:
    llm_config: LLMConfig
    fallback: CompletionJudge | None = None

    def __post_init__(self) -> None:
        self.fallback = self.fallback or RuleBasedCompletionJudge()
        self.node = LLMJsonNode(
            name="completion_judge",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/completion_judge.md"),
            build_prompt=_completion_judge_prompt,
            fallback=lambda state, context: self.fallback.judge(state) if self.fallback else {},
            response_model=CompletionJudgeResponse,
            normalize=_normalize_completion_judge,
        )

    def judge(self, state: AgentState) -> dict[str, Any]:
        return self.node.run(
            state,
            {"fallback_judgement": self.fallback.judge(state) if self.fallback else {}},
        )


def _completion_judge_prompt(state: AgentState, context: dict[str, Any]) -> str:
    return render_prompt(
        "user/completion_judge.md",
        title=state.get("title", ""),
        description=state.get("description", ""),
        status=state.get("status", ""),
        current_step=state.get("current_step", ""),
        error=state.get("error", ""),
        project_profile=json.dumps(state.get("project_profile", {}), ensure_ascii=False),
        verification_required=json.dumps(bool(state.get("verification_required", True))),
        verification_reason=state.get("verification_reason", ""),
        verification_stale=json.dumps(bool(state.get("verification_stale", False))),
        verification_commands=json.dumps(
            state.get("verification_commands", [])[-5:],
            ensure_ascii=False,
            default=str,
        ),
        plan_mode=json.dumps(bool(state.get("plan_mode", False))),
        plan_mode_approved=json.dumps(bool(state.get("plan_mode_approved", False))),
        debug_technical_plan=_truncate_text(str(state.get("debug_technical_plan", "")), 5000),
        plan_mode_evaluation=_truncate_text(str(state.get("plan_mode_evaluation", "")), 3000),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False, default=str),
        plan=json.dumps(state.get("plan", []), ensure_ascii=False),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
        read_files=json.dumps(
            read_file_summaries(state, limit=12, excerpt_chars=2500),
            ensure_ascii=False,
            default=str,
        ),
        tool_calls=json.dumps(tool_call_summaries(state), ensure_ascii=False, default=str),
        test_results=json.dumps(state.get("test_results", [])[-5:], ensure_ascii=False, default=str),
        edit_results=json.dumps(state.get("edit_results", [])[-5:], ensure_ascii=False, default=str),
        patch_summary=state.get("patch_summary") or "",
        has_patch=json.dumps(bool(state.get("patch"))),
        llm_observations=json.dumps(
            _trim_observations(state.get("llm_observations", [])),
            ensure_ascii=False,
            default=str,
        ),
        user_inputs=json.dumps(state.get("user_inputs", []), ensure_ascii=False, default=str),
        selected_skills=json.dumps(state.get("selected_skills", []), ensure_ascii=False),
        skill_context=_truncate_text(
            json.dumps(state.get("skill_context", []), ensure_ascii=False, default=str),
            1600,
        ),
        fallback_judgement=json.dumps(
            context.get("fallback_judgement", {}),
            ensure_ascii=False,
            default=str,
        ),
    )


def _normalize_completion_judge(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    decision = str(data.get("decision") or "complete").strip().lower()
    if decision not in {"complete", "needs_user_input", "continue"}:
        decision = "complete"
    questions = _clean_string_list(data.get("questions"), 3, 300)
    reason = str(data.get("reason") or "").strip()[:1000]
    suggested_next_action = str(data.get("suggested_next_action") or "").strip()[:120]
    if decision == "needs_user_input" and not questions:
        decision = "continue"
        suggested_next_action = suggested_next_action or "search_code_context"
    if decision != "needs_user_input":
        questions = []
    confidence = _safe_float(data.get("confidence"), default=0.5)
    return {
        "decision": decision,
        "reason": reason,
        "questions": questions,
        "suggested_next_action": suggested_next_action,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _trim_observations(observations: Any) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    if not isinstance(observations, list):
        return trimmed
    for item in observations[-5:]:
        if not isinstance(item, dict):
            continue
        trimmed.append(
            {
                "latest_tool": item.get("latest_tool"),
                "status": item.get("status"),
                "summary": str(item.get("summary") or "")[:500],
                "new_findings": _clean_string_list(item.get("new_findings"), 5, 220),
                "missing_context": _clean_string_list(item.get("missing_context"), 5, 220),
            }
        )
    return trimmed
