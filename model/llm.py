from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _coerce_score(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        aliases = {
            "low": 0.25,
            "medium": 0.5,
            "med": 0.5,
            "high": 0.85,
        }
        if text in aliases:
            return aliases[text]
    return value


def _coerce_string_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        lines = [
            line.strip().lstrip("-*•").strip()
            for line in text.splitlines()
            if line.strip()
        ]
        if len(lines) > 1:
            return lines
        return [text]
    return [str(value).strip()] if str(value).strip() else []


class PlanResponse(BaseModel):
    plan: list[str] = Field(default_factory=list)
    user_update: str = ""

    @field_validator("plan", mode="before")
    @classmethod
    def _normalize_plan(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class TaskAnalysisResponse(BaseModel):
    task_type: Literal["BUG_FIX", "FEATURE_IMPL", "DIAGNOSE"]
    verification_required: bool = True
    verification_reason: str = ""
    task_category: str
    entities: list[str]
    acceptance_criteria: list[str]
    risk_notes: list[str]
    search_hints: list[str]
    user_update: str = ""

    @field_validator("entities", "acceptance_criteria", "risk_notes", "search_hints", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class ObservationResponse(BaseModel):
    latest_tool: str = "unknown"
    status: str = "inconclusive"
    summary: str = ""
    new_findings: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    invalidated_hypotheses: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    next_search_terms: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    user_update: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        return _coerce_score(value)


class FinalReportResponse(BaseModel):
    summary: str = ""
    work_done: list[str] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    test_results: list[str] = Field(default_factory=list)
    has_patch: bool = False
    patch_status: str = ""
    next_steps: list[str] = Field(default_factory=list)
    user_update: str = ""

    @field_validator("work_done", "candidate_files", "test_results", "next_steps", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class CompletionJudgeResponse(BaseModel):
    decision: Literal["complete", "needs_user_input", "continue"] = "complete"
    reason: str = ""
    questions: list[str] = Field(default_factory=list)
    suggested_next_action: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    user_update: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        return _coerce_score(value)

    @field_validator("questions", mode="before")
    @classmethod
    def _normalize_questions(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class MemoryQueryPlanResponse(BaseModel):
    queries: list[str]
    rationale: str = ""
    user_update: str = ""

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class MemorySelectionResponse(BaseModel):
    memory_id: str
    relevance: float
    reason: str = ""

    @field_validator("relevance", mode="before")
    @classmethod
    def _normalize_relevance(cls, value: Any) -> Any:
        return _coerce_score(value)


class MemoryRerankResponse(BaseModel):
    selected: list[MemorySelectionResponse]
    user_update: str = ""


class CodeContextQueryPlanResponse(BaseModel):
    queries: list[str]
    rationale: str = ""
    user_update: str = ""

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class CodeContextSelectionResponse(BaseModel):
    candidate_id: str
    relevance: float
    reason: str = ""

    @field_validator("relevance", mode="before")
    @classmethod
    def _normalize_relevance(cls, value: Any) -> Any:
        return _coerce_score(value)


class CodeContextRerankResponse(BaseModel):
    selected: list[CodeContextSelectionResponse]
    rationale: str = ""
    user_update: str = ""


class SkillSelectionResponse(BaseModel):
    skill_name: str
    relevance: float
    reason: str = ""

    @field_validator("relevance", mode="before")
    @classmethod
    def _normalize_relevance(cls, value: Any) -> Any:
        return _coerce_score(value)


class SkillSelectorResponse(BaseModel):
    selected: list[SkillSelectionResponse]
    rationale: str = ""
    user_update: str = ""


class ActionChoiceResponse(BaseModel):
    action: str = ""
    candidate_actions: list[str] = Field(default_factory=list)
    reason: str = ""
    action_input: dict[str, Any] = Field(default_factory=dict)
    uncertainty_questions: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    user_update: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return _coerce_score(value)


class ToolResultResponse(BaseModel):
    name: str = ""
    status: str = ""
    summary: str = ""


class ContextCompressionResponse(BaseModel):
    summary: str = ""
    current_goal: str = ""
    constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    key_observations: list[str] = Field(default_factory=list)
    tool_results: list[ToolResultResponse] = Field(default_factory=list)
    code_changes: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    user_update: str = ""


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    raw: dict = field(default_factory=dict)
    parsed: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    """ 后面需要通过 skill 的限制去保证 raw 内部的结构 """


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    response_format: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

@dataclass
class GuardDecision:
    state_key: str
    q_values: dict[str, float]
    legal_actions: list[str]
    hard_denied: dict[str, float]
    allow_list: list[str]
    allow_scores: dict[str, float]
    fallback_forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "q_values": self.q_values,
            "legal_actions": self.legal_actions,
            "hard_denied": self.hard_denied,
            "allow_list": self.allow_list,
            "allow_scores": self.allow_scores,
            "fallback_forced": self.fallback_forced,
        }
