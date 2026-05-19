from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanResponse(BaseModel):
    plan: list[str] = Field(default_factory=list)


class TaskAnalysisResponse(BaseModel):
    task_type: Literal["BUG_FIX", "FEATURE_IMPL", "DIAGNOSE"]
    task_category: str
    entities: list[str]
    acceptance_criteria: list[str]
    risk_notes: list[str]
    search_hints: list[str]


class ObservationResponse(BaseModel):
    latest_tool: str = "unknown"
    status: str = "inconclusive"
    summary: str = ""
    new_findings: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    next_search_terms: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class FinalReportResponse(BaseModel):
    summary: str = ""
    work_done: list[str] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    test_results: list[str] = Field(default_factory=list)
    has_patch: bool = False
    patch_status: str = ""
    next_steps: list[str] = Field(default_factory=list)


class MemoryQueryPlanResponse(BaseModel):
    queries: list[str]
    rationale: str = ""


class MemorySelectionResponse(BaseModel):
    memory_id: str
    relevance: float
    reason: str = ""


class MemoryRerankResponse(BaseModel):
    selected: list[MemorySelectionResponse]


class CodeContextQueryPlanResponse(BaseModel):
    queries: list[str]
    rationale: str = ""


class CodeContextSelectionResponse(BaseModel):
    candidate_id: str
    relevance: float
    reason: str = ""


class CodeContextRerankResponse(BaseModel):
    selected: list[CodeContextSelectionResponse]
    rationale: str = ""


class SkillSelectionResponse(BaseModel):
    skill_name: str
    relevance: float
    reason: str = ""


class SkillSelectorResponse(BaseModel):
    selected: list[SkillSelectionResponse]
    rationale: str = ""


class ActionChoiceResponse(BaseModel):
    action: str = ""
    reason: str = ""


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


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    raw: dict = field(default_factory=dict)
    parsed: Any = None
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
