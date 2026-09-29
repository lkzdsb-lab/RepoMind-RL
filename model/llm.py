from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator


def _empty_text_if_null(value: Any) -> Any:
    return "" if value is None else value


def _empty_list_if_null(value: Any) -> Any:
    return [] if value is None else value


EmptyText = Annotated[str, BeforeValidator(_empty_text_if_null)]
EmptyStringList = Annotated[list[str], BeforeValidator(_empty_list_if_null)]


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


class PlanResponse(BaseModel):
    plan: list[str] = Field(min_length=1, max_length=8)
    user_update: str = ""


class TaskAnalysisResponse(BaseModel):
    intent: Literal["diagnose", "implement", "explain", "review"]
    task_type: Literal["BUG_FIX", "FEATURE_IMPL", "DIAGNOSE"]
    task_category: str
    entities: list[str]
    acceptance_criteria: list[str]
    risk_notes: list[str]
    review_focus: list[str] = Field(default_factory=list)
    search_hints: EmptyStringList = Field(default_factory=list)
    historical_context: list[str] = Field(default_factory=list)
    user_update: str = ""


class MemoryExtractionItemResponse(BaseModel):
    candidate_id: str
    should_store: bool = True
    memory_type: Literal[
        "episodic", "semantic", "procedural", "anti_pattern", "preference"
    ]
    title: str
    knowledge: str
    applicability: str
    evidence_ids: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    scope_level: Literal["global", "user", "repo", "module", "file", "symbol"] = "repo"
    module: str = ""
    files: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    invalidation: str = ""
    constraint_scope: Literal[
        "", "task", "session", "repository", "user", "ambiguous"
    ] = ""
    constraint_durability: Literal[
        "", "temporary", "durable", "ambiguous"
    ] = ""
    constraint_explicit: bool = False
    semantic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_reason: str = ""
    rejection_reason: Literal[
        "",
        "temporary_constraint",
        "insufficient_evidence",
        "not_reusable",
        "duplicate_in_batch",
        "unsupported_claim",
        "scope_ambiguous",
        "one_off_detail",
        "wrong_origin_type",
    ] = ""

    @field_validator("semantic_confidence", mode="before")
    @classmethod
    def _normalize_semantic_confidence(cls, value: Any) -> Any:
        return _coerce_score(value)


class MemoryExtractionResponse(BaseModel):
    memories: list[MemoryExtractionItemResponse] = Field(default_factory=list, max_length=100)
    rationale: str = ""


class MemoryRelationAssessmentResponse(BaseModel):
    target_memory_id: str
    relation: Literal["duplicate", "refine", "conflict", "unrelated"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    merged_title: str = ""
    merged_knowledge: str = ""
    merged_applicability: str = ""
    merged_invalidation: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        return _coerce_score(value)


class MemoryRelationResponse(BaseModel):
    assessments: list[MemoryRelationAssessmentResponse] = Field(
        default_factory=list, max_length=20
    )
    rationale: str = ""


class FindingLocationResponse(BaseModel):
    file_path: str
    symbol: str = ""
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class DraftFindingResponse(BaseModel):
    candidate_id: str = ""
    claim: str
    locations: list[FindingLocationResponse] = Field(default_factory=list)
    related_tests: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    category: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        return _coerce_score(value)


class ObservationResponse(BaseModel):
    latest_tool: str = "unknown"
    status: str = "inconclusive"
    summary: str = ""
    new_findings: list[str] = Field(default_factory=list)
    finding_candidates: list[DraftFindingResponse] = Field(default_factory=list)
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
    findings: list[str] = Field(default_factory=list)
    work_done: list[str] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    test_results: list[str] = Field(default_factory=list)
    has_patch: bool = False
    patch_status: str = ""
    next_steps: list[str] = Field(default_factory=list)
    user_update: str = ""


class ReviewedFindingResponse(BaseModel):
    candidate_id: str
    verdict: Literal["confirmed", "rejected", "needs_more_evidence"]
    claim: str
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = ""
    recommended_next_action: EmptyText = ""

class CompletionJudgeResponse(BaseModel):
    decision: Literal["complete", "needs_user_input", "continue"]
    reason: str = ""
    questions: EmptyStringList = Field(default_factory=list)
    suggested_next_action: EmptyText = ""
    reviewed_findings: list[ReviewedFindingResponse] = Field(default_factory=list)
    missing_evidence: EmptyStringList = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    user_update: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        return _coerce_score(value)

class CodeContextQueryPlanResponse(BaseModel):
    queries: list[str]
    rationale: str = ""
    user_update: str = ""

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


class PlanStepResponse(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "done", "blocked"]


class PlanUpdateResponse(BaseModel):
    steps: list[PlanStepResponse] = Field(default_factory=list)
    current_focus: str = ""
    open_questions: list[str] = Field(default_factory=list)


class AnalysisUpdateResponse(BaseModel):
    file_path: str = Field(min_length=1)
    file_revision: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    dimension: str = Field(min_length=1, max_length=120)
    conclusion: str = Field(min_length=1, max_length=1000)
    finding_ids: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    status: Literal["partial", "complete"]


class ActionChoiceResponse(BaseModel):
    action: str = Field(min_length=1)
    reason: str = ""
    action_input: dict[str, Any] = Field(default_factory=dict)
    uncertainty_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    plan_update: PlanUpdateResponse = Field(default_factory=PlanUpdateResponse)
    draft_findings: list[DraftFindingResponse] = Field(default_factory=list)
    analysis_updates: list[AnalysisUpdateResponse] = Field(default_factory=list, max_length=20)
    user_update: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "q_values": self.q_values,
            "legal_actions": self.legal_actions,
            "hard_denied": self.hard_denied,
            "allow_list": self.allow_list,
            "allow_scores": self.allow_scores,
        }
