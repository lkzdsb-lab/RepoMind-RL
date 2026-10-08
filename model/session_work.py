"""Flat JSON contracts shared by session storage, views and LLM review.

Storage states and review outcomes intentionally differ: satisfied -> completed.
Runtime view fields never participate in persisted work-item version comparison.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, TypeAdapter, ValidationError, model_validator


class StringValue(str, Enum):
    def __str__(self) -> str:
        return self.value


class WorkKind(StringValue):
    REQUIREMENT = "requirement"
    FINDING = "finding"
    # 显式阻塞的计划步骤，或失败运行里尚未完成的计划步骤
    ACTION = "action"


class WorkStatus(StringValue):
    PENDING = "pending"
    BLOCKED = "blocked"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    REVOKED = "revoked"
    SUPERSEDED = "superseded"


class EditScope(StringValue):
    NONE = "none"
    TESTS = "tests"
    REPOSITORY = "repository"


class ReviewOutcome(StringValue):
    SATISFIED = "satisfied"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    PENDING = "pending"


class EvidenceValidity(StringValue):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ProjectionAudience(StringValue):
    ANALYZER = "analyzer"
    PLANNER = "planner"
    POLICY = "policy"
    OBSERVER = "observer"
    COMPLETION = "completion"
    REPORTER = "reporter"
    SEARCH = "search"
    SKILLS = "skills"
    CODE_CONTEXT = "code_context"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkItemBase(ContractModel):
    item_id: str = Field(min_length=1, description="Stable identity, never recomputed on review.")
    session_id: str = Field(min_length=1, description="Owning conversation.")
    topic_id: str = Field(min_length=1, description="Owning task topic.")
    kind: WorkKind = Field(description="Discriminator: requirement, finding or action.")
    description: str = Field(description="Full description; prompt previews may truncate it.")
    status: WorkStatus = Field(description="Persisted processing state; restricted further by kind.")
    version: StrictInt = Field(default=1, ge=1, description="Allocated by storage on effective changes.")
    updated_at: str = Field(default="", description="Storage update timestamp.")
    source_task_id: str = Field(default="", description="Origin run; empty for unbound recovered messages.")
    source_message_id: str = Field(default="", description="Origin user message, when available.")
    source_refs: list[str] = Field(default_factory=list, description="message:, finding: or plan: origin references.")
    source_revisions: dict[str, str | None] = Field(default_factory=dict, description="Tracked file hashes; existing review refresh semantics retained.")
    last_review_task_id: str = Field(default="", description="Run that last reviewed this item.")
    review_reason: str = Field(default="", description="Latest review explanation.")
    review_evidence_refs: list[str] = Field(default_factory=list, description="Current tool:N or finding:ID review references.")

    @model_validator(mode="after")
    def valid_kind_status(self) -> WorkItemBase:
        if self.status not in ALLOWED_STATUSES[WorkKind(self.kind)]:
            raise ValueError(f"Status {self.status} is not allowed for {self.kind}")
        return self


class RequirementItem(WorkItemBase):
    kind: Literal["requirement"] = "requirement"
    quote: str = Field(description="User text anchoring this requirement.")
    edit_scope: EditScope = Field(description="Permission granted by this requirement, not a global prohibition.")
    raw_request: StrictBool = Field(default=False, description="True for a whole original message rather than an extracted clause.")
    replaces: str = Field(default="", description="Previous requirement ID replaced by this one.")
    change_quote: str = Field(default="", description="User quote supporting revocation/replacement.")
    change_message_id: str = Field(default="", description="Message supporting revocation/replacement.")


class FindingLocation(ContractModel):
    file_path: str = Field(min_length=1)
    symbol: str = ""
    start_line: StrictInt | None = Field(default=None, ge=1)
    end_line: StrictInt | None = Field(default=None, ge=1)


class FindingContent(ContractModel):
    candidate_id: str = Field(min_length=1)
    claim: str
    locations: list[FindingLocation] = Field(default_factory=list)
    related_tests: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    category: str = ""


class FindingItem(WorkItemBase):
    kind: Literal["finding"] = "finding"
    candidate_id: str = Field(min_length=1, description="Associated agent finding identity.")
    finding: FindingContent = Field(description="Complete normalized candidate, not a prompt preview.")

    @model_validator(mode="after")
    def matching_candidate(self) -> FindingItem:
        if self.candidate_id != self.finding.candidate_id:
            raise ValueError("candidate_id differs from finding.candidate_id")
        return self


class ActionItem(WorkItemBase):
    kind: Literal["action"] = "action"
    plan_status: Literal["pending", "in_progress", "done", "blocked"] = Field(description="Plan status when this obligation was collected.")


SessionWorkItem = Annotated[RequirementItem | FindingItem | ActionItem, Field(discriminator="kind")]
ALLOWED_STATUSES = {
    WorkKind.REQUIREMENT: frozenset({WorkStatus.PENDING, WorkStatus.BLOCKED, WorkStatus.COMPLETED, WorkStatus.REVOKED, WorkStatus.SUPERSEDED}),
    WorkKind.FINDING: frozenset({WorkStatus.PENDING, WorkStatus.BLOCKED, WorkStatus.CONFIRMED, WorkStatus.REJECTED}),
    WorkKind.ACTION: frozenset({WorkStatus.PENDING, WorkStatus.BLOCKED, WorkStatus.COMPLETED, WorkStatus.REJECTED}),
}
WORK_ITEM_ADAPTER = TypeAdapter(SessionWorkItem)


class WorkItemView(ContractModel):
    """Typed runtime wrapper; to_flat_dict retains the existing external shape."""
    item: SessionWorkItem
    historical: StrictBool | None = None
    evidence_validity: EvidenceValidity | None = None
    source_user_message: str | None = None

    def to_flat_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_unset=True)
        item = data.pop("item")
        item["kind"] = self.item.kind
        return {**item, **data}


class WorkItemContractError(ValueError):
    """Identifies the bad record and field without dumping user content."""


VIEW_FIELDS = frozenset({"historical", "evidence_validity", "source_user_message"})


def parse_work_view(value: Mapping[str, Any] | WorkItemView | WorkItemBase) -> WorkItemView:
    if isinstance(value, WorkItemView):
        value = value.to_flat_dict()
    elif isinstance(value, WorkItemBase):
        value = value.model_dump(mode="json", exclude_unset=True)
    if not isinstance(value, Mapping):
        raise WorkItemContractError("Session work item must be an object")
    try:
        item = WORK_ITEM_ADAPTER.validate_python({key: entry for key, entry in value.items() if key not in VIEW_FIELDS})
        return WorkItemView.model_validate({"item": item, **{key: value[key] for key in VIEW_FIELDS if key in value}})
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors(include_input=False))
        raise WorkItemContractError(f"Invalid session work item {value.get('item_id', '<unknown>')}: {details}") from exc


def work_item_data(value: Mapping[str, Any] | WorkItemBase | WorkItemView, *, stored: bool = False) -> dict[str, Any]:
    """Validate without adding absent defaults or persisting runtime view fields."""
    view = parse_work_view(value)
    data = view.item.model_dump(mode="json", exclude_unset=True)
    data["kind"] = view.item.kind
    if stored and not {"version", "updated_at"}.issubset(data):
        raise WorkItemContractError(f"Stored session work item {view.item.item_id} lacks version/updated_at")
    return data


class WorkItemBaseline(ContractModel):
    item_id: str = Field(min_length=1)
    version: StrictInt = Field(ge=1)


class WorkItemPreview(ContractModel):
    item_id: str
    version: StrictInt = Field(ge=1)
    kind: WorkKind
    status: WorkStatus
    description: str
    source_task_id: str | None
    source_message_id: str | None
    historical: StrictBool | None
    evidence_validity: EvidenceValidity | None
    detail_omitted: Literal[True] = True
    quote: str | None = None
    edit_scope: EditScope | None = None
    candidate_id: str | None = None


class HistoryPreview(ContractModel):
    turn_index: StrictInt | None
    user_message: str
    summary: str
    status: str | None


class TopicPreview(ContractModel):
    topic_id: str
    title: str


class UnassignedMessagePreview(ContractModel):
    message_id: str
    task_id: str | None
    content: str
    preview_only: Literal[True]


class SessionProjection(ContractModel):
    schema_version: Literal[1]
    audience: ProjectionAudience
    revision: str = Field(description="Fingerprint of all selected items, including budget omissions.")
    topic_id: str
    objective: str
    total_items: StrictInt = Field(ge=0)
    items: list[WorkItemPreview]
    omitted_items: StrictInt = Field(ge=0)
    detail_tool: Literal["read_session_memory"]
    history: list[HistoryPreview]
    history_omitted: StrictInt = Field(ge=0)
    topics: list[TopicPreview]
    preview_only: Literal[True]
    unassigned_messages: list[UnassignedMessagePreview]
    unassigned_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def complete_count(self) -> SessionProjection:
        if self.total_items != len(self.items) + self.omitted_items:
            raise ValueError("total_items must equal visible plus omitted items")
        return self


class WorkItemPage(ContractModel):
    items: list[dict[str, Any]]
    total: StrictInt = Field(ge=0)
    revision: str = Field(description="Fingerprint of all topic items and evidence validity for pagination.")
    next_offset: StrictInt | None = Field(ge=0)
    historical_notice: str

    @model_validator(mode="after")
    def valid_items(self) -> WorkItemPage:
        for item in self.items:
            parse_work_view(item)
        if len(self.items) > self.total:
            raise ValueError("Page has more items than total")
        return self


def validate_work_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep JSON transport while validating only the session-work fields.

    Before-snapshots may legitimately belong to the preceding topic.
    """
    result = dict(state)
    if "session_edit_scope" in result:
        try:
            result["session_edit_scope"] = EditScope(result["session_edit_scope"]).value
        except ValueError as exc:
            raise WorkItemContractError("Invalid session_edit_scope; expected none/tests/repository") from exc
    if "session_work_items" in result:
        if not isinstance(result["session_work_items"], list):
            raise WorkItemContractError("session_work_items must be a JSON array")
        result["session_work_items"] = [parse_work_view(item).to_flat_dict() for item in result["session_work_items"]]
        for item in result["session_work_items"]:
            if result.get("session_id") and item["session_id"] != result["session_id"]:
                raise WorkItemContractError(f"Session mismatch for work item {item['item_id']}")
            if result.get("topic_id") and item["topic_id"] != result["topic_id"]:
                raise WorkItemContractError(f"Topic mismatch for work item {item['item_id']}")
    if "session_work_baseline" in result:
        if not isinstance(result["session_work_baseline"], list):
            raise WorkItemContractError("session_work_baseline must be a JSON array")
        try:
            result["session_work_baseline"] = [WorkItemBaseline.model_validate(item).model_dump(mode="json")
                                               for item in result["session_work_baseline"]]
        except ValidationError as exc:
            raise WorkItemContractError("Invalid session_work_baseline; expected item_id and positive integer version") from exc
    for key in ("session_memory", "session_before_snapshot"):
        if isinstance(result.get(key), Mapping) and "work_items" in result[key]:
            if not isinstance(result[key]["work_items"], list):
                raise WorkItemContractError(f"{key}.work_items must be a JSON array")
            result[key] = {**result[key], "work_items": [parse_work_view(item).to_flat_dict()
                                                       for item in result[key]["work_items"]]}
    return result


class RequirementUpdate(BaseModel):
    item_id: str = ""
    quote: str = Field(min_length=1)
    description: str = Field(min_length=1)
    operation: Literal["add", "replace", "revoke"] = "add"
    edit_scope: EditScope = EditScope.NONE


class WorkItemReview(BaseModel):
    item_id: str
    version: int = Field(ge=1)
    outcome: ReviewOutcome
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
