"""Session work decisions shared by projection, review, permissions and reports.

Preserves the existing seven storage states and unknown-evidence behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

from model.session_work import (
    EditScope, EvidenceValidity, ProjectionAudience, ReviewOutcome, WorkItemView,
    WorkKind, WorkStatus, parse_work_view,
)


TERMINAL_STATUSES = frozenset({WorkStatus.COMPLETED, WorkStatus.REJECTED, WorkStatus.REVOKED, WorkStatus.SUPERSEDED})
ALLOWED_REVIEW_OUTCOMES = {
    WorkKind.REQUIREMENT: frozenset({ReviewOutcome.SATISFIED, ReviewOutcome.BLOCKED, ReviewOutcome.PENDING}),
    WorkKind.FINDING: frozenset({ReviewOutcome.CONFIRMED, ReviewOutcome.REJECTED, ReviewOutcome.BLOCKED, ReviewOutcome.PENDING}),
    WorkKind.ACTION: frozenset({ReviewOutcome.SATISFIED, ReviewOutcome.REJECTED, ReviewOutcome.BLOCKED, ReviewOutcome.PENDING}),
}
OUTCOME_STATUS = {outcome: WorkStatus.COMPLETED if outcome == ReviewOutcome.SATISFIED else WorkStatus(outcome.value)
                  for outcome in ReviewOutcome}
ItemInput = Mapping[str, Any] | WorkItemView


def is_active(item: ItemInput) -> bool:
    return parse_work_view(item).item.status not in TERMINAL_STATUSES


def requires_current_review(item: ItemInput, task_id: str) -> bool:
    view = parse_work_view(item)
    return view.item.status not in TERMINAL_STATUSES and (
        view.item.status != WorkStatus.CONFIRMED or view.item.last_review_task_id != task_id
        or view.evidence_validity == EvidenceValidity.STALE)


def items_requiring_review(items: Iterable[ItemInput], task_id: str) -> list[ItemInput]:
    return [item for item in items if requires_current_review(item, task_id)]


def should_project(item: ItemInput, audience: ProjectionAudience | str, task_id: str) -> bool:
    view = parse_work_view(item)
    if ProjectionAudience(audience) == ProjectionAudience.COMPLETION:
        return requires_current_review(view, task_id)
    return view.item.status not in TERMINAL_STATUSES or (
        view.item.kind == WorkKind.REQUIREMENT and view.item.status == WorkStatus.COMPLETED)


def grants_edit_scope(item: ItemInput) -> EditScope:
    value = parse_work_view(item).item
    if value.kind == WorkKind.REQUIREMENT and value.status not in {WorkStatus.REVOKED, WorkStatus.SUPERSEDED}:
        return value.edit_scope
    return EditScope.NONE


def effective_edit_scope(items: Iterable[ItemInput]) -> str:
    scopes = {grants_edit_scope(item) for item in items}
    return (EditScope.REPOSITORY if EditScope.REPOSITORY in scopes else
            EditScope.TESTS if EditScope.TESTS in scopes else EditScope.NONE).value


def is_current_confirmed_finding(item: ItemInput, task_id: str) -> bool:
    view = parse_work_view(item)
    return (view.item.kind == WorkKind.FINDING and view.item.status == WorkStatus.CONFIRMED
            and view.item.last_review_task_id == task_id and view.evidence_validity != EvidenceValidity.STALE)


def reviewed_status(kind: WorkKind | str, outcome: ReviewOutcome | str) -> WorkStatus:
    kind, outcome = WorkKind(kind), ReviewOutcome(outcome)
    if outcome not in ALLOWED_REVIEW_OUTCOMES[kind]:
        raise ValueError(f"Review outcome {outcome} is not allowed for {kind}")
    return OUTCOME_STATUS[outcome]


def is_unresolved_in_report(item: ItemInput, task_id: str) -> bool:
    """Report filtering retains its existing meaning, separate from review selection."""
    value = parse_work_view(item).item
    return value.status not in TERMINAL_STATUSES and (
        value.last_review_task_id != task_id or value.status in {WorkStatus.PENDING, WorkStatus.BLOCKED})
