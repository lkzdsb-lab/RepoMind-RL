"""Versioned session work records. No model calls and no long-term promotion."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from model.session_work import (
    EvidenceValidity, RequirementUpdate,
    EditScope, WorkItemContractError, WorkItemReview, WorkItemView, WorkKind,
    WorkStatus, parse_work_view, work_item_data,
)
from agent_runtime.memory.work_policy import (
    ALLOWED_REVIEW_OUTCOMES, is_active, items_requiring_review, reviewed_status,
)

from agent_runtime.memory.file_cache import source_revision
from agent_runtime.llm.findings import finding_change
from model.finding_updates import FindingChange
from loguru import logger
from agent_runtime.context.progress import register_progress
from utils import utc_now


WORK_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_messages (
 message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
 task_id TEXT, content TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_messages ON session_messages(session_id, created_at);
CREATE TABLE IF NOT EXISTS session_runs (
 task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
 topic_id TEXT REFERENCES topics(topic_id), message_id TEXT NOT NULL,
 status TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0,
 before_snapshot TEXT NOT NULL, checkpoint TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_work_items (
 item_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
 topic_id TEXT NOT NULL REFERENCES topics(topic_id), kind TEXT NOT NULL,
 version INTEGER NOT NULL, content TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_work_topic ON session_work_items(session_id, topic_id);
CREATE TABLE IF NOT EXISTS session_work_versions (
 item_id TEXT NOT NULL REFERENCES session_work_items(item_id), version INTEGER NOT NULL,
 task_id TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(item_id, version)
);
"""


def stable_id(prefix: str, *values: Any) -> str:
    return prefix + hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def work_items(connection: sqlite3.Connection, session_id: str, topic_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT content FROM session_work_items WHERE session_id=? AND topic_id=? ORDER BY item_id",
        (session_id, topic_id),
    ).fetchall()
    return [work_item_data(json.loads(row[0]), stored=True) for row in rows]


def save_item(connection: sqlite3.Connection, item: dict[str, Any], task_id: str) -> dict[str, Any]:
    item = work_item_data(item)
    previous = connection.execute("SELECT content FROM session_work_items WHERE item_id=?", (item["item_id"],)).fetchone()
    old = work_item_data(json.loads(previous[0]), stored=True) if previous else {}
    if old and any(item[key] != old[key] for key in ("item_id", "session_id", "topic_id", "kind")):
        raise WorkItemContractError(f"Cannot change identity of session work item {item['item_id']}")
    value = {key: val for key, val in item.items() if key not in {"version", "updated_at"}}
    if value == {key: val for key, val in old.items() if key not in {"version", "updated_at"}}:
        return old
    value.update(version=int(old.get("version", 0)) + 1, updated_at=utc_now())
    value = work_item_data(value, stored=True)
    payload = json.dumps(value, ensure_ascii=False)
    connection.execute(
        "INSERT INTO session_work_items VALUES (?,?,?,?,?,?,?) ON CONFLICT(item_id) DO UPDATE SET "
        "version=excluded.version,content=excluded.content,updated_at=excluded.updated_at",
        (value["item_id"], value["session_id"], value["topic_id"], value["kind"], value["version"], payload, value["updated_at"]),
    )
    connection.execute("INSERT INTO session_work_versions VALUES (?,?,?,?,?)",
                       (value["item_id"], value["version"], task_id, payload, value["updated_at"]))
    logger.bind(task_id=task_id).debug("session work saved id={} version={} previous={}",
                                      value["item_id"], value["version"], old.get("version", 0))
    return value


def _item_fields(state: dict[str, Any], kind: WorkKind, identity: str, content: str) -> dict[str, Any]:
    return dict(item_id=stable_id("work_", state["session_id"], state["topic_id"], kind, identity),
                session_id=state["session_id"], topic_id=state["topic_id"], kind=kind,
                description=content, status=WorkStatus.PENDING, source_task_id=state["task_id"],
                source_message_id=state.get("message_id", ""), source_refs=[], source_revisions={},
                last_review_task_id="", review_reason="")


def create_requirement(state: dict[str, Any], identity: str, content: str, *, quote: str,
                       source_message_id: str, edit_scope: EditScope | str = EditScope.NONE, raw_request: bool | None = None,
                       replaces: str | None = None) -> dict[str, Any]:
    fields = _item_fields(state, WorkKind.REQUIREMENT, identity, content)
    fields.update(quote=quote, edit_scope=edit_scope, source_message_id=source_message_id,
                  source_refs=[f"message:{source_message_id}"])
    if raw_request is not None:
        fields["raw_request"] = raw_request
    if replaces is not None:
        fields["replaces"] = replaces
    return work_item_data(fields)


def create_finding(state: dict[str, Any], candidate: dict[str, Any], revisions: dict[str, str | None]) -> dict[str, Any]:
    key = candidate["candidate_id"]
    fields = _item_fields(state, WorkKind.FINDING, state["task_id"] + key, candidate["claim"])
    fields.update(candidate_id=key, finding=dict(candidate), source_revisions=revisions, source_refs=[f"finding:{key}"])
    return work_item_data(fields)


def create_action(state: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    fields = _item_fields(state, WorkKind.ACTION, step["description"].strip(), step["description"])
    fields.update(status=WorkStatus.BLOCKED if step.get("status") == "blocked" else WorkStatus.PENDING,
                  source_refs=[f"plan:{step.get('id', '')}"], plan_status=step.get("status"))
    return work_item_data(fields)


def apply_requirements(connection: sqlite3.Connection, state: dict[str, Any], updates: list[dict[str, Any]]) -> None:
    messages = {row[0]: row[1] for row in connection.execute(
        "SELECT message_id,content FROM session_messages WHERE session_id=? AND task_id=?",
        (state["session_id"], state["task_id"]),
    )}
    items = {item["item_id"]: item for item in work_items(connection, state["session_id"], state["topic_id"])}
    for raw in updates:
        update = RequirementUpdate.model_validate(raw)
        origin = next((key for key, text in messages.items() if update.quote in text), None)
        if not origin:
            raise ValueError("Requirement quote must occur in a user message from this run")
        if update.operation != "add":
            old = items.get(update.item_id)
            if not old or old["kind"] != WorkKind.REQUIREMENT:
                raise ValueError("Cannot replace/revoke an unknown requirement")
            old.update(status=WorkStatus.REVOKED if update.operation == "revoke" else WorkStatus.SUPERSEDED,
                       change_quote=update.quote, change_message_id=origin)
            save_item(connection, old, state["task_id"])
        if update.operation != "revoke":
            item = create_requirement(state, origin + update.quote, update.description, source_message_id=origin,
                                      quote=update.quote, edit_scope=update.edit_scope, replaces=update.item_id)
            if item["item_id"] not in items:
                items[item["item_id"]] = save_item(connection, item, state["task_id"])


def collect_progress(connection: sqlite3.Connection, state: dict[str, Any]) -> None:
    existing = work_items(connection, state["session_id"], state["topic_id"])
    findings = {item.get("candidate_id"): item for item in existing if item["kind"] == WorkKind.FINDING}
    revisions: dict[str, str | None] = {}
    for candidate in state.get("draft_findings", []):
        key = candidate["candidate_id"]
        old = findings.get(key)
        for location in candidate.get("locations", []):
            path = location["file_path"]
            if path not in revisions:
                revisions[path] = source_revision(state, path)
        # A rephrased historical finding is not new independent evidence.
        item = dict(old) if old else create_finding(state, candidate, {
            loc["file_path"]: revisions[loc["file_path"]] for loc in candidate.get("locations", [])})
        item.update(description=candidate["claim"], candidate_id=key, finding=dict(candidate))
        refs = set(item.get("source_refs", []))
        for event in state.get("finding_update_events", []):
            if event["candidate_id"] == key and event["change"] != FindingChange.METADATA:
                refs.update(f"task:{state['task_id']}:{ref}" for ref in event["evidence_refs"])
        new_evidence = refs - set(item.get("source_refs", []))
        item["source_refs"] = sorted(refs)
        # New source anchors get a current snapshot; existing anchors keep their
        # original revision until an explicit review refreshes them.
        item["source_revisions"] = dict(item.get("source_revisions", {}))
        for location in candidate.get("locations", []):
            item["source_revisions"].setdefault(location["file_path"], revisions[location["file_path"]])
        change = finding_change(old["finding"], candidate) if old else FindingChange.CREATED
        if old and (change in {FindingChange.REVISED, FindingChange.EVIDENCE} or new_evidence):
            item.update(status=WorkStatus.PENDING, last_review_task_id="", review_reason="")
            logger.bind(task_id=state["task_id"]).info(
                "finding review invalidated id={} change={} new_evidence={}", key, change.value, bool(new_evidence))
        findings[key] = save_item(connection, item, state["task_id"])
    # Blockers, plus unfinished steps of failed runs, survive summary eviction.
    for step in (state.get("work_plan") or {}).get("steps", []):
        if (step.get("status") != "blocked" and not (
            state.get("status") == "failed" and step.get("status") in {"pending", "in_progress"}
        )) or not step.get("description"):
            continue
        item = create_action(state, step)
        prior = next((entry for entry in existing if entry["item_id"] == item["item_id"]), None)
        if prior:
            continue
        save_item(connection, item, state["task_id"])


def active_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if is_active(item)]


def refresh_validity(state: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    revisions: dict[str, str | None] = {}
    result = []
    for item in items:
        item = work_item_data(item)
        refs = item.get("source_revisions") or {}
        for path in refs:
            if path not in revisions:
                revisions[path] = source_revision(state, path)
        validity = EvidenceValidity.UNKNOWN if not refs or any(not value for value in refs.values()) else (
            EvidenceValidity.CURRENT if all(revisions[path] == value for path, value in refs.items()) else EvidenceValidity.STALE)
        result.append(WorkItemView(item=parse_work_view(item).item, evidence_validity=validity,
                                  historical=item.get("source_task_id") != state.get("task_id")).to_flat_dict())
    return result


def apply_reviews(connection: sqlite3.Connection, state: dict[str, Any]) -> None:
    judgement = state.get("completion_judgement") or {}
    if judgement.get("status") != "succeeded":
        return
    current = {item["item_id"]: item for item in work_items(connection, state["session_id"], state["topic_id"])}
    for raw in judgement.get("work_item_reviews", []):
        review = WorkItemReview.model_validate(raw).model_dump(mode="json")
        item = current.get(review["item_id"])
        if not item or item["version"] != review["version"]:
            continue  # Already applied, or superseded since this review.
        outcome = review["outcome"]
        if (outcome in {"pending", "blocked"} and item.get("status") == outcome
                and item.get("last_review_task_id") == state["task_id"]
                and item.get("review_evidence_refs") == review["evidence_refs"]):
            continue  # Rephrasing a pending verdict is not progress.
        item.update(status=reviewed_status(item["kind"], outcome).value,
                    review_reason=review["reason"], last_review_task_id=state["task_id"],
                    review_evidence_refs=review["evidence_refs"])
        if outcome in {"confirmed", "satisfied", "rejected"}:
            item["source_revisions"] = {path: source_revision(state, path) for path in item.get("source_revisions", {})}
        save_item(connection, item, state["task_id"])
        if outcome in {"confirmed", "satisfied", "rejected"}:
            register_progress(state, [f"review:{item['item_id']}:{outcome}"], source="work_review")


def validate_work_reviews(state: dict[str, Any], judgement: dict[str, Any]) -> list[str]:
    """Validate coverage/provenance, not the semantic truth of an LLM verdict."""
    items = {item["item_id"]: item for item in items_requiring_review(state.get("session_work_items", []), state.get("task_id", ""))}
    errors, seen = [], set()
    valid_refs = current_evidence_refs(state)
    valid_refs.update(f"finding:{entry['candidate_id']}" for entry in judgement.get("reviewed_findings", [])
                      if entry.get("verdict") in {"confirmed", "rejected"})
    for review in judgement.get("work_item_reviews", []):
        key = review["item_id"]
        item = items.get(key)
        if key in seen or not item or review["version"] != item["version"]:
            errors.append(f"Invalid/duplicate work item or stale version: {key}")
            continue
        seen.add(key)
        outcome = review["outcome"]
        allowed = ALLOWED_REVIEW_OUTCOMES[WorkKind(item["kind"])]
        if outcome not in allowed:
            errors.append(f"Invalid outcome for {key}: {outcome}")
        if outcome in {"satisfied", "confirmed", "rejected"}:
            refs = set(review["evidence_refs"])
            if not refs or not refs.issubset(valid_refs):
                errors.append(f"Missing/invalid current evidence for {key}")
        if outcome in {"pending", "blocked"} and judgement.get("decision") == "complete":
            errors.append(f"Unresolved work item: {key}")
    if judgement.get("decision") == "complete":
        errors.extend(f"Unreviewed work item: {key}" for key in items.keys() - seen)
    return errors


def current_evidence_refs(state: dict[str, Any]) -> set[str]:
    refs = set()
    for index, call in enumerate(state.get("tool_calls", [])):
        output = call.get("output") if isinstance(call.get("output"), dict) else {}
        if not call.get("error") and not output.get("error") and not output.get("skipped") and call.get("name") not in {
            "read_session_memory", "finish", "request_user_input", "enter_plan_mode", "exit_plan_mode",
            "EnterPlanMode", "ExitPlanMode"
        }:
            refs.add(f"tool:{index}")
    return refs
