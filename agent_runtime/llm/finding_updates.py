"""The single admission point for policy and observer candidate commands."""
from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import ValidationError

from agent_runtime.context.progress import register_progress, tool_evidence_key
from agent_runtime.llm.findings import (
    normalize_finding_candidates, new_candidate_id, finding_change, shares_source_anchor, normalize_locations,
)
from agent_runtime.memory.file_cache import source_revision
from agent_runtime.memory.working import current_evidence_refs
from model.finding_updates import FindingChange, FindingOperation, FindingUpdateEvent, FindingUpdateResult
from model.llm import FindingUpdateResponse


def normalize_finding_updates(values: Any) -> list[dict[str, Any]]:
    return [FindingUpdateResponse.model_validate(item).model_dump(mode="json") for item in (values or [])]


def evidence_manifest(state: dict[str, Any]) -> list[dict[str, Any]]:
    valid = current_evidence_refs(state)
    return [{"ref": f"tool:{index}", "name": call.get("name"),
             "file_path": (call.get("output") or {}).get("file_path", ""),
             "command": str((call.get("output") or {}).get("command", ""))[:300]}
            for index, call in enumerate(state.get("tool_calls", []))
            if f"tool:{index}" in valid][-12:]


def _evidence(state: dict[str, Any], refs: list[str]) -> list[str]:
    if not refs or not set(refs).issubset(current_evidence_refs(state)):
        raise ValueError("Use successful current-run tool references; memory reads are not evidence")
    keys = []
    for ref in sorted(set(refs)):
        call = state["tool_calls"][int(ref.split(":")[1])]
        output = call.get("output") or {}
        path, revision = output.get("file_path"), output.get("file_revision")
        if path and revision and source_revision(state, path) != revision:
            raise ValueError(f"Evidence {ref} refers to stale source; read the current revision")
        keys.append(tool_evidence_key(call))
    return sorted(set(keys))


def apply_finding_updates(state: dict[str, Any], values: list[dict[str, Any]], *, source: str) -> FindingUpdateResult:
    """
        三种 operation：
        create
        enrich：ID 必须存在，claim 必须与旧值完全一致
        revise：ID 必须存在，revision\_reason 不能为空；接受新的候选内容
    """
    # 获取未审核的 finding
    result = FindingUpdateResult(findings=normalize_finding_candidates(state.get("draft_findings", [])))
    current = {item["candidate_id"]: item for item in result.findings}
    # 获取由历史 session 得到的 finding
    historical = {item["candidate_id"]: normalize_finding_candidates([item["finding"]])[0]
                  for item in state.get("session_work_items", []) if item.get("kind") == "finding"}
    # 获取更新过的 finding
    prior_events = [FindingUpdateEvent.model_validate(item).model_dump(mode="json")
                    for item in state.get("finding_update_events", [])]
    for raw in values or []:
        try:
            update = FindingUpdateResponse.model_validate(raw)
            refs = sorted(set(update.evidence_refs))
            keys = _evidence(state, refs)
            candidate = normalize_finding_candidates([update.model_dump(mode="json")])[0]
            operation = update.operation
            # 从未审核的 finding 和 记忆中的 finding 提取这轮更新过的 old 信息
            old = current.get(update.candidate_id) or historical.get(update.candidate_id)

            # 处理新增的 findings
            if operation == FindingOperation.CREATE:
                if update.candidate_id:
                    raise ValueError("create requires an empty candidate_id")
                candidate["candidate_id"] = new_candidate_id(candidate)
                old = current.get(candidate["candidate_id"]) or historical.get(candidate["candidate_id"])
                if old:
                    # Replaying a create cannot undo an accepted revision/metadata update.
                    candidate = dict(old)
            elif not old:
                raise ValueError("Unknown candidate_id; read_session_memory can retrieve omitted candidates")
            if old:
                if operation == FindingOperation.ENRICH and candidate["claim"] != old["claim"]:
                    raise ValueError("enrich must retain the exact claim; use revise to change the conclusion")
                if operation == FindingOperation.REVISE and not update.revision_reason.strip():
                    raise ValueError("revise requires revision_reason")
                if not shares_source_anchor(old, candidate):
                    raise ValueError("Update must retain a source anchor of the existing candidate")
                if operation != FindingOperation.REVISE:
                    candidate["claim"] = old["claim"]
                    candidate["locations"] = normalize_locations(old["locations"] + candidate["locations"])
                    candidate["related_tests"] = sorted(set(old["related_tests"]) | set(candidate["related_tests"]))
                change = finding_change(old, candidate)
            else:
                if not candidate["locations"]:
                    raise ValueError("New findings need a concrete source location")
                change = FindingChange.CREATED
            previous_keys = {key for event in prior_events if event["candidate_id"] == candidate["candidate_id"]
                             for key in event["evidence_keys"]}
            if change in {FindingChange.UNCHANGED, FindingChange.METADATA} and set(keys) - previous_keys:
                change = FindingChange.EVIDENCE
            event = FindingUpdateEvent(candidate_id=candidate["candidate_id"], operation=operation,
                                       change=change, evidence_refs=refs, evidence_keys=keys,
                                       reason=update.revision_reason)
            current[candidate["candidate_id"]] = candidate
            result.events.append(event)
            if change != FindingChange.UNCHANGED:
                prior_events.append(event.model_dump(mode="json"))
                # Clearing the current verdict prevents new claims being reported with an old review.
                state["completion_judgement"] = {}
                if change != FindingChange.METADATA:
                    register_progress(state, ["finding_evidence:" + key for key in keys], source=source)
            logger.bind(task_id=state.get("task_id")).info(
                "finding update source={} id={} operation={} change={} refs={}",
                source, candidate["candidate_id"], operation.value, change.value, refs)
        except (ValidationError, ValueError, IndexError) as exc:
            identity = raw.get("candidate_id", "new") if isinstance(raw, dict) else "new"
            error = f"{identity or 'new'}: {exc}"
            result.errors.append(error)
            logger.bind(task_id=state.get("task_id")).warning("finding update rejected source={} error={}", source, error)
    result.findings = list(current.values())
    state["draft_findings"] = result.findings
    state["finding_update_events"] = prior_events
    state["finding_update_errors"] = result.errors
    return result
