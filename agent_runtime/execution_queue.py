from __future__ import annotations

from typing import Any

from model.agent.graph import AgentState
from completion_state import _completion_signal_present


ACTIVE_STATUSES = {"pending", "in_progress", "blocked"}


def normalize_execution_queue(queue: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """ 提取 execution_queue 中的信息"""
    items: list[dict[str, Any]] = []
    for raw in queue or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["kind"] = str(item.get("kind") or "").strip()
        item["target_files"] = [
            str(path).strip()
            for path in item.get("target_files", []) or []
            if str(path).strip()
        ]
        status = str(item.get("status") or "pending").strip().lower()
        if status not in {"pending", "in_progress", "completed", "blocked", "skipped"}:
            status = "pending"
        item["status"] = status
        items.append(item)
    return items


def current_execution_item(state: AgentState) -> dict[str, Any] | None:
    for item in normalize_execution_queue(state.get("execution_queue", [])):
        if str(item.get("status") or "") in ACTIVE_STATUSES:
            return item
    return None


def reconcile_execution_queue(state: AgentState) -> list[dict[str, Any]]:
    """ 处理执行队列的状态"""
    queue = normalize_execution_queue(state.get("execution_queue", []))
    if not queue:
        return queue
    completion_signal = _completion_signal_present(state)
    verification_stale = bool(state.get("verification_stale", False))
    pending_resolution = state.get("pending_resolution") or {}
    resolution_kind = str(pending_resolution.get("kind") or "").strip().lower()
    has_recovery = resolution_kind == "recovery"
    has_deferred = resolution_kind == "deferred"

    patched: list[dict[str, Any]] = []
    for item in queue:
        next_item = dict(item)
        kind = str(next_item.get("kind") or "")
        status = str(next_item.get("status") or "pending")
        if status in {"completed", "skipped"}:
            patched.append(next_item)
            continue
        if has_recovery and kind == "patch":
            next_item["status"] = "blocked"
            next_item["status_reason"] = "pending_resolution:recovery"
        elif has_deferred and kind == "patch":
            next_item["status"] = "in_progress"
            next_item["status_reason"] = "pending_resolution:deferred"
        elif completion_signal and kind == "patch":
            next_item["status"] = "skipped"
            next_item["status_reason"] = "completion_confirmed"
        elif completion_signal and kind == "verify" and not verification_stale:
            next_item["status"] = "completed"
            next_item["status_reason"] = "verification_satisfied"
        elif status == "blocked" and not has_recovery:
            next_item["status"] = "pending"
            next_item.pop("status_reason", None)
        patched.append(next_item)
    return patched


def advance_execution_queue_for_patch(
    state: AgentState | dict[str, Any],
    changed_files: set[str],
) -> list[dict[str, Any]]:
    """ 每次执行完 patch 更新 execution queue 标记为已执行"""
    queue = normalize_execution_queue((state or {}).get("execution_queue", []))
    if not changed_files:
        return queue
    for item in queue:
        if str(item.get("status") or "pending") not in {"pending", "in_progress", "blocked"}:
            continue
        if str(item.get("kind") or "") != "patch":
            continue
        targets = {
            str(path).strip()
            for path in item.get("target_files", []) or []
            if str(path).strip()
        }
        if targets and targets.issubset(changed_files):
            item["status"] = "completed"
            item["status_reason"] = "patch_applied"
            break
    return queue


def advance_execution_queue_for_verification(
    state: AgentState | dict[str, Any],
) -> list[dict[str, Any]]:
    queue = normalize_execution_queue((state or {}).get("execution_queue", []))
    for item in queue:
        if str(item.get("status") or "pending") not in {"pending", "in_progress", "blocked"}:
            continue
        if str(item.get("kind") or "") == "verify":
            item["status"] = "completed"
            item["status_reason"] = "verification_passed"
            break
    return queue
