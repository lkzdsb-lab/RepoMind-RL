from __future__ import annotations

from typing import Any

from loguru import logger

from model.agent.graph import AgentState


QUEUE_STATUSES = {"pending", "in_progress", "completed", "blocked", "skipped"}
KIND_BY_CAPABILITY = {
    "read_code": "diagnose",
    "patch": "patch",
    "verification": "verify",
}


def validate_execution_queue(queue: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(queue or []):
        if not isinstance(raw, dict):
            logger.warning("ignoring non-object execution queue item index={}", index)
            continue
        item = dict(raw)
        item_id = str(item.get("id") or f"queue_item_{index + 1}").strip()
        if item_id in seen_ids:
            logger.warning("ignoring duplicate execution queue item id={}", item_id)
            continue
        seen_ids.add(item_id)
        item["id"] = item_id
        item["criterion_id"] = str(item.get("criterion_id") or "").strip()
        if not item["criterion_id"]:
            logger.warning(
                "ignoring execution queue item without criterion_id item_id={}",
                item_id,
            )
            continue
        item["kind"] = str(item.get("kind") or "").strip().lower()
        item["required_capability"] = str(item.get("required_capability") or "").strip()
        item["required"] = bool(item.get("required", True))
        item["target_files"] = [
            str(path).strip()
            for path in item.get("target_files", []) or []
            if str(path).strip()
        ]
        item["commands"] = [
            str(command).strip()
            for command in item.get("commands", []) or []
            if str(command).strip()
        ]
        status = str(item.get("status") or "pending").strip().lower()
        item["status"] = status if status in QUEUE_STATUSES else "pending"
        items.append(item)
    return items


def _completion_signal_present(state: AgentState) -> bool:
    judgement = state.get("completion_judgement") or {}
    return str(judgement.get("decision") or "").strip().lower() == "complete"


def current_execution_item(state: AgentState) -> dict[str, Any] | None:
    """ 获取执行 item"""
    queue = state.get("execution_queue", [])
    obligation = state.get("next_obligation") if isinstance(state.get("next_obligation"), dict) else {}
    compatible = [item for item in queue if _item_matches_obligation(item, obligation)]
    for status in ("in_progress", "blocked"):
        for item in compatible:
            if item.get("status") == status:
                return item
    return None


def reconcile_execution_queue(
    state: AgentState,
    obligation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """ 重新编排调整 execution queue"""
    queue = _copy_execution_queue(state.get("execution_queue", []))
    current_obligation = obligation if isinstance(obligation, dict) else state.get("next_obligation")
    if not isinstance(current_obligation, dict):
        current_obligation = {}
    obligation_kind = str(current_obligation.get("kind") or "")
    contract = state.get("goal_contract") if isinstance(state.get("goal_contract"), dict) else {}
    known_criteria = {
        str(item.get("id") or "")
        for item in contract.get("criteria", []) or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    ledger = state.get("progress_ledger") if isinstance(state.get("progress_ledger"), dict) else {}
    criterion_progress = ledger.get("criteria") if isinstance(ledger.get("criteria"), dict) else {}
    pending_resolution = state.get("pending_resolution") or {}
    resolution_kind = str(pending_resolution.get("kind") or "").strip().lower()

    active_assigned = False
    reconciled: list[dict[str, Any]] = []
    for item in queue:
        next_item = dict(item)
        item_criterion = str(next_item.get("criterion_id") or "")
        if item_criterion and known_criteria and item_criterion not in known_criteria:
            logger.warning(
                "ignoring execution queue item with unknown criterion item_id={} criterion_id={}",
                next_item.get("id"),
                item_criterion,
            )
            continue
        progress_entry = criterion_progress.get(item_criterion)
        if (
            item_criterion
            and isinstance(progress_entry, dict)
            and progress_entry.get("status") in {"passed", "not_required"}
        ):
            next_item["status"] = "completed"
            next_item["status_reason"] = "criterion_satisfied"
        status = str(next_item.get("status") or "pending")
        if status in {"completed", "skipped"}:
            reconciled.append(next_item)
            continue

        matches = _item_matches_obligation(next_item, current_obligation)
        if obligation_kind == "complete" or not matches:
            next_item["status"] = "pending"
            next_item.pop("status_reason", None)
        elif resolution_kind == "recovery" and next_item.get("kind") == "patch":
            next_item["status"] = "blocked"
            next_item["status_reason"] = "pending_resolution:recovery"
            active_assigned = True
        elif resolution_kind == "deferred" and next_item.get("kind") == "patch":
            next_item["status"] = "in_progress"
            next_item["status_reason"] = "pending_resolution:deferred"
            active_assigned = True
        elif not active_assigned:
            next_item["status"] = "in_progress"
            next_item["status_reason"] = "current_obligation"
            active_assigned = True
        else:
            next_item["status"] = "pending"
            next_item.pop("status_reason", None)
        reconciled.append(next_item)
    if (
        obligation_kind
        and obligation_kind != "complete"
        and not any(_item_matches_obligation(item, current_obligation) for item in reconciled)
    ):
        reconciled.append(_minimal_queue_item(current_obligation))
    return reconciled


def advance_execution_queue_for_patch(
    state: AgentState | dict[str, Any],
    changed_files: set[str],
) -> list[dict[str, Any]]:
    queue = _copy_execution_queue((state or {}).get("execution_queue", []))
    if not changed_files:
        return queue
    for item in queue:
        if item.get("kind") == "verify" and item.get("status") == "completed":
            item["status"] = "pending"
            item["status_reason"] = "verification_stale_after_patch"
            continue
        if item.get("status") not in {"pending", "in_progress", "blocked"}:
            continue
        if item.get("kind") != "patch":
            continue
        targets = set(item.get("target_files", []) or [])
        if (targets and targets.issubset(changed_files)) or (
            not targets and item.get("status") == "in_progress"
        ):
            item["status"] = "completed"
            item["status_reason"] = "patch_applied"
    return queue


def advance_execution_queue_for_verification(
    state: AgentState | dict[str, Any],
) -> list[dict[str, Any]]:
    queue = _copy_execution_queue((state or {}).get("execution_queue", []))
    for item in queue:
        if item.get("status") not in {"pending", "in_progress", "blocked"}:
            continue
        if item.get("kind") == "verify" and _item_matches_obligation(
            item,
            (state or {}).get("next_obligation", {}),
        ):
            item["status"] = "completed"
            item["status_reason"] = "verification_passed"
            break
    return queue


def _item_matches_obligation(item: dict[str, Any], obligation: dict[str, Any]) -> bool:
    """ 判断 item 是否匹配符合当前 obligation"""
    if not obligation or str(obligation.get("kind") or "") == "complete":
        return False
    # 从 id 维度，去评估与下一个 obligation 匹配的 queue item
    criterion_id = str(obligation.get("criterion_id") or "")
    item_criterion = str(item.get("criterion_id") or "")
    return bool(criterion_id) and criterion_id == item_criterion


def _minimal_queue_item(obligation: dict[str, Any]) -> dict[str, Any]:
    criterion_id = str(obligation.get("criterion_id") or "unscoped")
    capability = str(obligation.get("required_capability") or "")
    return {
        "id": f"{criterion_id}:work",
        "criterion_id": str(obligation.get("criterion_id") or ""),
        "kind": KIND_BY_CAPABILITY.get(capability, str(obligation.get("kind") or "work")),
        "required_capability": capability,
        "required": True,
        "target_files": [],
        "commands": [],
        "status": "in_progress",
        "status_reason": "current_obligation:minimal_item",
    }


def _copy_execution_queue(queue: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "target_files": list(item.get("target_files", []) or []),
            "commands": list(item.get("commands", []) or []),
        }
        for item in queue or []
    ]
