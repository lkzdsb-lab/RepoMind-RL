"""Track criterion evidence without delegating completion authority to the LLM."""

from __future__ import annotations

from typing import Any

from agent_runtime.lifecycle.evidence import EvidenceDecision, evaluate_evidence
from model.agent.actions import Action
from model.agent.graph import AgentState


def initial_progress_ledger(contract: dict[str, Any]) -> dict[str, Any]:
    criteria = contract.get("criteria") if isinstance(contract.get("criteria"), list) else []
    return {
        "criteria": {
            str(item.get("id")): {
                "status": "pending" if bool(item.get("required", True)) else "optional",
                "kind": str(item.get("kind") or ""),
                "evidence_policy": str(item.get("evidence_policy") or ""),
                "evidence": [],
                "coverage": {},
                "blockers": [],
                "last_updated_loop": -1,
            }
            for item in criteria
            if isinstance(item, dict) and str(item.get("id") or "")
        }
    }


def update_progress_ledger(
    state: AgentState,
    action: Action,
    output: dict[str, Any],
) -> dict[str, Any]:
    contract = state.get("goal_contract") if isinstance(state.get("goal_contract"), dict) else {}
    ledger = _copy_ledger(state.get("progress_ledger"), contract)
    active = _active_criterion(state, contract, ledger)
    if active is not None:
        criterion_id, criterion, entry = active
        decision = evaluate_evidence(state, criterion, entry, action, output)
        if decision is not None:
            _apply_decision(
                entry,
                decision,
                loop_count=int(state.get("loop_count", 0) or 0),
            )
            ledger["criteria"][criterion_id] = entry

    if action.name == "apply_code_patch" and output.get("applied"):
        _reset_policy(ledger, "verification_passed", blocker="verification_stale_after_patch")
    return ledger


def _copy_ledger(value: Any, contract: dict[str, Any]) -> dict[str, Any]:
    base = initial_progress_ledger(contract)
    if not isinstance(value, dict) or not isinstance(value.get("criteria"), dict):
        return base
    existing = value["criteria"]
    for criterion_id, entry in base["criteria"].items():
        previous = existing.get(criterion_id)
        if not isinstance(previous, dict):
            continue
        entry.update(previous)
        entry["evidence"] = list(previous.get("evidence", []) or [])[-20:]
        entry["coverage"] = (
            dict(previous.get("coverage"))
            if isinstance(previous.get("coverage"), dict)
            else {}
        )
        entry["blockers"] = [
            str(item)
            for item in previous.get("blockers", []) or []
            if str(item)
        ][:20]
    return base


def _active_criterion(
    state: AgentState,
    contract: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    obligation = state.get("next_obligation")
    if not isinstance(obligation, dict):
        return None
    criterion_id = str(obligation.get("criterion_id") or "").strip()
    obligation_policy = str(obligation.get("evidence_policy") or "").strip()
    if not criterion_id or not obligation_policy:
        return None
    criterion = next(
        (
            item
            for item in contract.get("criteria", []) or []
            if isinstance(item, dict) and str(item.get("id") or "") == criterion_id
        ),
        None,
    )
    entry = ledger.get("criteria", {}).get(criterion_id)
    if not isinstance(criterion, dict) or not isinstance(entry, dict):
        return None
    if str(criterion.get("evidence_policy") or "") != obligation_policy:
        return None
    if str(entry.get("evidence_policy") or "") != obligation_policy:
        return None
    return criterion_id, criterion, entry


def _apply_decision(
    entry: dict[str, Any],
    decision: EvidenceDecision,
    *,
    loop_count: int,
) -> None:
    for evidence in decision.evidence:
        _append_evidence(entry, evidence)
    if decision.coverage:
        entry["coverage"] = dict(decision.coverage)
    entry["blockers"] = list(dict.fromkeys(decision.blockers))[:20]
    entry["last_updated_loop"] = loop_count
    if decision.passed and not entry["blockers"]:
        entry["status"] = "passed"
    elif decision.failed:
        entry["status"] = "failed"
    elif entry.get("status") != "optional":
        entry["status"] = "pending"


def _reset_policy(
    ledger: dict[str, Any],
    policy: str,
    *,
    blocker: str,
) -> None:
    for entry in ledger.get("criteria", {}).values():
        if not isinstance(entry, dict) or entry.get("evidence_policy") != policy:
            continue
        entry["status"] = "pending" if entry.get("status") != "optional" else "optional"
        entry["blockers"] = [blocker]


def _append_evidence(entry: dict[str, Any], evidence: dict[str, Any]) -> None:
    items = list(entry.get("evidence", []) or [])
    if all(str(item) != str(evidence) for item in items):
        items.append(evidence)
    entry["evidence"] = items[-20:]
