"""Select the next ready criterion from a task goal contract."""

from __future__ import annotations

from typing import Any


def derive_next_obligation(contract: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    criteria = contract.get("criteria") if isinstance(contract.get("criteria"), list) else []
    progress = ledger.get("criteria") if isinstance(ledger.get("criteria"), dict) else {}
    pending = [item for item in criteria if _is_pending_required(item, progress)]
    if not pending:
        return {
            "kind": "complete",
            "criterion_id": "",
            "reason": "All required goal criteria have evidence.",
            "required_capability": "",
        }

    for criterion in pending:
        dependencies = [str(item) for item in criterion.get("depends_on", []) or []]
        if all(_criterion_passed(progress, dependency) for dependency in dependencies):
            return _as_obligation(criterion, progress)

    blocked = pending[0]
    unmet = [
        str(item)
        for item in blocked.get("depends_on", []) or []
        if not _criterion_passed(progress, str(item))
    ]
    return {
        **_as_obligation(blocked, progress),
        "blocked_by": unmet,
        "reason": f"Criterion is waiting for dependencies: {', '.join(unmet)}.",
    }


def required_action_for_obligation(obligation: dict[str, Any]) -> str:
    capability = str(obligation.get("required_capability") or "")
    if capability == "read_code":
        return "read_file"
    if capability == "patch":
        return "EnterPlanMode"
    if capability == "verification":
        return "run_shell_command"
    if str(obligation.get("kind") or "") == "complete":
        return "finish"
    return ""


def _is_pending_required(criterion: Any, progress: dict[str, Any]) -> bool:
    if not isinstance(criterion, dict) or not bool(criterion.get("required", True)):
        return False
    return not _criterion_passed(progress, str(criterion.get("id") or ""))


def _criterion_passed(progress: dict[str, Any], criterion_id: str) -> bool:
    entry = progress.get(criterion_id)
    return (
        isinstance(entry, dict)
        and entry.get("status") in {"passed", "not_required"}
        and not list(entry.get("blockers", []) or [])
    )


def _as_obligation(
    criterion: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    criterion_id = str(criterion.get("id") or "")
    entry = progress.get(criterion_id)
    evidence_blockers = (
        list(entry.get("blockers", []) or [])
        if isinstance(entry, dict)
        else []
    )
    return {
        "kind": str(criterion.get("kind") or "diagnose"),
        "criterion_id": criterion_id,
        "description": str(criterion.get("description") or ""),
        "evidence_policy": str(criterion.get("evidence_policy") or ""),
        "reason": f"Required criterion `{criterion.get('id')}` has no passing evidence.",
        "required_capability": str(criterion.get("required_capability") or ""),
        "evidence_blockers": evidence_blockers,
    }
