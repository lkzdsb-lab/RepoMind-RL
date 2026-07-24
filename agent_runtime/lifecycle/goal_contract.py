"""Build and validate data-driven task completion contracts."""

from __future__ import annotations

from typing import Any

from loguru import logger
from model.agent.graph import AgentState


CAPABILITY_BY_KIND = {
    "diagnose": "read_code",
    "implement": "patch",
    "verify": "verification",
}
CAPABILITY_BY_POLICY = {
    "command_evidence": "verification",
}


def build_goal_contract(state: AgentState) -> dict[str, Any]:
    analysis = state.get("task_analysis") if isinstance(state.get("task_analysis"), dict) else {}
    raw_criteria = analysis.get("completion_criteria")
    criteria, contract_valid = _validate_criteria(
        raw_criteria,
        task_id=str(state.get("task_id") or ""),
    )
    source = "llm"
    if not contract_valid:
        criteria = _fallback_criteria(
            task_type=str(state.get("task_type") or "BUG_FIX").upper(),
            verification_required=bool(analysis.get("verification_required", True)),
        )
        source = "invalid_contract_fallback"
    elif not criteria:
        criteria = _fallback_criteria(
            task_type=str(state.get("task_type") or "BUG_FIX").upper(),
            verification_required=bool(analysis.get("verification_required", True)),
        )
        source = "runtime_fallback"

    objective = str(state.get("description") or state.get("title") or "Complete the requested task.").strip()
    acceptance = analysis.get("acceptance_criteria") if isinstance(analysis.get("acceptance_criteria"), list) else []
    required = [item for item in criteria if item["required"]]
    return {
        "version": 2,
        "objective": objective[:1000],
        "criteria": criteria,
        "acceptance_criteria": [str(item).strip()[:500] for item in acceptance if str(item).strip()][:12],
        "source": source,
        # Compatibility projections for policy/reporting code. Runtime decisions use criteria.
        "requires_diagnosis": any(item["kind"] == "diagnose" for item in required),
        "requires_code_change": any(item["kind"] == "implement" for item in required),
        "requires_verification": any(item["kind"] == "verify" for item in required),
    }


def goal_contract_satisfied(contract: dict[str, Any], ledger: dict[str, Any]) -> bool:
    criteria = contract.get("criteria") if isinstance(contract.get("criteria"), list) else []
    progress = ledger.get("criteria") if isinstance(ledger.get("criteria"), dict) else {}
    if not criteria:
        return False
    for criterion in criteria:
        if not isinstance(criterion, dict) or not bool(criterion.get("required", True)):
            continue
        entry = progress.get(str(criterion.get("id") or ""))
        if (
            not isinstance(entry, dict)
            or entry.get("status") not in {"passed", "not_required"}
            or bool(entry.get("blockers"))
        ):
            return False
    return True


def _validate_criteria(value: Any, *, task_id: str) -> tuple[list[dict[str, Any]], bool]:
    log = logger.bind(task_id=task_id, component="goal_contract")
    if value is None:
        return [], True
    if not isinstance(value, list):
        log.warning("completion criteria is not a list; using runtime fallback")
        return [], True
    criteria: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            log.warning(
                "ignoring non-object completion criterion value_type={}",
                type(item).__name__,
            )
            continue
        criteria.append(dict(item))
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in criteria:
        criterion_id = str(item.get("id") or "")
        kind = str(item.get("kind") or "")
        evidence_policy = str(item.get("evidence_policy") or "")
        if criterion_id in seen:
            log.warning(
                "ignoring duplicate completion criterion criterion_id={}",
                criterion_id,
            )
            continue
        if not _kind_policy_valid(kind, evidence_policy):
            log.error(
                "invalid criterion policy criterion_id={} kind={} evidence_policy={}; using fallback contract",
                criterion_id,
                kind,
                evidence_policy,
            )
            return [], False
        seen.add(criterion_id)
        item["required_capability"] = CAPABILITY_BY_POLICY.get(
            evidence_policy,
            CAPABILITY_BY_KIND[kind],
        )
        if evidence_policy in {"command_evidence", "verification_passed"}:
            item["verification_scope"] = str(item.get("verification_scope") or "repo")
        accepted.append(item)

    known = set(seen)
    for item in accepted:
        criterion_id = item["id"]
        dependencies: list[str] = []
        raw_dependencies = item.get("depends_on", [])
        if not isinstance(raw_dependencies, list):
            log.warning(
                "ignoring non-list dependencies criterion_id={}",
                criterion_id,
            )
            raw_dependencies = []
        for dependency in raw_dependencies:
            if dependency == criterion_id:
                log.warning("ignoring self dependency criterion_id={}", criterion_id)
                continue
            if dependency not in known:
                log.warning(
                    "ignoring unknown dependency criterion_id={} dependency={}",
                    criterion_id,
                    dependency,
                )
                continue
            if dependency not in dependencies:
                dependencies.append(dependency)
        item["depends_on"] = dependencies
    if _has_dependency_cycle(accepted):
        log.error("completion criteria contain a dependency cycle; using fallback contract")
        return [], False
    return accepted, True


def _kind_policy_valid(kind: str, policy: str) -> bool:
    allowed = {
        "diagnose": {"repository_evidence", "diagnosis_evidence", "command_evidence"},
        "implement": {"patch_applied"},
        "verify": {"verification_passed"},
    }
    return policy in allowed.get(kind, set())


def _has_dependency_cycle(criteria: list[dict[str, Any]]) -> bool:
    """ 判断 llm 的返回是否存在循环依赖"""
    graph = {item["id"]: list(item.get("depends_on", [])) for item in criteria}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(criterion_id: str) -> bool:
        if criterion_id in visiting:
            return True
        if criterion_id in visited:
            return False
        visiting.add(criterion_id)
        if any(visit(dependency) for dependency in graph.get(criterion_id, [])):
            return True
        visiting.remove(criterion_id)
        visited.add(criterion_id)
        return False

    return any(visit(criterion_id) for criterion_id in graph)


def _fallback_criteria(task_type: str, verification_required: bool) -> list[dict[str, Any]]:
    if task_type == "DIAGNOSE":
        specs = [
            ("investigation", "diagnose", "Inspect repository evidence needed to answer the task.", "repository_evidence", []),
        ]
    elif task_type == "FEATURE_IMPL":
        specs = [
            ("implementation", "implement", "Apply the requested implementation change.", "patch_applied", []),
        ]
    else:
        specs = [
            ("diagnosis", "diagnose", "Collect concrete evidence for the reported failure.", "diagnosis_evidence", []),
            ("implementation", "implement", "Apply the code change that addresses the diagnosis.", "patch_applied", ["diagnosis"]),
        ]
    if verification_required and task_type != "DIAGNOSE":
        dependency = "implementation"
        specs.append(
            ("verification", "verify", "Pass command-based verification for the completed change.", "verification_passed", [dependency])
        )
    return [
        {
            "id": criterion_id,
            "kind": kind,
            "description": description,
            "required": True,
            "depends_on": dependencies,
            "evidence_policy": policy,
            "required_capability": CAPABILITY_BY_KIND[kind],
        }
        for criterion_id, kind, description, policy, dependencies in specs
    ]
