"""Evaluate whether tool evidence satisfies the active goal criterion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ext.file_requirements import full_read_requirements, is_full_read
from model.agent.actions import Action
from model.agent.graph import AgentState


@dataclass
class EvidenceDecision:
    evidence: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    passed: bool = False
    failed: bool = False


def evaluate_evidence(
    state: AgentState,
    criterion: dict[str, Any],
    entry: dict[str, Any],
    action: Action,
    output: dict[str, Any],
) -> EvidenceDecision | None:
    policy = str(criterion.get("evidence_policy") or "")
    if policy == "repository_evidence":
        return _evaluate_repository(state, entry, action, output)
    if policy == "diagnosis_evidence":
        return _evaluate_diagnosis(state, entry, action, output)
    if policy == "command_evidence":
        return _evaluate_command(state, criterion, action, output)
    if policy == "patch_applied":
        return _evaluate_patch(state, criterion, entry, action, output)
    if policy == "verification_passed":
        return _evaluate_verification(state, criterion, action, output)
    return None


def _evaluate_repository(
    state: AgentState,
    entry: dict[str, Any],
    action: Action,
    output: dict[str, Any],
) -> EvidenceDecision | None:
    evidence = _repository_evidence(state, action, output)
    if evidence is None:
        return None
    coverage = _repository_coverage(state, entry, evidence)
    required = set(coverage["required_files"])
    covered = set(coverage["fully_read_files"])
    blockers = [f"full_read:{path}" for path in sorted(required - covered)]
    if not required and evidence.get("tool") != "read_file":
        blockers.append("repository_scope_not_read")
    passed = bool(required) and required.issubset(covered)
    return EvidenceDecision(
        evidence=[evidence],
        coverage=coverage,
        blockers=blockers,
        passed=passed,
    )


def _evaluate_diagnosis(
    state: AgentState,
    entry: dict[str, Any],
    action: Action,
    output: dict[str, Any],
) -> EvidenceDecision | None:
    repository = _repository_evidence(state, action, output)
    findings = _diagnosis_findings(action, output)
    if repository is None and not findings:
        return None

    evidence = ([repository] if repository is not None else []) + findings
    coverage = _repository_coverage(state, entry, repository)
    required = set(coverage["required_files"])
    covered = set(coverage["fully_read_files"])
    prior = list(entry.get("evidence", []) or [])
    has_symptom = any(_is_diagnosis_finding(item) for item in [*prior, *findings])
    source_ready = bool(covered) and (not required or required.issubset(covered))
    blockers: list[str] = []
    if not has_symptom:
        blockers.append("diagnostic_failure_evidence_missing")
    blockers.extend(f"full_read:{path}" for path in sorted(required - covered))
    if not source_ready and not any(item.startswith("full_read:") for item in blockers):
        blockers.append("diagnostic_source_evidence_missing")
    return EvidenceDecision(
        evidence=evidence,
        coverage=coverage,
        blockers=blockers,
        passed=has_symptom and source_ready,
    )


def _evaluate_patch(
    state: AgentState,
    criterion: dict[str, Any],
    entry: dict[str, Any],
    action: Action,
    output: dict[str, Any],
) -> EvidenceDecision | None:
    if action.name != "apply_code_patch":
        return None
    changed_files = _clean_paths(output.get("changed_files"))
    applied = bool(output.get("applied")) and not bool(
        output.get("dry_run") or action.args.get("dry_run")
    )
    if not applied and not output.get("error"):
        return None

    criterion_id = str(criterion.get("id") or "")
    linked = _linked_queue_items(state, criterion_id, kind="patch")
    previous = entry.get("coverage") if isinstance(entry.get("coverage"), dict) else {}
    required = set(_clean_paths(previous.get("required_files")))
    covered = set(_clean_paths(previous.get("changed_files")))
    for item in linked:
        required.update(_clean_paths(item.get("target_files")))
    covered.update(changed_files)
    queue_ready = not linked or all(item.get("status") == "completed" for item in linked)
    blockers: list[str] = []
    if not applied:
        blockers.append("patch_not_applied")
    if not changed_files:
        blockers.append("changed_files_missing")
    blockers.extend(f"patch_target:{path}" for path in sorted(required - covered))
    if not queue_ready:
        blockers.append("patch_queue_incomplete")
    evidence = {
        "type": "patch_applied" if applied else "patch_failed",
        "files": changed_files,
        "dry_run": bool(output.get("dry_run") or action.args.get("dry_run")),
        "summary": str(output.get("message") or output.get("summary") or output.get("error") or "")[:500],
    }
    return EvidenceDecision(
        evidence=[evidence],
        coverage={
            "required_files": sorted(required),
            "changed_files": sorted(covered),
        },
        blockers=blockers,
        passed=applied and bool(changed_files) and not blockers,
        failed=bool(output.get("error")) and not applied,
    )


def _evaluate_verification(
    state: AgentState,
    criterion: dict[str, Any],
    action: Action,
    output: dict[str, Any],
) -> EvidenceDecision | None:
    if not _is_verification_action(action):
        return None
    exit_code = _exit_code(output)
    if exit_code is None:
        return None

    guard = action.args.get("verification_guard")
    guard_allowed = isinstance(guard, dict) and bool(guard.get("allowed"))
    required_scope = str(criterion.get("verification_scope") or "repo")
    actual_scope = str(guard.get("scope") or "") if isinstance(guard, dict) else ""
    scope_sufficient = _scope_satisfies(actual_scope, required_scope)
    command = str(output.get("command") or action.args.get("command") or "").strip()
    last_edit = _state_int(state.get("last_edit_at_loop"), -1)
    last_verified = _state_int(state.get("last_verified_edit_loop"), -1)
    after_latest_edit = last_edit < 0 or last_verified >= last_edit
    linked = _linked_queue_items(state, str(criterion.get("id") or ""), kind="verify")
    queue_ready = not linked or all(item.get("status") == "completed" for item in linked)
    blockers: list[str] = []
    if not guard_allowed:
        blockers.append("verification_command_not_guarded")
    if not command:
        blockers.append("verification_command_missing")
    if not scope_sufficient:
        blockers.append(f"verification_scope:{actual_scope or 'unknown'}<{required_scope}")
    if exit_code != 0:
        blockers.append(f"verification_exit_code:{exit_code}")
    if not after_latest_edit or bool(state.get("verification_stale", False)):
        blockers.append("verification_stale")
    if not queue_ready:
        blockers.append("verification_queue_incomplete")
    evidence = {
        "type": "verification_result",
        "command": command,
        "exit_code": exit_code,
        "guard": guard if isinstance(guard, dict) else {},
        "required_scope": required_scope,
        "verified_after_loop": last_verified,
        "stdout": str(output.get("stdout") or "")[-1600:],
        "stderr": str(output.get("stderr") or output.get("error") or "")[-1600:],
    }
    return EvidenceDecision(
        evidence=[evidence],
        blockers=blockers,
        passed=not blockers,
        failed=exit_code != 0,
    )


def _evaluate_command(
    state: AgentState,
    criterion: dict[str, Any],
    action: Action,
    output: dict[str, Any],
) -> EvidenceDecision | None:
    del state
    if not _is_verification_action(action):
        return None
    exit_code = _exit_code(output)
    if exit_code is None:
        return None
    guard = action.args.get("verification_guard")
    guard_allowed = isinstance(guard, dict) and bool(guard.get("allowed"))
    required_scope = str(criterion.get("verification_scope") or "repo")
    actual_scope = str(guard.get("scope") or "") if isinstance(guard, dict) else ""
    command = str(output.get("command") or action.args.get("command") or "").strip()
    blockers: list[str] = []
    if not guard_allowed:
        blockers.append("command_not_guarded")
    if not command:
        blockers.append("command_missing")
    if not _scope_satisfies(actual_scope, required_scope):
        blockers.append(f"command_scope:{actual_scope or 'unknown'}<{required_scope}")
    evidence = {
        "type": "command_result",
        "command": command,
        "exit_code": exit_code,
        "guard": guard if isinstance(guard, dict) else {},
        "required_scope": required_scope,
        "stdout": str(output.get("stdout") or "")[-1600:],
        "stderr": str(output.get("stderr") or output.get("error") or "")[-1600:],
    }
    return EvidenceDecision(
        evidence=[evidence],
        blockers=blockers,
        passed=not blockers,
    )


def _repository_evidence(
    state: AgentState,
    action: Action,
    output: dict[str, Any],
) -> dict[str, Any] | None:
    if output.get("error") or output.get("skipped"):
        return None
    if action.name == "read_file":
        file_path = str(output.get("file_path") or action.args.get("file_path") or "").strip()
        if not file_path:
            return None
        return {
            "type": "repository_evidence",
            "tool": action.name,
            "file_path": file_path,
            "full_read": is_full_read(state, file_path),
            "summary": str(output.get("summary") or output.get("excerpt") or "File inspected.")[:500],
        }
    if action.name == "search_text" and output.get("matches"):
        return {
            "type": "repository_evidence",
            "tool": action.name,
            "match_count": len(output.get("matches", []) or []),
            "summary": str(output.get("summary") or "Text matches collected.")[:500],
        }
    if action.name == "search_code_context" and (
        output.get("candidate_files")
        or output.get("files")
        or output.get("results")
        or output.get("items")
    ):
        return {
            "type": "repository_evidence",
            "tool": action.name,
            "files": _context_paths(output),
            "summary": str(output.get("summary") or "Code context collected.")[:500],
        }
    return None


def _repository_coverage(
    state: AgentState,
    entry: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    previous = entry.get("coverage") if isinstance(entry.get("coverage"), dict) else {}
    required = set(_clean_paths(previous.get("required_files")))
    covered = set(_clean_paths(previous.get("fully_read_files")))
    required.update(
        str(item.get("file_path") or "").strip()
        for item in full_read_requirements(state, limit=12)
        if str(item.get("file_path") or "").strip()
    )
    if evidence and evidence.get("tool") == "read_file":
        path = str(evidence.get("file_path") or "").strip()
        if path:
            required.add(path)
            if bool(evidence.get("full_read")):
                covered.add(path)
    return {
        "required_files": sorted(required),
        "fully_read_files": sorted(covered),
    }


def _diagnosis_findings(action: Action, output: dict[str, Any]) -> list[dict[str, Any]]:
    if action.name not in {"run_shell_command", "run_tests"}:
        return []
    exit_code = _exit_code(output)
    if exit_code in (None, 0):
        return []
    text = "\n".join(str(output.get(key) or "") for key in ("stdout", "stderr", "error"))
    findings: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.search(r"([A-Za-z0-9_./\\-]+):(\d+):\s*(.+)", stripped)
        if match:
            findings.append(
                {
                    "type": "diagnostic_failure",
                    "file_path": match.group(1).replace("\\", "/"),
                    "line": int(match.group(2)),
                    "message": match.group(3)[:500],
                }
            )
        elif stripped.startswith("--- FAIL:") or stripped.startswith("panic:"):
            findings.append({"type": "diagnostic_failure", "message": stripped[:500]})
    if not findings and text.strip():
        findings.append({"type": "diagnostic_failure", "message": text[-800:]})
    return findings[:8]


def _is_diagnosis_finding(evidence: Any) -> bool:
    return isinstance(evidence, dict) and evidence.get("type") == "diagnostic_failure"


def _linked_queue_items(
    state: AgentState,
    criterion_id: str,
    *,
    kind: str,
) -> list[dict[str, Any]]:
    if not criterion_id:
        return []
    return [
        item
        for item in state.get("execution_queue", []) or []
        if isinstance(item, dict)
        and str(item.get("criterion_id") or "") == criterion_id
        and str(item.get("kind") or "") == kind
        and bool(item.get("required", True))
    ]


def _is_verification_action(action: Action) -> bool:
    return action.name == "run_tests" or (
        action.name == "run_shell_command"
        and str(action.args.get("purpose") or "").strip().lower() == "verification"
    )


def _exit_code(output: dict[str, Any]) -> int | None:
    try:
        return int(output.get("exit_code"))
    except (TypeError, ValueError):
        return None


def _clean_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    paths: list[str] = []
    for item in value:
        path = str(item or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _context_paths(output: dict[str, Any]) -> list[str]:
    paths = _clean_paths(output.get("candidate_files"))
    for item in output.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("file_path") or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _state_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scope_satisfies(actual: str, required: str) -> bool:
    rank = {"test": 1, "package": 2, "repo": 3}
    return rank.get(actual, 0) >= rank.get(required, 3)
