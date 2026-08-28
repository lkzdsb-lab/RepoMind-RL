from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from evaluation.models import (
    EvaluationCase,
    EvaluationResult,
    FindingMatch,
    VerificationResult,
)


def score_evaluation(
    *,
    run_id: str,
    case: EvaluationCase,
    raw_result: dict[str, Any],
    changed: list[str],
    verification: VerificationResult,
    fixture_hash: str,
    config_fingerprint: str,
    artifacts: dict[str, str],
    worker_timed_out: bool,
    worker_exit_code: int | None,
) -> EvaluationResult:
    turns = [item for item in raw_result.get("turns", []) if isinstance(item, dict)]
    findings = _collect_findings(turns)
    matches = _match_findings(case.expected_findings, findings)
    found = sum(1 for item in matches if item.matched)
    total = len(matches)
    recall = found / total if total else 1.0
    latest_task_turns = _latest_turn_per_task(turns)
    statuses = [str(item.get("status") or "") for item in latest_task_turns]
    agent_status = str(turns[-1].get("status") or "") if turns else "not_started"
    loops = _sum_task_max(turns, lambda item: item.get("loop_count"))
    tool_calls = _sum_task_max(turns, lambda item: item.get("tool_call_count"))
    llm_calls = _sum_task_max(turns, lambda item: item.get("llm_call_count"))
    total_tokens = _sum_task_max(
        turns,
        lambda item: (item.get("llm_token_usage") or {}).get("total_tokens")
        if isinstance(item.get("llm_token_usage"), dict)
        else 0,
    )
    unauthorized = _unauthorized_changes(case, changed)
    failures: list[str] = []
    if worker_timed_out:
        failures.append("worker timed out")
    if worker_exit_code not in (0, None):
        failures.append(f"worker exited with code {worker_exit_code}")
    if raw_result.get("worker_status") != "completed":
        failures.append(str(raw_result.get("worker_error") or "worker did not complete"))
    if len(turns) != len(case.turns):
        failures.append(f"completed {len(turns)} of {len(case.turns)} conversation turns")
    if any(status != "finished" for status in statuses):
        failures.append(f"non-finished agent status: {agent_status}")
    if not verification.passed:
        failures.append("independent verification failed")
    if unauthorized:
        failures.append(f"unauthorized changes: {', '.join(unauthorized)}")
    if found < total:
        missing = [item.bug_id for item in matches if not item.matched]
        failures.append(f"missing expected findings: {', '.join(missing)}")

    completion_score = 20 if turns and all(item == "finished" for item in statuses) else 0
    correctness_score = round(25 * int(verification.passed) + 25 * recall)
    scope_score = 20 if not unauthorized else 0
    efficiency_score = _efficiency_score(case, loops, tool_calls, total_tokens)
    score = max(0, min(100, completion_score + correctness_score + scope_score + efficiency_score))
    passed = not failures
    return EvaluationResult(
        run_id=run_id,
        case_id=case.case_id,
        case_version=case.version,
        passed=passed,
        score=score,
        agent_status=agent_status,
        fixture_hash=fixture_hash,
        config_fingerprint=config_fingerprint,
        findings_found=found,
        findings_total=total,
        finding_matches=matches,
        changed_files=changed,
        unauthorized_changes=unauthorized,
        verification=verification,
        loops=loops,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        total_tokens=total_tokens,
        failures=failures,
        metrics={
            "finding_recall": recall,
            "completion_score": completion_score,
            "correctness_score": correctness_score,
            "scope_score": scope_score,
            "efficiency_score": efficiency_score,
        },
        artifacts=artifacts,
    )


def _collect_findings(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for turn in turns:
        for item in turn.get("draft_findings", []) or []:
            if isinstance(item, dict):
                findings.append({**item, "_source": f"turn:{turn.get('turn_index')}:draft"})
        report = turn.get("final_report")
        if not isinstance(report, dict):
            continue
        for item in report.get("findings", []) or []:
            if isinstance(item, dict):
                findings.append({**item, "_source": f"turn:{turn.get('turn_index')}:final"})
            elif str(item).strip():
                findings.append(
                    {
                        "claim": str(item).strip(),
                        "locations": [],
                        "_source": f"turn:{turn.get('turn_index')}:final",
                    }
                )
    return findings


def _match_findings(
    expected_findings: list[Any],
    findings: list[dict[str, Any]],
) -> list[FindingMatch]:
    used: set[int] = set()
    matches: list[FindingMatch] = []
    for expected in expected_findings:
        match = FindingMatch(bug_id=expected.bug_id, matched=False)
        for index, finding in enumerate(findings):
            if index in used or not _finding_matches(expected, finding):
                continue
            used.add(index)
            match = FindingMatch(
                bug_id=expected.bug_id,
                matched=True,
                source=str(finding.get("_source") or ""),
            )
            break
        matches.append(match)
    return matches


def _finding_matches(expected: Any, finding: dict[str, Any]) -> bool:
    claim = str(finding.get("claim") or finding.get("summary") or "").lower()
    locations = [
        item
        for item in finding.get("locations", []) or []
        if isinstance(item, dict)
    ]
    paths = {_normalized_path(item.get("file_path")) for item in locations}
    symbols = {str(item.get("symbol") or "").lower() for item in locations}
    expected_path = _normalized_path(expected.file)
    file_matches = expected_path in paths or Path(expected_path).name.lower() in claim
    symbol_matches = (
        not expected.symbol
        or expected.symbol.lower() in symbols
        or expected.symbol.lower() in claim
    )
    category_matches = (
        not expected.category
        or str(finding.get("category") or "").lower() == expected.category.lower()
    )
    keyword_matches = not expected.keywords or any(
        word.lower() in claim for word in expected.keywords
    )
    return file_matches and symbol_matches and category_matches and keyword_matches


def _unauthorized_changes(case: EvaluationCase, changed: list[str]) -> list[str]:
    if case.intent != "implement":
        return sorted(changed)
    result: list[str] = []
    for path in changed:
        if _matches_any(path, case.forbidden_changed_files):
            result.append(path)
            continue
        if not _matches_any(path, case.allowed_changed_files):
            result.append(path)
    return sorted(set(result))


def _matches_any(path: str, patterns: list[str]) -> bool:
    normalized = _normalized_path(path)
    return any(fnmatch.fnmatch(normalized, _normalized_path(pattern)) for pattern in patterns)


def _efficiency_score(case: EvaluationCase, loops: int, tool_calls: int, tokens: int) -> int:
    limits = case.limits
    checks = (
        loops <= limits.max_loops * len(case.turns),
        tool_calls <= limits.max_tool_calls,
        tokens <= limits.max_total_tokens,
    )
    return round(10 * sum(int(item) for item in checks) / len(checks))


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip("./").lower()


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _latest_turn_per_task(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, turn in enumerate(turns):
        key = str(turn.get("task_id") or f"turn:{index}")
        if key not in latest:
            order.append(key)
        latest[key] = turn
    return [latest[key] for key in order]


def _sum_task_max(turns: list[dict[str, Any]], getter: Any) -> int:
    maxima: dict[str, int] = {}
    for index, turn in enumerate(turns):
        key = str(turn.get("task_id") or f"turn:{index}")
        maxima[key] = max(maxima.get(key, 0), _integer(getter(turn)))
    return sum(maxima.values())
