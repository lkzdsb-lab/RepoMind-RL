"""Task-local, versioned analysis progress; reading alone is not analysis."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agent_runtime.memory.file_cache import range_is_cached, source_revision


def refresh_analysis_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    """ 将记录中文件被修改的 record 标记为被污染"""
    revisions = {}
    result = []
    for record in state.get("analysis_records", []):
        record = dict(record)
        path = record["file_path"]
        if path not in revisions:
            revisions[path] = source_revision(state, path)
        if revisions[path] != record["file_revision"]:
            record["status"] = "stale"
        result.append(record)
    return result


def merge_analysis_updates(
    state: dict[str, Any], updates: list[dict[str, Any]],
    shown: list[dict[str, Any]], findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """ 检查模型返回的更新结果，并生成稳定的 records 集合"""
    records = {item["record_id"]: item for item in refresh_analysis_records(state)}
    errors = []
    finding_ids = {item["candidate_id"] for item in findings}
    for update in updates:
        path, revision = update["file_path"], update["file_revision"]
        spans = [s for s in shown if s["file_path"] == path and s["file_revision"] == revision]
        snapshot = (state.get("read_file_cache") or {}).get(path, {})
        start, end = update["start_line"], update["end_line"]
        if (snapshot.get("file_revision") != revision or source_revision(state, path) != revision
                or not range_is_cached({"spans": spans}, start, end)):
            errors.append(f"{path}:{start}-{end}: analysis must cite current source actually shown in this decision")
            continue
        if not set(update.get("finding_ids", [])).issubset(finding_ids):
            errors.append(f"{path}: unknown finding_ids; use existing candidate IDs or submit the finding separately")
            continue
        if update["status"] == "complete" and update.get("open_questions"):
            errors.append(f"{path}: complete analysis cannot contain unresolved questions")
            continue
        identity = [path, revision, start, end, update["dimension"].strip().lower()]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
        records[key] = {**update, "record_id": key}
    return list(records.values()), errors


def analysis_fingerprint(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence_request_key(action: str, args: dict[str, Any]) -> str:
    if action not in {"read_file", "run_tests", "run_shell_command"}:
        return ""
    ignored = {"reason", "confidence", "max_chars"}
    return hashlib.sha256(json.dumps(
        [action, {key: value for key, value in args.items() if key not in ignored}],
        sort_keys=True, ensure_ascii=False, default=str,
    ).encode()).hexdigest()


def record_execution_progress(state: dict[str, Any], action: str, output: dict[str, Any]) -> dict[str, Any]:
    """Count repeated evidence across alternating actions, not just consecutive calls."""
    tracker = dict(state.get("analysis_progress") or {})
    revision = int((state.get("runtime_facts") or {}).get("edit_revision", 0))
    if tracker.get("edit_revision") != revision:
        tracker = {"edit_revision": revision, "seen": [], "no_progress": 0}
    if action == "read_file" and not output.get("error"):
        identity = [action, output.get("file_path"), output.get("file_revision"),
                    output.get("start_line"), output.get("end_line"), output.get("content")]
        tracker["read_revisions"] = {
            **tracker.get("read_revisions", {}), output.get("file_path"): output.get("file_revision"),
        }
    elif action in {"run_tests", "run_shell_command"}:
        identity = ["command", output.get("command"), revision, output.get("exit_code"),
                    _stable_command_text(output.get("stdout")), _stable_command_text(output.get("stderr")), output.get("error")]
    else:
        return {"analysis_progress": tracker}
    key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, default=str).encode()).hexdigest()
    seen = list(tracker.get("seen", []))
    tracker["no_progress"] = int(tracker.get("no_progress", 0)) + 1 if key in seen else 0
    if key not in seen:
        seen.append(key)
    tracker["seen"] = seen
    requests = list(tracker.get("requests", []))
    request = evidence_request_key(action, state.get("next_action_input") or {})
    if request and request not in requests:
        requests.append(request)
    tracker["requests"] = requests
    return {"analysis_progress": tracker}


def _stable_command_text(value: Any) -> str:
    # Timing noise alone (e.g. Go test 0.031s) is not new diagnostic evidence.
    return re.sub(r"\b\d+(?:\.\d+)?(?:ms|s)\b", "<duration>", str(value or ""))
