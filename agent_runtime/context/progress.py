"""Task-local progress based on evidence and coverage, independent of narration."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from loguru import logger
from agent_runtime.memory.file_cache import merge_ranges


def stable_material(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def evidence_value(value: Any) -> Any:
    if isinstance(value, dict):
        ignored = {"created_at", "updated_at", "timestamp", "loop_count", "elapsed_ms", "duration_ms",
                   "access_seq", "source", "session_cache_reused", "reason", "confidence", "max_chars",
                   "cache_hit", "from_cache"}
        return {key: evidence_value(item) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [evidence_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\b\d+(?:\.\d+)?(?:ms|s)\b", "<duration>", value)
    return value


def tool_evidence_key(call: dict[str, Any]) -> str:
    output = call.get("output") or {}
    if call.get("name") == "read_file":
        material = ["read_file", *(output.get(key) for key in (
            "file_path", "file_revision", "start_line", "end_line", "content"))]
    else:
        material = evidence_value({"name": call.get("name"), "input": call.get("input"), "output": output})
    return stable_material(material)


def register_progress(state: dict[str, Any], keys: list[str], *, source: str) -> bool:
    tracker = dict(state.get("analysis_progress") or {})
    seen = set(tracker.get("material_seen", []))
    fresh = set(keys) - seen
    if fresh:
        tracker.update(material_seen=sorted(seen | fresh), no_progress=0,
                       material_revision=int(tracker.get("material_revision", 0)) + 1)
        state["analysis_progress"] = tracker
    logger.bind(task_id=state.get("task_id")).debug(
        "material progress source={} new_keys={} progressed={}", source, len(fresh), bool(fresh))
    return bool(fresh)


def accept_analysis_records(state: dict[str, Any], records: list[dict[str, Any]]) -> None:
    # Coverage is a union: splitting/reordering ranges cannot manufacture progress.
    from_records = state.get("analysis_records", [])
    tracker = dict(state.get("analysis_progress") or {})
    coverage = dict(tracker.get("coverage", {}))
    fresh = []
    for batch, credit in ((from_records, False), (records, True)):
        for record in batch:
            if record.get("status") == "stale":
                continue
            base = [record["file_path"], record["file_revision"], record["dimension"].strip().lower()]
            stages = ["covered"] + (["complete"] if record.get("status") == "complete" else [])
            for stage in stages:
                key = stable_material([*base, stage])
                ranges = coverage.get(key, [])
                start, end = record["start_line"], record["end_line"]
                if not any(left <= start and right >= end for left, right in ranges):
                    merged = merge_ranges([*ranges, [start, end]])
                    if merged != ranges and credit:
                        fresh.append(stable_material([key, merged]))
                    coverage[key] = merged
    tracker["coverage"] = coverage
    state["analysis_progress"] = tracker
    state["analysis_records"] = records
    register_progress(state, fresh, source="analysis_coverage")
