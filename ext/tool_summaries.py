"""Compact prompt-safe summaries for heterogeneous tool outputs."""

from __future__ import annotations

from typing import Any

from ext.focus_files import current_focus_files
from model.agent.graph import AgentState
from utils import _truncate_text, _put_if_present


INTERESTING_OUTPUT_FIELDS = (
    "error",
    "skipped",
    "exit_code",
    "command",
    "reason",
    "purpose",
    "duration_ms",
    "file_path",
    "applied",
    "entered",
    "exited",
    "approved",
    "changed_files",
    "changed_line_count",
    "needs_user_input",
    "questions",
    "line_count",
    "query",
    "queries",
    "selected_ids",
)

PREVIEW_FIELDS = ("summary", "content", "stdout", "stderr", "diff", "message")


def tool_call_summaries(
    state: AgentState,
    *,
    limit: int = 12,
    preview_chars: int = 800,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for call in state.get("tool_calls", [])[-limit:]:
        if not isinstance(call, dict):
            continue
        output = call.get("output")
        if not isinstance(output, dict):
            output = {}
        summary: dict[str, Any] = {}
        _put_if_present(summary, "name", call.get("name"))
        _put_if_present(summary, "input", call.get("input"))
        _put_if_present(summary, "error", call.get("error"))
        if output:
            summary["output_keys"] = sorted(str(key) for key in output.keys())
            output_fields = {
                key: output[key]
                for key in INTERESTING_OUTPUT_FIELDS
                if key in output
            }
            if output_fields:
                summary["output_fields"] = output_fields
            preview = _output_preview(output, preview_chars)
            if preview:
                summary["output_preview"] = preview
        if summary:
            summaries.append(summary)
    return summaries


def read_file_summaries(
    state: AgentState,
    *,
    limit: int = 8,
    excerpt_chars: int = 1800,
) -> list[dict[str, Any]]:
    cache = state.get("read_file_cache")
    order = state.get("read_file_order")
    if isinstance(cache, dict) and isinstance(order, list) and cache:
        focus_paths = [
            path for path in current_focus_files(state, limit=min(3, limit)) if path in cache
        ]
        recent_paths = [
            str(path).strip()
            for path in reversed(order)
            if str(path).strip() and str(path).strip() not in focus_paths
        ]
        selected_paths = focus_paths + recent_paths[: max(0, limit - len(focus_paths))]
        focus_set = set(focus_paths)
        summaries: list[dict[str, Any]] = []
        for file_path in selected_paths:
            snapshot = cache.get(file_path)
            if not isinstance(snapshot, dict):
                continue
            is_focus = file_path in focus_set
            summary: dict[str, Any] = {
                "file_path": str(snapshot.get("file_path") or file_path),
                "line_count": int(snapshot.get("total_lines") or 0),
                "is_empty": bool(snapshot.get("is_empty", False)),
                "file_revision": str(snapshot.get("file_revision") or "")[:16],
                "read_ranges": [
                    {
                        "start_line": item.get("start_line"),
                        "end_line": item.get("end_line"),
                    }
                    for item in snapshot.get("spans", [])[-8:]
                    if isinstance(item, dict)
                ],
                "dirty_ranges": snapshot.get("dirty_ranges", [])[:8],
            }
            if is_focus:
                summary["detail_level"] = "target"
                imports_excerpt = str(snapshot.get("imports_excerpt") or "")
                if imports_excerpt:
                    summary["imports_excerpt"] = _truncate_text(imports_excerpt, min(500, excerpt_chars))
                if snapshot.get("focus_ranges"):
                    summary["focus_ranges"] = snapshot.get("focus_ranges")
                excerpt = _detailed_file_excerpt(
                    snapshot,
                    excerpt_chars=excerpt_chars,
                    prefer_full_content=is_focus,
                )
                if excerpt:
                    summary["content_excerpt"] = excerpt
                elif summary["is_empty"]:
                    summary["content_excerpt"] = "<empty file>"
            else:
                summary["detail_level"] = "compact"
                note = _compact_file_note(snapshot)
                if note:
                    summary["short_note"] = note
            summaries.append(summary)
        if summaries:
            return summaries

    summaries: list[dict[str, Any]] = []
    for call in state.get("tool_calls", [])[-limit:]:
        if not isinstance(call, dict) or call.get("name") != "read_file":
            continue
        output = call.get("output")
        if not isinstance(output, dict):
            output = {}
        call_input = call.get("input")
        if not isinstance(call_input, dict):
            call_input = {}
        summary: dict[str, Any] = {}
        _put_if_present(summary, "file_path", output.get("file_path") or call_input.get("file_path"))
        if "line_count" in output:
            summary["line_count"] = output["line_count"]
        elif "total_lines" in output:
            summary["line_count"] = output["total_lines"]
        content = output.get("content")
        if content is not None:
            text = str(content)
            if text:
                summary["content_excerpt"] = _truncate_text(text, excerpt_chars)
            elif int(output.get("total_lines") or 0) == 0:
                summary["content_excerpt"] = "<empty file>"
        _put_if_present(summary, "error", call.get("error") or output.get("error"))
        if summary:
            summaries.append(summary)
    return summaries


def read_file_range_context(
    state: AgentState,
    *,
    file_limit: int = 4,
    ranges_per_file: int = 3,
    total_chars: int = 12000,
    context_lines: int = 8,
) -> list[dict[str, Any]]:
    """Build exact, bounded source ranges for action decisions and patch anchors."""
    cache = state.get("read_file_cache")
    if not isinstance(cache, dict) or not cache:
        return []

    paths = current_focus_files(state, limit=file_limit)
    order = state.get("read_file_order")
    if isinstance(order, list):
        for value in reversed(order):
            path = str(value or "").strip()
            if path and path in cache and path not in paths:
                paths.append(path)
            if len(paths) >= file_limit:
                break

    remaining = max(1000, int(total_chars))
    result: list[dict[str, Any]] = []
    for path in paths[:file_limit]:
        snapshot = cache.get(path)
        if not isinstance(snapshot, dict):
            continue
        ranges = _source_ranges_for_file(
            state,
            path,
            snapshot,
            limit=max(1, ranges_per_file),
            padding=max(0, context_lines),
        )
        excerpts: list[dict[str, Any]] = []
        for start, end, reason in ranges:
            content = _content_for_range(snapshot, start, end)
            if not content:
                continue
            if len(content) > remaining:
                if excerpts or result:
                    break
                content = content[:remaining]
            excerpts.append(
                {
                    "start_line": start,
                    "end_line": start + max(0, len(content.splitlines()) - 1),
                    "reason": reason,
                    "content": content,
                }
            )
            remaining -= len(content)
            if remaining <= 0:
                break
        if excerpts:
            result.append(
                {
                    "file_path": str(snapshot.get("file_path") or path),
                    "file_revision": str(snapshot.get("file_revision") or "")[:16],
                    "total_lines": int(snapshot.get("total_lines") or 0),
                    "session_cache_reused": bool(snapshot.get("session_cache_reused", False)),
                    "ranges": excerpts,
                }
            )
        elif snapshot.get("dirty_ranges"):
            result.append(
                {
                    "file_path": str(snapshot.get("file_path") or path),
                    "file_revision": str(snapshot.get("file_revision") or "")[:16],
                    "total_lines": int(snapshot.get("total_lines") or 0),
                    "dirty_ranges": snapshot.get("dirty_ranges", [])[:8],
                    "requires_refresh": True,
                    "ranges": [],
                }
            )
        if remaining <= 0:
            break
    return result


def validated_cache_summary(state: AgentState, *, limit: int = 20) -> list[dict[str, Any]]:
    """Expose cache coverage metadata without sending source text to the planner."""
    cache = state.get("read_file_cache")
    if not isinstance(cache, dict):
        return []
    order = state.get("read_file_order")
    paths = [str(path) for path in order or [] if str(path) in cache]
    for path in cache:
        if path not in paths:
            paths.append(path)
    summaries: list[dict[str, Any]] = []
    for path in paths[-limit:]:
        snapshot = cache.get(path)
        if not isinstance(snapshot, dict):
            continue
        spans = [item for item in snapshot.get("spans", []) if isinstance(item, dict)]
        summaries.append(
            {
                "file_path": path,
                "file_revision": str(snapshot.get("file_revision") or "")[:16],
                "total_lines": int(snapshot.get("total_lines") or 0),
                "covered_ranges": [
                    [int(item.get("start_line") or 1), int(item.get("end_line") or 1)]
                    for item in spans[-8:]
                ],
                "dirty_ranges": snapshot.get("dirty_ranges", [])[:8],
                "has_full_content": _has_full_span(snapshot),
                "session_cache_reused": bool(snapshot.get("session_cache_reused", False)),
            }
        )
    return summaries


def candidate_evidence_packets(
    state: AgentState,
    candidates: Any,
    *,
    total_chars: int = 16000,
) -> list[dict[str, Any]]:
    """Bind each policy candidate to exact current source and test evidence."""
    if not isinstance(candidates, list):
        return []
    cache = state.get("read_file_cache")
    if not isinstance(cache, dict):
        cache = {}
    remaining = max(2000, int(total_chars))
    packets: list[dict[str, Any]] = []
    for raw_candidate in candidates[:20]:
        if not isinstance(raw_candidate, dict):
            continue
        candidate_id = str(raw_candidate.get("candidate_id") or "").strip()
        claim = str(raw_candidate.get("claim") or "").strip()
        if not candidate_id or not claim:
            continue
        packet: dict[str, Any] = {
            "candidate_id": candidate_id,
            "claim": claim,
            "policy_confidence": raw_candidate.get("confidence", 0.5),
            "source_evidence": [],
            "test_source_evidence": [],
            "runtime_evidence": [],
            "missing_evidence": [],
        }
        for index, location in enumerate(raw_candidate.get("locations", []) or [], start=1):
            if not isinstance(location, dict):
                continue
            file_path = str(location.get("file_path") or "").strip()
            snapshot = cache.get(file_path)
            if not isinstance(snapshot, dict):
                packet["missing_evidence"].append(f"source cache missing: {file_path}")
                continue
            source_range = _candidate_location_range(snapshot, location, padding=10)
            if source_range is None:
                packet["missing_evidence"].append(f"source range unresolved: {file_path}")
                continue
            start, end = source_range
            content = _content_for_range(snapshot, start, end)
            if not content:
                packet["missing_evidence"].append(
                    f"source range not covered: {file_path}:{start}-{end}"
                )
                continue
            if len(content) > remaining:
                packet["missing_evidence"].append(
                    f"source evidence exceeds prompt budget: {file_path}:{start}-{end}"
                )
                continue
            evidence_id = f"{candidate_id}:source:{index}"
            packet["source_evidence"].append(
                {
                    "evidence_id": evidence_id,
                    "file_path": file_path,
                    "file_revision": str(snapshot.get("file_revision") or "")[:16],
                    "start_line": start,
                    "end_line": end,
                    "symbol": str(location.get("symbol") or "").strip(),
                    "content": content,
                }
            )
            remaining -= len(content)

        related_tests = [
            str(value).strip()
            for value in raw_candidate.get("related_tests", []) or []
            if str(value).strip()
        ][:8]
        for test_index, test_name in enumerate(related_tests, start=1):
            test_evidence = _test_source_evidence(cache, candidate_id, test_index, test_name)
            if test_evidence:
                content = str(test_evidence.get("content") or "")
                if len(content) <= remaining:
                    packet["test_source_evidence"].append(test_evidence)
                    remaining -= len(content)
                else:
                    packet["missing_evidence"].append(
                        f"test source exceeds prompt budget: {test_name}"
                    )
            else:
                packet["missing_evidence"].append(f"test source not found: {test_name}")

        for result_index, result in enumerate(state.get("test_results", [])[-5:], start=1):
            if not isinstance(result, dict):
                continue
            output = "\n".join(
                text for text in (str(result.get("stdout") or ""), str(result.get("stderr") or "")) if text
            )
            if related_tests and not any(test_name in output for test_name in related_tests):
                continue
            if not related_tests and not output:
                continue
            packet["runtime_evidence"].append(
                {
                    "evidence_id": f"{candidate_id}:runtime:{result_index}",
                    "command": str(result.get("command") or ""),
                    "exit_code": result.get("exit_code"),
                    "output": _truncate_text(output, 2500),
                }
            )
        packets.append(packet)
    return packets


def _candidate_location_range(
    snapshot: dict[str, Any],
    location: dict[str, Any],
    *,
    padding: int,
) -> tuple[int, int] | None:
    total_lines = max(1, int(snapshot.get("total_lines") or 1))
    normalized = _normalize_source_range(location, total_lines, padding)
    if normalized and _range_is_covered(snapshot, *normalized):
        return normalized
    symbol = str(location.get("symbol") or "").strip()
    if not symbol:
        return None
    line = _find_text_line(snapshot, symbol.split(".")[-1])
    if line is None:
        return None
    start = max(1, line - padding)
    end = min(total_lines, line + padding * 2)
    return (start, end) if _range_is_covered(snapshot, start, end) else None


def _test_source_evidence(
    cache: dict[str, Any],
    candidate_id: str,
    index: int,
    test_name: str,
) -> dict[str, Any] | None:
    for file_path, snapshot in cache.items():
        if not isinstance(snapshot, dict):
            continue
        if "test" not in str(file_path).lower():
            continue
        line = _find_text_line(snapshot, test_name)
        if line is None:
            continue
        total_lines = max(1, int(snapshot.get("total_lines") or 1))
        start = max(1, line - 3)
        end = min(total_lines, line + 18)
        content = _content_for_range(snapshot, start, end)
        if not content:
            continue
        return {
            "evidence_id": f"{candidate_id}:test_source:{index}",
            "file_path": str(file_path),
            "file_revision": str(snapshot.get("file_revision") or "")[:16],
            "start_line": start,
            "end_line": end,
            "test_name": test_name,
            "content": content,
        }
    return None


def _find_text_line(snapshot: dict[str, Any], needle: str) -> int | None:
    if not needle:
        return None
    for span in snapshot.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        start = int(span.get("start_line") or 1)
        for offset, line in enumerate(str(span.get("content") or "").splitlines()):
            if needle in line:
                return start + offset
    return None


def _source_ranges_for_file(
    state: AgentState,
    file_path: str,
    snapshot: dict[str, Any],
    *,
    limit: int,
    padding: int,
) -> list[tuple[int, int, str]]:
    total_lines = max(1, int(snapshot.get("total_lines") or 1))
    candidates: list[tuple[int, int, str]] = []

    for item in reversed(state.get("edit_results", []) or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("recovery_file") or "").strip() != file_path:
            continue
        suggested = item.get("suggested_range")
        normalized = _normalize_source_range(suggested, total_lines, padding)
        if normalized:
            candidates.append((*normalized, "patch_recovery"))
        break

    focus = state.get("attention_focus")
    if isinstance(focus, dict):
        focus_ranges = focus.get("focus_ranges")
        if isinstance(focus_ranges, dict):
            for item in focus_ranges.get(file_path, []) or []:
                normalized = _normalize_source_range(item, total_lines, padding)
                if normalized:
                    candidates.append((*normalized, "attention_focus"))

    for context in (state.get("selected_code_context"), state.get("code_context")):
        if not isinstance(context, dict):
            continue
        for key in ("functions", "symbols", "api_routes", "db_models", "call_graph"):
            for item in context.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("file_path") or "").strip() != file_path:
                    continue
                normalized = _normalize_source_range(item, total_lines, padding)
                if normalized:
                    candidates.append((*normalized, f"code_context:{key}"))

    for item in snapshot.get("focus_ranges", []) or []:
        normalized = _normalize_source_range(item, total_lines, padding)
        if normalized:
            candidates.append((*normalized, "cached_focus"))

    spans = [item for item in snapshot.get("spans", []) if isinstance(item, dict)]
    if not candidates and _has_full_span(snapshot):
        content = _content_for_range(snapshot, 1, total_lines)
        if len(content) <= 6000:
            return [(1, total_lines, "validated_full_file")]
    if not candidates:
        for item in reversed(spans):
            normalized = _normalize_source_range(item, total_lines, 0)
            if normalized:
                candidates.append((*normalized, "recent_read"))

    merged: list[tuple[int, int, str]] = []
    for start, end, reason in candidates:
        if not _range_is_covered(snapshot, start, end):
            continue
        overlap_index = next(
            (
                index
                for index, (existing_start, existing_end, _) in enumerate(merged)
                if start <= existing_end + 1 and end >= existing_start - 1
            ),
            None,
        )
        if overlap_index is None:
            merged.append((start, end, reason))
        else:
            existing_start, existing_end, existing_reason = merged[overlap_index]
            merged[overlap_index] = (
                min(start, existing_start),
                max(end, existing_end),
                existing_reason if reason in existing_reason else f"{existing_reason},{reason}",
            )
        if len(merged) >= limit:
            break
    return merged[:limit]


def _normalize_source_range(
    value: Any,
    total_lines: int,
    padding: int,
) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        raw_start, raw_end = value[0], value[1]
    elif isinstance(value, dict):
        raw_start = value.get("start_line") or value.get("line") or value.get("start")
        raw_end = value.get("end_line") or value.get("line") or value.get("end") or raw_start
    else:
        return None
    try:
        start = int(raw_start)
        end = int(raw_end)
    except (TypeError, ValueError):
        return None
    if start <= 0:
        return None
    end = max(start, end)
    return max(1, start - padding), min(total_lines, end + padding)


def _range_is_covered(snapshot: dict[str, Any], start: int, end: int) -> bool:
    return any(
        int(item.get("start_line") or 1) <= start
        and int(item.get("end_line") or 0) >= end
        for item in snapshot.get("spans", []) or []
        if isinstance(item, dict)
    )


def _content_for_range(snapshot: dict[str, Any], start: int, end: int) -> str:
    for item in reversed(snapshot.get("spans", []) or []):
        if not isinstance(item, dict):
            continue
        span_start = int(item.get("start_line") or 1)
        span_end = int(item.get("end_line") or span_start)
        if span_start > start or span_end < end:
            continue
        lines = str(item.get("content") or "").splitlines(keepends=True)
        return "".join(lines[start - span_start : end - span_start + 1])
    return ""


def _has_full_span(snapshot: dict[str, Any]) -> bool:
    total_lines = int(snapshot.get("total_lines") or 0)
    if total_lines <= 0:
        return bool(snapshot.get("is_empty", False))
    return _range_is_covered(snapshot, 1, total_lines)


def _compact_file_note(snapshot: dict[str, Any], *, max_chars: int = 180) -> str:
    if bool(snapshot.get("is_empty", False)):
        return "<empty file>"
    imports_excerpt = str(snapshot.get("imports_excerpt") or "").strip()
    if imports_excerpt:
        first = next((line.strip() for line in imports_excerpt.splitlines() if line.strip()), "")
        if first:
            return _truncate_text(first, max_chars)
    focus_excerpt = str(snapshot.get("focus_excerpt") or "").strip()
    if focus_excerpt:
        first = next(
            (
                line.strip()
                for line in focus_excerpt.splitlines()
                if line.strip() and not line.strip().startswith("# ")
            ),
            "",
        )
        if first:
            return _truncate_text(first, max_chars)
    return ""


def _detailed_file_excerpt(
    snapshot: dict[str, Any],
    *,
    excerpt_chars: int,
    prefer_full_content: bool,
) -> str:
    if bool(snapshot.get("is_empty", False)):
        return "<empty file>"
    if prefer_full_content:
        total_lines = max(1, int(snapshot.get("total_lines") or 1))
        content = _content_for_range(snapshot, 1, total_lines)
        if content:
            return _truncate_text(content, min(2200, excerpt_chars))
    excerpt = str(snapshot.get("focus_excerpt") or "")
    if excerpt:
        return _truncate_text(excerpt, min(900, excerpt_chars))
    spans = [item for item in snapshot.get("spans", []) or [] if isinstance(item, dict)]
    content = str(spans[-1].get("content") or "") if spans else ""
    if content:
        return _truncate_text(content, min(1200, excerpt_chars))
    return ""


def _output_preview(output: dict[str, Any], limit: int) -> dict[str, str]:
    for key in PREVIEW_FIELDS:
        if key not in output:
            continue
        value = output[key]
        if value is None:
            continue
        text = str(value)
        if text:
            return {"field": key, "text": _truncate_text(text, limit)}
    return {}
