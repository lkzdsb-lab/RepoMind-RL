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
                "full_read": bool(snapshot.get("full_read", False)),
            }
            if is_focus:
                summary["detail_level"] = "target"
                imports_excerpt = str(snapshot.get("imports_excerpt") or "")
                if imports_excerpt:
                    summary["imports_excerpt"] = _truncate_text(imports_excerpt, min(500, excerpt_chars))
                if snapshot.get("focus_ranges"):
                    summary["focus_ranges"] = snapshot.get("focus_ranges")
                excerpt = str(snapshot.get("focus_excerpt") or "")
                if excerpt:
                    summary["content_excerpt"] = _truncate_text(excerpt, min(900, excerpt_chars))
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
