"""Compact prompt-safe summaries for heterogeneous tool outputs."""

from __future__ import annotations

from typing import Any

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
        content = output.get("content")
        if content is not None:
            text = str(content)
            if text:
                summary["content_excerpt"] = _truncate_text(text, excerpt_chars)
        _put_if_present(summary, "error", call.get("error") or output.get("error"))
        if summary:
            summaries.append(summary)
    return summaries


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

