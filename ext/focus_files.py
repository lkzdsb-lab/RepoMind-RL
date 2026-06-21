from __future__ import annotations

from typing import Any

from agent_runtime.execution_queue import current_execution_item as queue_current_execution_item
from ext.file_requirements import full_read_requirements
from model.agent.graph import AgentState


def current_execution_item(state: AgentState) -> dict[str, Any] | None:
    return queue_current_execution_item(state)


def execution_target_files(state: AgentState) -> list[str]:
    item = current_execution_item(state)
    if not isinstance(item, dict):
        return []
    return [
        str(path).strip()
        for path in item.get("target_files", []) or []
        if str(path).strip()
    ]


def edited_files_needing_reread(state: AgentState) -> list[str]:
    edited_files = [
        str(path).strip()
        for path in state.get("edited_files", []) or []
        if str(path).strip()
    ]
    if not edited_files or not bool(state.get("verification_stale", False)):
        return []
    calls = state.get("tool_calls", []) or []
    latest_edit_index = -1
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or call.get("name") != "apply_code_patch":
            continue
        output = call.get("output")
        if isinstance(output, dict) and output.get("applied"):
            latest_edit_index = index
    if latest_edit_index < 0:
        return []
    read_after_edit: set[str] = set()
    for call in calls[latest_edit_index + 1 :]:
        if not isinstance(call, dict) or call.get("name") != "read_file":
            continue
        output = call.get("output")
        call_input = call.get("input")
        if not isinstance(output, dict) or output.get("error"):
            continue
        path = str(output.get("file_path") or "").strip()
        if not path and isinstance(call_input, dict):
            path = str(call_input.get("file_path") or "").strip()
        if path:
            read_after_edit.add(path)
    return [path for path in edited_files if path not in read_after_edit]


def current_focus_files(
    state: AgentState,
    *,
    limit: int = 4,
) -> list[str]:
    files: list[str] = []
    _append_unique(files, execution_target_files(state))
    _append_unique(files, edited_files_needing_reread(state))
    _append_unique(
        files,
        [
            str(item.get("file_path") or "").strip()
            for item in full_read_requirements(
                state,
                candidate_files=list(state.get("candidate_files", []) or []),
                limit=max(limit, 6),
            )
            if isinstance(item, dict) and str(item.get("file_path") or "").strip()
        ],
    )
    if not files:
        _append_unique(
            files,
            [
                str(path).strip()
                for path in state.get("candidate_files", []) or []
                if str(path).strip()
            ],
        )
    return files[:limit]


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        item = str(value).strip()
        if item and item not in target:
            target.append(item)
