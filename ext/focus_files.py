"""Advisory file focus helpers without execution-queue authority."""

from __future__ import annotations

from model.agent.graph import AgentState


def edited_files_needing_reread(state: AgentState) -> list[str]:
    """Compatibility helper: edits no longer require a mandatory reread."""
    del state
    return []


def current_focus_files(state: AgentState, *, limit: int = 4) -> list[str]:
    files: list[str] = []
    focus = state.get("attention_focus")
    if isinstance(focus, dict):
        _append_unique(files, focus.get("focus_files", []))
    _append_unique(files, state.get("edited_files", []))
    _append_unique(files, state.get("candidate_files", []))
    return files[:limit]


def _append_unique(target: list[str], values: object) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        path = str(value or "").strip()
        if path and path not in target:
            target.append(path)
