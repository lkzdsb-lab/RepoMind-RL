"""Thin runtime completion gates.

Semantic completion belongs to the LLM completion judge. This module only
derives an advisory phase and reports deterministic blockers.
"""

from __future__ import annotations

from model.agent.graph import AgentState


def derive_phase(state: AgentState) -> str:
    status = str(state.get("status") or "").strip().lower()
    if status in {"finished", "failed"}:
        return "complete"
    if status == "awaiting_user_input":
        return "awaiting_user_input"
    if bool(state.get("plan_mode", False)):
        return "plan"
    pending = state.get("pending_resolution")
    if isinstance(pending, dict) and pending:
        return "recover"
    if bool(state.get("verification_stale", False)):
        return "verify"
    if state.get("edited_files"):
        return "execute"
    if not state.get("code_context") and not state.get("read_file_cache"):
        return "collect_context"
    return "execute"


def evaluate_completion_transition(state: AgentState) -> dict[str, Any]:
    """Return deterministic blockers without deciding task semantics."""
    phase = derive_phase(state)
    blockers: list[str] = []
    required_next_action = ""
    pending = state.get("pending_resolution")
    if not isinstance(pending, dict):
        pending = {}

    if state.get("error"):
        blockers.append("unresolved_tool_error")
    if phase == "awaiting_user_input":
        blockers.append("awaiting_user_input")
        required_next_action = "request_user_input"
    if phase == "plan":
        blockers.append("plan_mode_active")
        required_next_action = "ExitPlanMode"
    if pending:
        blockers.append(f"pending_resolution:{pending.get('kind') or 'unknown'}")
        required_next_action = str(pending.get("required_next_action") or "read_file")
    if bool(state.get("verification_stale", False)):
        blockers.append("verification_stale")
        required_next_action = required_next_action or "run_shell_command"

    runtime_facts = state.get("runtime_facts")
    if isinstance(runtime_facts, dict):
        edit_revision = int(runtime_facts.get("edit_revision", 0) or 0)
        verified_revision = int(runtime_facts.get("verified_revision", 0) or 0)
        if edit_revision > verified_revision and "verification_stale" not in blockers:
            blockers.append("unverified_edit_revision")
            required_next_action = required_next_action or "run_shell_command"

    return {
        "phase": phase,
        # A non-terminal run is completed only after the LLM selects finish and
        # the completion judge accepts it.
        "is_complete": str(state.get("status") or "") == "finished",
        "blockers": blockers,
        "required_next_action": required_next_action,
        "completion_signal": _completion_signal_present(state),
    }


def _completion_signal_present(state: AgentState) -> bool:
    judgement = state.get("completion_judgement")
    return (
        isinstance(judgement, dict)
        and str(judgement.get("decision") or "").strip().lower() == "complete"
    )
