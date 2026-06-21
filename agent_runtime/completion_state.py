from __future__ import annotations

from typing import Any

from agent_runtime.execution_queue import current_execution_item, reconcile_execution_queue
from model.agent.graph import AgentState


def derive_phase(state: AgentState) -> str:
    queue = reconcile_execution_queue(state)
    status = str(state.get("status") or "").strip().lower()
    if status in {"finished", "failed"}:
        return "complete"
    if status == "awaiting_user_input":
        return "awaiting_user_input"
    if bool(state.get("plan_mode", False)):
        return "plan"
    pending_resolution = state.get("pending_resolution") or {}
    if str(pending_resolution.get("kind") or "") == "recovery":
        return "recover"
    if str(pending_resolution.get("kind") or "") == "deferred":
        return "resolve_action"
    if bool(state.get("verification_stale", False)):
        return "verify"
    execution = current_execution_item({**state, "execution_queue": queue})
    if isinstance(execution, dict):
        kind = str(execution.get("kind") or "").strip().lower()
        if kind == "verify":
            return "verify"
        if kind == "patch":
            return "execute_patch"
    if not state.get("code_context") and not state.get("selected_code_context"):
        return "collect_context"
    if bool(state.get("plan_mode_approved", False)) and not current_execution_item({**state, "execution_queue": queue}):
        return "complete"
    return "collect_context"


def evaluate_completion_transition(state: AgentState) -> dict[str, Any]:
    """ 评估下一个 action"""
    queue = reconcile_execution_queue(state)
    phase = derive_phase({**state, "execution_queue": queue})
    blockers: list[str] = []
    next_action = ""
    pending_resolution = state.get("pending_resolution") or {}

    if state.get("error"):
        blockers.append("error")
    if phase == "awaiting_user_input":
        blockers.append("awaiting_user_input")
        next_action = "request_user_input"
    if phase == "plan":
        blockers.append("plan_mode_active")
        next_action = "ExitPlanMode"
    if str(pending_resolution.get("kind") or "") == "recovery":
        blockers.append("pending_resolution:recovery")
        next_action = str(pending_resolution.get("required_next_action") or "read_file")
    elif str(pending_resolution.get("kind") or "") == "deferred":
        blockers.append("pending_resolution:deferred")
        next_action = str(pending_resolution.get("required_next_action") or "")
    if bool(state.get("verification_stale", False)):
        blockers.append("verification_stale")
        next_action = next_action or "run_shell_command"

    execution = current_execution_item({**state, "execution_queue": queue})
    completion_signal = _completion_signal_present(state)
    if isinstance(execution, dict) and str(execution.get("status") or "pending") == "pending":
        kind = str(execution.get("kind") or "").strip().lower()
        if not completion_signal or kind == "verify":
            blockers.append(f"execution_queue_pending:{kind or 'unknown'}")
            if not next_action:
                next_action = "run_shell_command" if kind == "verify" else "apply_code_patch"

    is_complete = not blockers and (
        _has_meaningful_task_result(state) or completion_signal or phase == "complete"
    )
    if is_complete:
        phase = "complete"
        next_action = "finish"
    return {
        "phase": phase,
        "is_complete": is_complete,
        "blockers": blockers,
        "required_next_action": next_action,
        "completion_signal": completion_signal,
        "execution_queue": queue,
    }


def _completion_signal_present(state: AgentState) -> bool:
    judgement = state.get("completion_judgement") or {}
    if str(judgement.get("decision") or "").strip().lower() == "complete":
        return True
    for item in reversed(state.get("llm_observations", []) or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() == "complete":
            return True
    return False


def _has_meaningful_task_result(state: AgentState) -> bool:
    return bool(
        state.get("edited_files")
        or state.get("patch_summary") is not None
        or state.get("test_results")
        or state.get("verification_commands")
        or state.get("change_events")
    )
