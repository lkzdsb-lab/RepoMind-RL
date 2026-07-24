from __future__ import annotations

from typing import Any

from agent_runtime.lifecycle.execution_queue import (
    _completion_signal_present,
    current_execution_item,
    reconcile_execution_queue,
)
from agent_runtime.lifecycle.goal_contract import goal_contract_satisfied
from agent_runtime.lifecycle.obligations import derive_next_obligation, required_action_for_obligation
from model.agent.graph import AgentState


def derive_phase(
    state: AgentState,
    execution_queue: list[dict[str, Any]] | None = None,
) -> str:
    """ 推断下一个阶段"""
    queue = execution_queue if execution_queue is not None else reconcile_execution_queue(state)
    reconciled_state = {**state, "execution_queue": queue}
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
    execution = current_execution_item(reconciled_state)
    if isinstance(execution, dict):
        kind = str(execution.get("kind") or "").strip().lower()
        if kind == "verify":
            return "verify"
        if kind == "patch":
            return "execute_patch"
    if bool(state.get("verification_stale", False)):
        return "verify"
    if not state.get("code_context") and not state.get("selected_code_context"):
        return "collect_context"
    if bool(state.get("plan_mode_approved", False)) and not current_execution_item(reconciled_state):
        return "complete"
    return "collect_context"


def evaluate_completion_transition(state: AgentState) -> dict[str, Any]:
    """ 评估下一个 action，判断是否需要终止"""
    goal_contract = state.get("goal_contract") if isinstance(state.get("goal_contract"), dict) else {}
    progress_ledger = state.get("progress_ledger") if isinstance(state.get("progress_ledger"), dict) else {}
    obligation = derive_next_obligation(goal_contract, progress_ledger)
    queue = reconcile_execution_queue(state, obligation=obligation)
    reconciled_state = {
        **state,
        "execution_queue": queue,
        "next_obligation": obligation,
    }
    phase = derive_phase(reconciled_state, execution_queue=queue)
    # 阻塞队列
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

    completion_signal = _completion_signal_present(state)
    if goal_contract and str(obligation.get("kind") or "") != "complete":
        blockers.append(f"obligation:{obligation.get('kind')}")
        next_action = next_action or required_action_for_obligation(obligation)
        if phase == "complete":
            phase = _phase_for_obligation(obligation)
    if goal_contract:
        is_complete = not blockers and goal_contract_satisfied(goal_contract, progress_ledger)
    else:
        is_complete = not blockers and (completion_signal or phase == "complete")
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
        "next_obligation": obligation,
    }


def _phase_for_obligation(obligation: dict[str, Any]) -> str:
    kind = str(obligation.get("kind") or "")
    if kind == "diagnose":
        return "collect_context"
    if kind == "implement":
        return "execute_patch"
    if kind == "verify":
        return "verify"
    return "collect_context"
