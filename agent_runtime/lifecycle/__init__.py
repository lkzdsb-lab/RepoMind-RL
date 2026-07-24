"""Agent lifecycle state and completion coordination."""

from agent_runtime.lifecycle.completion import derive_phase, evaluate_completion_transition
from agent_runtime.lifecycle.goal_contract import build_goal_contract, goal_contract_satisfied
from agent_runtime.lifecycle.obligations import derive_next_obligation, required_action_for_obligation
from agent_runtime.lifecycle.progress_ledger import initial_progress_ledger, update_progress_ledger

__all__ = [
    "build_goal_contract",
    "derive_next_obligation",
    "derive_phase",
    "evaluate_completion_transition",
    "goal_contract_satisfied",
    "initial_progress_ledger",
    "required_action_for_obligation",
    "update_progress_ledger",
]
