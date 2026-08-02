"""Thin lifecycle helpers for runtime phase and completion gates."""

from agent_runtime.lifecycle.completion import derive_phase, evaluate_completion_transition

__all__ = ["derive_phase", "evaluate_completion_transition"]
