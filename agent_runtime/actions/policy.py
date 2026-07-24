"""Policy contract shared by LLM and RL action policies."""

from __future__ import annotations

from typing import Protocol

from model.agent.actions import Action
from model.agent.graph import AgentState


class ActionPolicy(Protocol):
    def make_initial_plan(self, state: AgentState) -> list[str]: ...

    def next_action(self, state: AgentState) -> Action: ...
