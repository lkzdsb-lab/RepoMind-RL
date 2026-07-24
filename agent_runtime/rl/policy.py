"""Epsilon-greedy Q-learning policy for the debug agent."""

from __future__ import annotations

import random
from dataclasses import dataclass

from loguru import logger

from agent_runtime.actions import ActionFactory
from agent_runtime.rl.action_space import ActionSpace
from agent_runtime.rl.state_encoder import StateEncoder
from agent_runtime.verification.capabilities import recommended_verification_command
from model.agent.actions import Action
from model.agent.graph import AgentState


@dataclass
class QLearningDebugPolicy:
    """RL-backed action policy using the current legal action space."""

    q_table: dict[str, dict[str, float]]
    epsilon: float = 0.15
    encoder: StateEncoder | None = None
    action_space: ActionSpace | None = None
    action_factory: ActionFactory | None = None

    def __post_init__(self) -> None:
        self.encoder = self.encoder or StateEncoder()
        self.action_space = self.action_space or ActionSpace()
        self.action_factory = self.action_factory or ActionFactory()

    def make_initial_plan(self, state: AgentState) -> list[str]:
        verification_example = _verification_example(state)
        verification_step = (
            "Skip command verification because the task contract does not require it."
            if not _verification_required(state)
            else f"Run an allowed verification command, for example: {verification_example}"
        )
        return [
            "Use RL policy to choose the next action from state features.",
            "Prefer structured codebase context to locate candidate files.",
            "Read candidate files and inspect diff before editing.",
            verification_step,
            "Update replay memory and Q-table from reward feedback.",
        ]

    def next_action(self, state: AgentState) -> Action:
        assert self.encoder is not None
        assert self.action_space is not None
        assert self.action_factory is not None

        legal_specs = self.action_space.legal_specs(state)
        if not legal_specs:
            logger.bind(task_id=state.get("task_id")).warning("rl action space empty; finishing")
            return self.action_factory.create("finish", thought="RL action space is empty.")

        encoded = self.encoder.encode(state)

        if random.random() < self.epsilon:
            spec = random.choice(legal_specs)
            action_args = self.action_factory.default_args(spec, state)
            logger.bind(task_id=state.get("task_id"), action=spec.name).info(
                "rl policy exploring epsilon={} state={}",
                self.epsilon,
                encoded.key,
            )
            return self.action_factory.build(
                spec,
                state,
                resolved_args=action_args,
                thought=(
                    f"{self.action_factory.default_thought(spec.name, action_args)} "
                    f"epsilon exploration, state={encoded.key}"
                ),
            )

        action_values = self.q_table.get(encoded.key, {})
        spec = max(
            legal_specs,
            key=lambda item: (action_values.get(item.name, 0.0), -legal_specs.index(item)),
        )
        action_args = self.action_factory.default_args(spec, state)
        logger.bind(task_id=state.get("task_id"), action=spec.name).debug(
            "rl policy exploiting q_value={:.3f} state={} legal_actions={}",
            action_values.get(spec.name, 0.0),
            encoded.key,
            [item.name for item in legal_specs],
        )
        return self.action_factory.build(
            spec,
            state,
            resolved_args=action_args,
            thought=(
                f"{self.action_factory.default_thought(spec.name, action_args)} "
                f"Q={action_values.get(spec.name, 0.0):.3f}, state={encoded.key}"
            ),
        )


def _verification_required(state: AgentState) -> bool:
    return bool(state.get("verification_required", True))


def _verification_example(state: AgentState) -> str:
    return recommended_verification_command(state)
