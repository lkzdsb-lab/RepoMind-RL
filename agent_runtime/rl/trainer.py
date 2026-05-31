"""Q-learning trainer and persistent Q-table with versioned envelope."""

from __future__ import annotations

import json
from pathlib import Path

from agent_runtime.rl.action_space import ActionSpace
from model.agent.transition import Transition
from loguru import logger


class QTableStore:
    """Persistent Q-table storage with versioned envelope.

    Envelope format (v2)::

        {
          "metadata": {
            "encoder_version": "state-encoder-v1",
            "action_space_version": "action-space-v1",
            "reward_version": "reward-v1"
          },
          "q_values": {
            "state_key_1": {"action_a": 0.5, "action_b": -0.1},
            ...
          }
        }

    Legacy format (plain dict) is detected and treated as unversioned —
    an empty q_table is returned to prevent old data from polluting new
    policies.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, float]]:
        if not self.path.exists():
            logger.info("rl q-table not found path={}", self.path)
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("rl q-table unreadable path={} error={}", self.path, exc)
            return {}

        # Envelope format (v2)
        if isinstance(data, dict) and "metadata" in data:
            meta = data.get("metadata", {})
            logger.info(
                "rl q-table loaded (envelope) path={} states={} encoder={} action_space={} reward={}",
                self.path,
                len(data.get("q_values", {})),
                meta.get("encoder_version", ""),
                meta.get("action_space_version", ""),
                meta.get("reward_version", ""),
            )
            q_values = data.get("q_values", {})
            return {
                str(state): {
                    str(action): float(value) for action, value in actions.items()
                }
                for state, actions in q_values.items()
            }

        # Legacy format (plain dict, no metadata) — treat as unversioned
        logger.warning(
            "rl q-table is legacy/unversioned (no metadata envelope); "
            "returning empty q_table to avoid polluting new policy. "
            "path={} keys={}",
            self.path,
            len(data),
        )
        return {}

    def save(
        self,
        q_table: dict[str, dict[str, float]],
        encoder_version: str = "",
        action_space_version: str = "",
        reward_version: str = "",
    ) -> None:
        envelope = {
            "metadata": {
                "encoder_version": encoder_version,
                "action_space_version": action_space_version,
                "reward_version": reward_version,
            },
            "q_values": q_table,
        }
        self.path.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.debug("rl q-table saved path={} states={}", self.path, len(q_table))


class QLearningTrainer:
    """Online Q-learning trainer.

    ``update()`` uses ``transition.next_legal_actions`` (not all historical
    actions in the Q-table) to compute ``next_max``.  When
    ``next_legal_actions`` is empty or ``done=True``, ``next_max=0``.
    """

    def __init__(
        self,
        q_table: dict[str, dict[str, float]],
        action_space: ActionSpace,
        learning_rate: float = 0.2,
        discount: float = 0.9,
    ) -> None:
        self.q_table = q_table
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount = discount

    def update(self, transition: Transition) -> float:
        current = self.q_value(transition.state_key, transition.action)
        next_max = 0.0
        if not transition.done and transition.next_legal_actions:
            next_max = self._max_legal_q(
                transition.next_state_key, transition.next_legal_actions
            )
        target = transition.reward + self.discount * next_max
        updated = current + self.learning_rate * (target - current)
        self.q_table.setdefault(transition.state_key, {})[transition.action] = updated
        logger.debug(
            "rl q-value updated action={} reward={:.3f} current={:.3f} target={:.3f} updated={:.3f}",
            transition.action,
            transition.reward,
            current,
            target,
            updated,
        )
        return updated

    def train_batch(self, transitions: list[Transition]) -> int:
        for transition in transitions:
            self.update(transition)
        return len(transitions)

    def q_value(self, state_key: str, action: str) -> float:
        return self.q_table.get(state_key, {}).get(action, 0.0)

    def max_q(self, state_key: str) -> float:
        """Maximum Q-value for *state_key* across all known actions.

        Prefer ``_max_legal_q`` when legal actions are known.
        """
        values = self.q_table.get(state_key, {})
        return max(values.values()) if values else 0.0

    def _max_legal_q(self, state_key: str, legal_actions: list[str]) -> float:
        """Maximum Q-value for *state_key*, restricted to *legal_actions*."""
        action_values = self.q_table.get(state_key, {})
        if not action_values or not legal_actions:
            return 0.0
        legal_values = [
            action_values.get(action, 0.0) for action in legal_actions
        ]
        return max(legal_values) if legal_values else 0.0
