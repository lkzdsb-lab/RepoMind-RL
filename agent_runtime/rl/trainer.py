"""Q-learning trainer and persistent Q-table."""

from __future__ import annotations

import json
from pathlib import Path

from agent_runtime.rl.action_space import ActionSpace
from model.agent.transition import Transition
from loguru import logger


class QTableStore:
    """
        表格记录奖励值，后续考虑是否持久化
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, float]]:
        if not self.path.exists():
            logger.info("rl q-table not found path={}", self.path)
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        q_table = {
            str(state): {str(action): float(value) for action, value in actions.items()}
            for state, actions in data.items()
        }
        logger.info("rl q-table loaded path={} states={}", self.path, len(q_table))
        return q_table

    def save(self, q_table: dict[str, dict[str, float]]) -> None:
        self.path.write_text(
            json.dumps(q_table, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.debug("rl q-table saved path={} states={}", self.path, len(q_table))



class QLearningTrainer:
    """
        强化学习训练策略，贝尔曼方程
        后续补优化
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
        next_max = 0.0 if transition.done else self.max_q(transition.next_state_key)
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
        values = self.q_table.get(state_key, {})
        return max(values.values()) if values else 0.0
