"""JSON replay buffer for RL transitions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from model.agent.transition import Transition


class ReplayBuffer:
    """
        rl 的记录
    """
    def __init__(self, path: str | Path, max_size: int = 10000) -> None:
        self.path = Path(path)
        self.max_size = max_size
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, transition: Transition) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(transition.to_dict(), ensure_ascii=False) + "\n")
        self._compact_if_needed()

    def list(self) -> list[Transition]:
        if not self.path.exists():
            return []
        transitions: list[Transition] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    transitions.append(Transition.from_dict(json.loads(line)))
        return transitions

    # 随机抽取
    def sample(self, batch_size: int) -> list[Transition]:
        transitions = self.list()
        if len(transitions) <= batch_size:
            return transitions
        return random.sample(transitions, batch_size)


    def _compact_if_needed(self) -> None:
        """
            记忆删除策略：滑动窗口删除旧记忆
        """
        transitions = self.list()
        if len(transitions) <= self.max_size:
            return
        kept = transitions[-self.max_size :]
        with self.path.open("w", encoding="utf-8") as fp:
            for transition in kept:
                fp.write(json.dumps(transition.to_dict(), ensure_ascii=False) + "\n")
