"""Epsilon-greedy Q-learning policy for the debug agent."""

from __future__ import annotations

import random
from dataclasses import dataclass

from agent_runtime.rl.action_space import ActionSpace
from agent_runtime.rl.state_encoder import StateEncoder
from model.agent.actions import Action
from model.agent.graph import AgentState
from loguru import logger



@dataclass
class QLearningDebugPolicy:
    """
        基于 rl 的 llm 决策，贪婪策略
        目前 agent 的精华部分
    """
    q_table: dict[str, dict[str, float]]
    epsilon: float = 0.15
    encoder: StateEncoder | None = None
    action_space: ActionSpace | None = None

    def __post_init__(self) -> None:
        self.encoder = self.encoder or StateEncoder()
        self.action_space = self.action_space or ActionSpace()

    def make_initial_plan(self, state: AgentState) -> list[str]:
        verify_command = state.get("verify_command") or "pytest"
        verification_step = (
            "跳过验证命令（LLM 判定本任务不需要命令验证）"
            if not _verification_required(state)
            else f"验证命令：{verify_command}"
        )
        return [
            "使用 RL policy 基于 state features 选择下一步 action",
            "优先利用结构化 codebase context 定位候选文件",
            "阅读候选文件并检查 diff",
            verification_step,
            "根据 reward 写入 replay buffer 并在线更新 Q-table",
        ]


    def next_action(self, state: AgentState) -> Action:
        """
            预估下一个步骤
            两个方向：
            1. 经验主义
            2. 随机分流到非经验主义
        """
        assert self.encoder is not None
        assert self.action_space is not None

        legal_specs = self.action_space.legal_specs(state)
        if not legal_specs:
            logger.bind(task_id=state.get("task_id")).warning("rl action space empty; finishing")
            return Action("finish", thought="RL action space 为空，结束任务。")

        encoded = self.encoder.encode(state)

        # 以一个不大的概率去探索一条 不按照经验得出的 新道路
        if random.random() < self.epsilon:
            spec = random.choice(legal_specs)
            action = self.action_space.to_action(spec, state)
            logger.bind(task_id=state.get("task_id"), action=action.name).info(
                "rl policy exploring epsilon={} state={}",
                self.epsilon,
                encoded.key,
            )
            return Action(
                action.name,
                action.args,
                thought=f"{action.thought} epsilon 探索，state={encoded.key}",
            )

        # 利用经验操作进行处理
        action_values = self.q_table.get(encoded.key, {})
        # 选择 q 表中分值最大且最靠前的那一个
        spec = max(
            legal_specs,
            key=lambda item: (action_values.get(item.name, 0.0), -legal_specs.index(item)),
        )
        action = self.action_space.to_action(spec, state)
        logger.bind(task_id=state.get("task_id"), action=action.name).debug(
            "rl policy exploiting q_value={:.3f} state={} legal_actions={}",
            action_values.get(spec.name, 0.0),
            encoded.key,
            [item.name for item in legal_specs],
        )
        return Action(
            action.name,
            action.args,
            thought=(
                f"{action.thought} Q={action_values.get(spec.name, 0.0):.3f}，"
                f"state={encoded.key}"
            ),
        )


def _verification_required(state: AgentState) -> bool:
    return bool(state.get("verification_required", True))
