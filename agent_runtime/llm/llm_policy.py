"""LLM-backed action policy with constrained action selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.policy import HeuristicDebugPolicy
from agent_runtime.rl.action_space import ActionSpace
from agent_runtime.rl.state_encoder import StateEncoder
from config import LLMConfig
from loguru import logger
from model.agent.actions import Action, ActionSpec
from model.agent.graph import AgentState
from model.llm import ActionChoiceResponse, GuardDecision


@dataclass
class LLMActionPolicy:
    """
        llm 执行决策层
    """
    llm_config: LLMConfig
    action_space: ActionSpace
    q_table: dict[str, dict[str, float]] = field(default_factory=dict)
    encoder: StateEncoder | None = None
    fallback: HeuristicDebugPolicy | None = None
    q_top_k: int = 3
    deny_threshold: float = -0.5

    def __post_init__(self) -> None:
        self.encoder = self.encoder or StateEncoder()
        self.fallback = self.fallback or HeuristicDebugPolicy()
        self.node = LLMJsonNode(
            name="action_policy",
            llm_config=self.llm_config,
            system_prompt=(
                "You are choosing the next action for a debugging agent. "
                "You must choose one action from the q-table allow list. "
                "Return only JSON matching the requested schema."
            ),
            build_prompt=_action_node_prompt,
            fallback=_action_fallback_payload,
            response_model=ActionChoiceResponse,
            normalize=_normalize_action_choice,
        )

    def next_action(self, state: AgentState) -> Action:
        """
            决策下一个动作
        """
        legal_specs = self.action_space.legal_specs(state)
        if not legal_specs:
            return Action("finish", thought="LLM policy found no legal actions.")

        guard = self._guard(legal_specs, state)
        guarded_specs = [
            spec
            for spec in legal_specs
            if spec.name in set(guard.allow_list)
        ]
        fallback_action = self._fallback_action(state, guarded_specs)
        logger.bind(task_id=state.get("task_id")).info(
            "llm action policy requested model={} provider={} legal_actions={} guarded_actions={}",
            self.llm_config.model,
            self.llm_config.provider,
            [spec.name for spec in legal_specs],
            [spec.name for spec in guarded_specs],
        )
        data = self.node.run(
            state,
            {
                "legal_specs": legal_specs,
                "guard": guard,
                "fallback_action": fallback_action,
            },
        )
        if data.get("source") == "fallback":
            return Action(
                fallback_action.name,
                fallback_action.args,
                thought=fallback_action.thought,
                metadata={
                    "llm_policy_fallback": {
                        "error": data.get("fallback_reason"),
                        "guard": guard.to_dict(),
                    }
                },
            )
        action_name = str(data.get("action", "")).strip()
        reason = str(data.get("reason", "")).strip()
        selected_spec = next((spec for spec in guarded_specs if spec.name == action_name), None)
        # 如果 llm 没有选择，则降级使用人工匹配
        if selected_spec is None:
            rejection = self._record_guard_rejection(
                state=state,
                selected_action=action_name,
                llm_reason=reason,
                guard=guard,
                fallback_action=fallback_action,
            )
            return Action(
                fallback_action.name,
                fallback_action.args,
                thought=(
                    f"LLM selected `{action_name}`, but q-table guard rejected it; "
                    f"fallback to `{fallback_action.name}`."
                ),
                metadata={"llm_guard_rejection": rejection},
            )
        action = self.action_space.to_action(selected_spec, state)
        thought = reason or action.thought or f"LLM selected `{selected_spec.name}`."
        return Action(
            action.name,
            action.args,
            thought=thought,
            metadata={
                "llm_guard": {
                    "selected_action": action.name,
                    "guard": guard.to_dict(),
                }
            },
        )

    def _fallback_action(self, state: AgentState, legal_specs: list[ActionSpec]) -> Action:
        if not self.fallback:
            logger.info("fallback action not found")
            return Action(name="", args={}, thought="", metadata={})
        action = self.fallback.next_action(state)
        if any(spec.name == action.name for spec in legal_specs):
            return action
        return self.action_space.to_action(legal_specs[0], state)

    def _guard(self, legal_specs: list[ActionSpec], state: AgentState) -> GuardDecision:
        """
            截断 reward 低于 threshold 的 action
            并记录到上下文中
        """
        assert self.encoder is not None
        state_key = self.encoder.encode(state).key
        q_values = self.q_table.get(state_key, {})
        legal_actions = [spec.name for spec in legal_specs]
        if not q_values:
            allowed = legal_actions[: self.q_top_k]
            logger.bind(task_id=state.get("task_id")).info(
                "do not find the value in q-table",
                state=state
            )
            return GuardDecision(
                state_key=state_key,
                q_values={},
                legal_actions=legal_actions,
                hard_denied={},
                allow_list=allowed,
                allow_scores={name: 0.0 for name in allowed},
            )

        # 过滤 spec
        passed_specs: list[tuple[ActionSpec, float]] = []
        hard_denied: dict[str, float] = {}
        for spec in legal_specs:
            score = float(q_values.get(spec.name, 0.0))
            if score <= self.deny_threshold:
                hard_denied[spec.name] = score
            else:
                passed_specs.append((spec, score))

        # 根据 reward 排序取 top_k 个
        scored_and_sorted = sorted(passed_specs, key=lambda item: item[1], reverse=True)
        allowed_scored = scored_and_sorted[: self.q_top_k]
        fallback_forced = False

        # 如果所有 legal actions 都低于阈值，至少放行一个，避免 agent 卡死。
        if not allowed_scored and legal_specs:
            all_scored = sorted(
                ((spec, float(q_values.get(spec.name, 0.0))) for spec in legal_specs),
                key=lambda item: item[1],
                reverse=True
            )
            allowed_scored = [all_scored[0]]
            fallback_forced = True
            logger.bind(task_id=state.get("task_id")).warning(
                "all legal actions hard-denied; allowing least-bad action state={} action={} q={:.3f}",
                state_key,
                allowed_scored[0][0].name,
                allowed_scored[0][1],
            )

        decision = GuardDecision(
            state_key=state_key,
            q_values={name: float(value) for name, value in q_values.items()},
            legal_actions=legal_actions,
            hard_denied=hard_denied,
            allow_list=[spec.name for spec, _ in allowed_scored],
            allow_scores={spec.name: score for spec, score in allowed_scored},
            fallback_forced=fallback_forced,
        )
        logger.bind(task_id=state.get("task_id")).debug(
            "llm qtable guard decision={}",
            decision.to_dict(),
        )
        return decision

    def _record_guard_rejection(
        self,
        state: AgentState,
        selected_action: str,
        llm_reason: str,
        guard: GuardDecision,
        fallback_action: Action,
    ) -> dict[str, Any]:
        """
            记录被 q-table 卡死的 llm 决策
        """
        if selected_action in guard.hard_denied:
            reason = "hard_denied"
        elif selected_action in guard.legal_actions:
            reason = "not_in_top_k_allow_list"
        else:
            reason = "illegal_action"

        event = {
            "type": "llm_guard_rejection",
            "selected_action": selected_action,
            "llm_reason": llm_reason,
            "rejection_reason": reason,
            "fallback_action": fallback_action.name,
            "guard": guard.to_dict(),
        }
        state["observations"] = state.get("observations", []) + [event]
        state["llm_guard_events"] = state.get("llm_guard_events", []) + [event]
        logger.bind(task_id=state.get("task_id")).warning(
            "llm action rejected by qtable guard selected={} reason={} fallback={}",
            selected_action,
            reason,
            fallback_action.name,
        )
        return event


def _action_node_prompt(state: AgentState, context: dict[str, Any]) -> str:
    legal_specs = context.get("legal_specs") or []
    guard = context.get("guard")
    fallback_action = context.get("fallback_action")
    if not isinstance(guard, GuardDecision):
        guard = GuardDecision(
            state_key="",
            q_values={},
            legal_actions=[getattr(spec, "name", "") for spec in legal_specs],
            hard_denied={},
            allow_list=[getattr(spec, "name", "") for spec in legal_specs],
            allow_scores={},
        )
    if not isinstance(fallback_action, Action):
        fallback_action = Action("finish", thought="Fallback action.")
    return _action_prompt(state, legal_specs, guard, fallback_action)


def _action_fallback_payload(state: AgentState, context: dict[str, Any]) -> dict[str, Any]:
    fallback_action = context.get("fallback_action")
    if not isinstance(fallback_action, Action):
        fallback_action = Action("finish", thought="Fallback action.")
    return {
        "action": fallback_action.name,
        "reason": fallback_action.thought,
    }


def _normalize_action_choice(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    fallback = _action_fallback_payload(state, context)
    action = str(data.get("action") or fallback["action"]).strip()
    reason = str(data.get("reason") or fallback["reason"]).strip()
    return {"action": action, "reason": reason}


def _action_prompt(
    state: AgentState,
    legal_specs: list[ActionSpec],
    guard: GuardDecision,
    fallback_action: Action,
) -> str:
    """
        构造每个 action 的 prompt 格式
    """
    legal = [
        {"name": spec.name, "description": spec.description}
        for spec in legal_specs
    ]
    return (
        f"title={state.get('title', '')}\n"
        f"description={state.get('description', '')}\n"
        f"current_step={state.get('current_step', '')}\n"
        f"status={state.get('status', '')}\n"
        f"candidate_files={state.get('candidate_files', [])}\n"
        f"test_results={state.get('test_results', [])[-2:]}\n"
        f"patch_summary={state.get('patch_summary')}\n"
        f"memory_context={state.get('memory_context', '')[:2500]}\n"
        f"compressed_context={state.get('compressed_context', '')[:2500]}\n"
        f"legal_actions={json.dumps(legal, ensure_ascii=False)}\n"
        f"qtable_guard={json.dumps(guard.to_dict(), ensure_ascii=False)}\n"
        f"allowed_actions={json.dumps(guard.allow_list, ensure_ascii=False)}\n"
        f"hard_denied_actions={json.dumps(guard.hard_denied, ensure_ascii=False)}\n"
        f"fallback_action={json.dumps({'name': fallback_action.name, 'args': fallback_action.args}, ensure_ascii=False)}\n"
        "Choose only from allowed_actions. Return JSON like {\"action\": \"read_file\", \"reason\": \"...\"}."
    )
