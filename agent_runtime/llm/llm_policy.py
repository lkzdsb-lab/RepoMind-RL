"""LLM-backed action policy with constrained action selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.llm.tool_summaries import read_file_summaries
from agent_runtime.policy import HeuristicDebugPolicy
from agent_runtime.rl.action_space import ActionSpace
from agent_runtime.rl.state_encoder import StateEncoder
from config import LLMConfig
from loguru import logger
from model.agent.actions import Action, ActionSpec
from model.agent.graph import AgentState
from model.llm import ActionChoiceResponse, GuardDecision
from prompts.templates import load_prompt, render_prompt
from utils import _truncate_text, _clamp_float, _as_bool, _clean_string_list


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
            system_prompt=load_prompt("system/action_policy.md"),
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

        qtable_context = self._guard(legal_specs, state)
        fallback_action = self._fallback_action(state, legal_specs)
        logger.bind(task_id=state.get("task_id")).info(
            "llm action policy requested model={} provider={} legal_actions={} qtable_hard_denied={}",
            self.llm_config.model,
            self.llm_config.provider,
            [spec.name for spec in legal_specs],
            qtable_context.hard_denied,
        )
        data = self.node.run(
            state,
            {
                "legal_specs": legal_specs,
                "guard": qtable_context,
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
                        "guard": qtable_context.to_dict(),
                    }
                },
            )
        action_name = str(data.get("action", "")).strip()
        candidate_actions = _candidate_actions_from_response(data, action_name, fallback_action)
        reason = str(data.get("reason", "")).strip()
        guard = self._guard(legal_specs, state, candidate_actions)
        legal_by_name = {spec.name: spec for spec in legal_specs}
        guarded_specs = [
            legal_by_name[name]
            for name in guard.allow_list
            if name in legal_by_name
        ]
        selected_spec = guarded_specs[0] if guarded_specs else None
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
        preferred_action = candidate_actions[0] if candidate_actions else action_name
        rejection = None
        if preferred_action and selected_spec.name != preferred_action:
            rejection = self._record_guard_rejection(
                state=state,
                selected_action=preferred_action,
                llm_reason=reason,
                guard=guard,
                fallback_action=self.action_space.to_action(selected_spec, state),
            )
        action = self.action_space.to_action(selected_spec, state)
        action_args = action.args
        if selected_spec.name in {
            "apply_code_patch",
            "request_user_input",
            "read_file",
            "search_text",
            "run_shell_command",
            "EnterPlanMode",
            "ExitPlanMode",
        }:
            action_args = _action_input_for(selected_spec.name, data, reason, action.args)
        thought = reason or action.thought or f"LLM selected `{selected_spec.name}`."
        if selected_spec.name != preferred_action:
            thought = (
                f"{thought} q-table guard clipped preferred action "
                f"`{preferred_action}` to `{selected_spec.name}`."
            )
        return Action(
            action.name,
            action_args,
            thought=thought,
            metadata={
                "llm_guard": {
                    "selected_action": action.name,
                    "llm_candidate_actions": candidate_actions,
                    "guard": guard.to_dict(),
                },
                **({"llm_guard_rejection": rejection} if rejection else {}),
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

    def _guard(
        self,
        legal_specs: list[ActionSpec],
        state: AgentState,
        candidate_actions: list[str] | None = None,
    ) -> GuardDecision:
        """
            截断 reward 低于 threshold 的 action
            并记录到上下文中
        """
        assert self.encoder is not None
        state_key = self.encoder.encode(state).key
        q_values = self.q_table.get(state_key, {})
        legal_actions = [spec.name for spec in legal_specs]
        candidate_specs = _candidate_specs(legal_specs, candidate_actions)
        candidate_names = [spec.name for spec in candidate_specs]
        if not candidate_specs:
            return GuardDecision(
                state_key=state_key,
                q_values={name: float(value) for name, value in q_values.items()},
                legal_actions=legal_actions,
                hard_denied={},
                allow_list=[],
                allow_scores={},
            )
        if not q_values:
            allowed = candidate_names[: self.q_top_k]
            logger.bind(task_id=state.get("task_id")).info(
                "do not find the value in q-table",
                # state=state
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
        for spec in candidate_specs:
            score = float(q_values.get(spec.name, 0.0))
            if score <= self.deny_threshold:
                hard_denied[spec.name] = score
            else:
                passed_specs.append((spec, score))

        # 保留 LLM 给出的候选顺序，只用 q-table 做截断，避免 guard 抢先替 LLM 排序。
        allowed_scored = passed_specs[: self.q_top_k]
        fallback_forced = False

        # 如果 LLM 选出的候选都低于阈值，至少放行候选里的 least-bad，避免 agent 卡死。
        if not allowed_scored and candidate_specs:
            all_scored = sorted(
                ((spec, float(q_values.get(spec.name, 0.0))) for spec in candidate_specs),
                key=lambda item: item[1],
                reverse=True
            )
            allowed_scored = [all_scored[0]]
            fallback_forced = True
            logger.bind(task_id=state.get("task_id")).warning(
                "all llm candidate actions hard-denied; allowing least-bad action state={} action={} q={:.3f}",
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
        "candidate_actions": [fallback_action.name],
        "reason": fallback_action.thought,
        "action_input": fallback_action.args,
        "uncertainty_questions": [],
        "confidence": 0.5,
    }


def _normalize_action_choice(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    fallback = _action_fallback_payload(state, context)
    action = str(data.get("action") or fallback["action"]).strip()
    raw_candidates = data.get("candidate_actions")
    candidate_actions: list[str] = []
    if isinstance(raw_candidates, list):
        candidate_actions = [
            str(item).strip()
            for item in raw_candidates
            if str(item).strip()
        ]
    if action and action not in candidate_actions:
        candidate_actions.insert(0, action)
    if not candidate_actions:
        candidate_actions = list(fallback["candidate_actions"])
        action = str(fallback["action"])
    reason = str(data.get("reason") or fallback["reason"]).strip()
    raw_action_input = data.get("action_input")
    if not isinstance(raw_action_input, dict):
        raw_action_input = dict(fallback.get("action_input") or {})
    uncertainty_questions = _clean_string_list(data.get("uncertainty_questions"))
    confidence = _clamp_float(data.get("confidence"), float(fallback.get("confidence", 0.5)), "invalid confidence")
    return {
        "action": action,
        "candidate_actions": candidate_actions,
        "reason": reason,
        "action_input": raw_action_input,
        "uncertainty_questions": uncertainty_questions,
        "confidence": confidence,
    }


def _action_prompt(
    state: AgentState,
    legal_specs: list[ActionSpec],
    guard: GuardDecision,
    fallback_action: Action,
) -> str:
    """
        构造每个 action 的 prompt 格式
    """
    manifest_by_name = {
        str(item.get("name")): item
        for item in state.get("tool_manifest", [])
        if isinstance(item, dict) and item.get("name")
    }
    legal = []
    for spec in legal_specs:
        item = dict(manifest_by_name.get(spec.name, {}))
        item["name"] = spec.name
        item["description"] = spec.description or item.get("description", "")
        legal.append(item)
    return render_prompt(
        "user/action_policy.md",
        title=state.get("title", ""),
        description=state.get("description", ""),
        current_step=state.get("current_step", ""),
        status=state.get("status", ""),
        verification_required=json.dumps(bool(state.get("verification_required", True))),
        verification_reason=state.get("verification_reason", ""),
        verification_stale=json.dumps(bool(state.get("verification_stale", False))),
        verification_commands=json.dumps(
            state.get("verification_commands", [])[-5:],
            ensure_ascii=False,
            default=str,
        ),
        command_results=json.dumps(
            state.get("command_results", [])[-3:],
            ensure_ascii=False,
            default=str,
        ),
        plan_mode=json.dumps(bool(state.get("plan_mode", False))),
        plan_mode_approved=json.dumps(bool(state.get("plan_mode_approved", False))),
        debug_technical_plan=_truncate_text(str(state.get("debug_technical_plan", "")), 5000),
        plan_mode_evaluation=_truncate_text(str(state.get("plan_mode_evaluation", "")), 3000),
        selected_skills=json.dumps(state.get("selected_skills", []), ensure_ascii=False),
        skill_context=_truncate_text(
            json.dumps(state.get("skill_context", []), ensure_ascii=False),
            3500,
        ),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
        read_files=json.dumps(
            read_file_summaries(state, excerpt_chars=1200),
            ensure_ascii=False,
            default=str,
        ),
        test_results=json.dumps(state.get("test_results", [])[-2:], ensure_ascii=False, default=str),
        patch_summary=state.get("patch_summary"),
        editing_enabled=json.dumps(bool(state.get("editing_enabled", False))),
        edit_results=json.dumps(state.get("edit_results", [])[-2:], ensure_ascii=False, default=str),
        user_inputs=json.dumps(state.get("user_inputs", []), ensure_ascii=False, default=str),
        memory_context=str(state.get("memory_context", ""))[:2500],
        compressed_context=str(state.get("compressed_context", ""))[:2500],
        legal_actions=json.dumps(legal, ensure_ascii=False),
        fallback_action=json.dumps(
            {"name": fallback_action.name, "args": fallback_action.args},
            ensure_ascii=False,
        ),
    )


def _candidate_actions_from_response(
    data: dict[str, Any],
    action_name: str,
    fallback_action: Action,
) -> list[str]:
    raw_candidates = data.get("candidate_actions")
    candidates: list[str] = []
    if isinstance(raw_candidates, list):
        candidates = [str(item).strip() for item in raw_candidates if str(item).strip()]
    if action_name and action_name not in candidates:
        candidates.insert(0, action_name)
    if not candidates and fallback_action.name:
        candidates = [fallback_action.name]
    return _dedupe(candidates)


def _action_input_for(
    action_name: str,
    data: dict[str, Any],
    reason: str,
    default_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
        对各种 action 的输入进行提取
    """
    raw_input = data.get("action_input")
    if not isinstance(raw_input, dict):
        raw_input = {}
    default_args = default_args or {}
    if action_name == "apply_code_patch":
        return {
            "changes": raw_input.get("changes", []),
            "reason": str(raw_input.get("reason") or reason or "").strip(),
            "confidence": _clamp_float(data.get("confidence"), 0.5, "invalid confidence from LLM"),
            "assumptions": _clean_string_list(raw_input.get("assumptions"), -1, 500),
            "uncertainty_questions": _clean_string_list(
                raw_input.get("uncertainty_questions") or data.get("uncertainty_questions")
            , -1, 500),
            "dry_run": bool(raw_input.get("dry_run", False)),
        }
    if action_name == "request_user_input":
        questions = _clean_string_list(raw_input.get("questions") or data.get("uncertainty_questions"), -1, 500)
        return {
            "reason": str(raw_input.get("reason") or reason or "").strip(),
            "questions": questions[:3],
        }
    if action_name == "read_file":
        return {
            "file_path": str(
                raw_input.get("file_path") or default_args.get("file_path") or ""
            ).strip(),
            "max_chars": _bounded_int(
                raw_input.get("max_chars", default_args.get("max_chars", 8000)),
                default=8000,
                minimum=1,
                maximum=200000,
            ),
            "start_line": _optional_bounded_int(raw_input.get("start_line"), minimum=1),
            "end_line": _optional_bounded_int(raw_input.get("end_line"), minimum=1),
        }
    if action_name == "search_text":
        pattern = str(raw_input.get("pattern") or raw_input.get("query") or "").strip()
        return {
            "pattern": pattern,
            "regex": bool(raw_input.get("regex", True)),
            "globs": _clean_string_list(raw_input.get("globs"), -1, 500),
            "context_lines": _bounded_int(raw_input.get("context_lines"), default=0, minimum=0, maximum=5),
            "max_results": _bounded_int(raw_input.get("max_results"), default=50, minimum=1, maximum=200),
        }
    if action_name == "run_shell_command":
        return {
            "command": str(
                raw_input.get("command") or default_args.get("command") or ""
            ).strip(),
            "purpose": _clean_purpose(
                raw_input.get("purpose") or default_args.get("purpose")
            ),
            "timeout": _bounded_int(
                raw_input.get("timeout", default_args.get("timeout", 120)),
                default=120,
                minimum=1,
                maximum=1800,
            ),
            "reason": str(
                raw_input.get("reason") or reason or default_args.get("reason") or ""
            ).strip()[:500],
            "allow_shell": bool(raw_input.get("allow_shell", default_args.get("allow_shell", False))),
        }
    if action_name == "EnterPlanMode":
        return {
            "technical_plan": str(
                raw_input.get("technical_plan")
                or raw_input.get("plan")
                or raw_input.get("debug_plan")
                or ""
            ).strip(),
            "risks": _clean_string_list(raw_input.get("risks"), -1, 500),
            "verification_commands": _clean_string_list(raw_input.get("verification_commands"), -1, 500),
            "assumptions": _clean_string_list(raw_input.get("assumptions"), -1, 500),
        }
    if action_name == "ExitPlanMode":
        return {
            "evaluation": str(raw_input.get("evaluation") or "").strip(),
            "approved": _as_bool(raw_input.get("approved", False)),
            "remaining_uncertainties": _clean_string_list(
                raw_input.get("remaining_uncertainties")
                or data.get("uncertainty_questions"), -1, 500
            ),
            "next_step": str(raw_input.get("next_step") or "").strip(),
        }
    return {}


def _candidate_specs(
    legal_specs: list[ActionSpec],
    candidate_actions: list[str] | None,
) -> list[ActionSpec]:
    legal_by_name = {spec.name: spec for spec in legal_specs}
    names = [spec.name for spec in legal_specs] if candidate_actions is None else candidate_actions
    specs: list[ActionSpec] = []
    for name in _dedupe(str(item).strip() for item in names if str(item).strip()):
        spec = legal_by_name.get(name)
        if spec is not None:
            specs.append(spec)
    return specs


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in result:
            result.append(item)
    return result


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_bounded_int(value: Any, *, minimum: int, maximum: int | None = None) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _clean_purpose(value: Any) -> str:
    purpose = str(value or "diagnostic").strip().lower()
    if purpose not in {"verification", "diagnostic", "search", "build"}:
        return "diagnostic"
    return purpose
