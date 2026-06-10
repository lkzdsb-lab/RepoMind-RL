"""LLM-backed action policy with constrained action selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ext.file_requirements import (
    choose_read_file_target,
    full_read_requirements,
    is_full_read,
    recommended_read_file_args,
)
from ext.focus_files import current_focus_files, edited_files_needing_reread, execution_target_files
from agent_runtime.llm.llm_nodes import LLMJsonNode
from ext.tool_summaries import read_file_summaries
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
        if (
            selected_spec is not None
            and selected_spec.name == "request_user_input"
            and preferred_action
            and preferred_action != "request_user_input"
        ):
            provisional_args = _action_input_for(
                state,
                "request_user_input",
                data,
                reason,
                {},
            )
            if not provisional_args.get("questions"):
                preferred_spec = legal_by_name.get(preferred_action)
                if preferred_spec is not None:
                    selected_spec = preferred_spec
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
            action_args = _action_input_for(state, selected_spec.name, data, reason, action.args)
        missing_required = _missing_required_action_args(state, selected_spec.name, action_args)

        # 如果缺失参数，则记录缺失哪些参数
        if missing_required:
            message = (
                f"Selected action `{selected_spec.name}` is missing required arguments: "
                f"{', '.join(missing_required)}. Defer execution and resolve from repository context in the next step."
            )
            return Action(
                action.name,
                action_args,
                thought=(
                    f"工具 `{selected_spec.name}` 缺少必填参数 "
                    f"{', '.join(missing_required)}，先记录缺参并在下一步补全。"
                ),
                metadata={
                    "llm_guard": {
                        "selected_action": action.name,
                        "llm_candidate_actions": candidate_actions,
                        "guard": guard.to_dict(),
                        "missing_required_args_for": selected_spec.name,
                        "missing_required_args": missing_required,
                        "guard_bypassed": "missing_required_args",
                    },
                    "deferred_action": {
                        "action": selected_spec.name,
                        "missing_required_args": missing_required,
                        "partial_args": action_args,
                        "reason": reason,
                        "message": message,
                    },
                },
            )
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


def _action_node_prompt(
    state: AgentState,
    context: dict[str, Any],
) -> str:
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
    uncertainty_questions = _clean_string_list(
        data.get("uncertainty_questions")
        or raw_action_input.get("uncertainty_questions"),
        -1,
        None,
    )
    confidence_value = data.get("confidence")
    if confidence_value in (None, ""):
        confidence_value = raw_action_input.get("confidence")
    confidence = _clamp_float(
        confidence_value,
        float(fallback.get("confidence", 0.5)),
        "invalid confidence",
    )
    logger.debug(f"confidence: {confidence}")
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
    legal = _compact_legal_actions(state, legal_specs)
    constraints = _action_constraints(state, guard, fallback_action, legal_specs)
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
            1800,
        ),
        selected_code_context_summary=_truncate_text(
            json.dumps(_selected_code_context_summary(state), ensure_ascii=False, default=str),
            2200,
        ),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
        read_files=json.dumps(
            read_file_summaries(state, limit=6, excerpt_chars=900),
            ensure_ascii=False,
            default=str,
        ),
        full_read_requirements=json.dumps(
            full_read_requirements(state),
            ensure_ascii=False,
            default=str,
        ),
        test_results=json.dumps(state.get("test_results", [])[-2:], ensure_ascii=False, default=str),
        patch_summary=state.get("patch_summary"),
        editing_enabled=json.dumps(bool(state.get("editing_enabled", False))),
        edit_results=json.dumps(state.get("edit_results", [])[-2:], ensure_ascii=False, default=str),
        user_inputs=json.dumps(state.get("user_inputs", []), ensure_ascii=False, default=str),
        pending_action_requirements=json.dumps(
            state.get("pending_action_requirements", {}),
            ensure_ascii=False,
            default=str,
        ),
        memory_context=str(state.get("memory_context", ""))[:2500],
        compressed_context=str(state.get("compressed_context", ""))[:2500],
        legal_actions=json.dumps(legal, ensure_ascii=False),
        action_constraints=json.dumps(constraints, ensure_ascii=False, default=str),
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
    return _clean_string_list(candidates, -1, None)


def _selected_code_context_summary(state: AgentState) -> dict[str, Any]:
    context = state.get("selected_code_context")
    if not isinstance(context, dict) or not context:
        context = state.get("code_context")
    if not isinstance(context, dict):
        return {}
    return {
        "files": [
            str(item.get("path") or "")
            for item in (context.get("files") or [])[:6]
            if isinstance(item, dict) and str(item.get("path") or "")
        ],
        "functions": [
            {
                "name": str(item.get("full_name") or item.get("name") or ""),
                "file_path": str(item.get("file_path") or ""),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
            }
            for item in (context.get("functions") or [])[:6]
            if isinstance(item, dict)
        ],
        "symbols": [
            {
                "name": str(item.get("name") or ""),
                "file_path": str(item.get("file_path") or ""),
                "line": item.get("line"),
            }
            for item in (context.get("symbols") or [])[:8]
            if isinstance(item, dict)
        ],
    }


def _compact_legal_actions(state: AgentState, legal_specs: list[ActionSpec]) -> list[dict[str, Any]]:
    """ 缩短 legal action 的长度"""
    manifest_by_name = {
        str(item.get("name")): item
        for item in state.get("tool_manifest", [])
        if isinstance(item, dict) and item.get("name")
    }
    focus = current_focus_files(state, limit=3)
    phase = _action_phase(state)
    legal: list[dict[str, Any]] = []
    for spec in legal_specs:
        manifest = dict(manifest_by_name.get(spec.name, {}))
        item: dict[str, Any] = {
            "name": spec.name,
            "description": (spec.description or str(manifest.get("description") or "")).strip()[:180],
        }
        required_fields = _required_action_fields(state, spec.name)
        if required_fields:
            item["required_fields"] = required_fields
        permissions = [
            str(permission).strip()
            for permission in manifest.get("permissions", []) or []
            if str(permission).strip()
        ]
        if permissions:
            item["permissions"] = permissions
        note = _legal_action_note(spec.name, state, phase, focus)
        if note:
            item["note"] = note
        legal.append(item)
    return legal


def _legal_action_note(action_name: str, state: AgentState, phase: str, focus_files: list[str]) -> str:
    if action_name == "read_file":
        if focus_files:
            return f"Prefer focus files first: {', '.join(focus_files[:3])}."
        return "Use only when exact source text is still needed."
    if action_name == "apply_code_patch":
        return "Patch targets must be full_read. Prefer exact anchors from read_files."
    if action_name == "run_shell_command":
        if bool(state.get("verification_stale", False)):
            return "Use for narrow verification of the latest edits."
        return "Use for diagnostic or verification commands only."
    if action_name == "EnterPlanMode":
        return "Use before code changes after enough local evidence has been collected."
    if action_name == "ExitPlanMode":
        return "Use only when the technical plan is concrete and remaining uncertainty is empty."
    if action_name == "request_user_input":
        return "Ask only concrete unresolved questions."
    if action_name == "finish":
        return f"Only valid when current {phase} work is complete."
    return ""


def _action_constraints(
    state: AgentState,
    guard: GuardDecision,
    fallback_action: Action,
    legal_specs: list[ActionSpec],
) -> dict[str, Any]:
    return {
        "phase": _action_phase(state),
        "focus_files": current_focus_files(state, limit=4),
        "candidate_action_names": [spec.name for spec in legal_specs],
        "guard_allow_list": list(getattr(guard, "allow_list", []) or [])[:6],
        "fallback_action": fallback_action.name,
    }


def _action_phase(state: AgentState) -> str:
    if bool(state.get("plan_mode", False)):
        return "plan"
    if bool(state.get("verification_stale", False)):
        return "verify"
    if state.get("execution_queue"):
        return "execute"
    return "explore"




def _action_input_for(
    state: AgentState,
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
        requested_path = str(
            raw_input.get("file_path") or default_args.get("file_path") or ""
        ).strip()
        file_path = _normalize_read_file_target(
            state,
            requested_path=requested_path,
            default_path=str(default_args.get("file_path") or "").strip(),
        )
        recommended = recommended_read_file_args(
            state,
            file_path,
            requested_max_chars=raw_input.get("max_chars", default_args.get("max_chars", 8000)),
            default_max_chars=int(default_args.get("max_chars", 8000) or 8000),
        )
        return {
            "file_path": file_path,
            "max_chars": _bounded_int(
                recommended.get("max_chars"),
                default=int(recommended.get("max_chars") or 8000),
                minimum=1,
                maximum=200000,
            ),
            "start_line": _optional_bounded_int(raw_input.get("start_line"), minimum=1),
            "end_line": _optional_bounded_int(raw_input.get("end_line"), minimum=1),
        }
    if action_name == "search_text":
        pattern = str(
            raw_input.get("pattern")
            or raw_input.get("query")
            or default_args.get("pattern")
            or default_args.get("query")
            or ""
        ).strip()
        return {
            "pattern": pattern,
            "regex": bool(raw_input.get("regex", default_args.get("regex", True))),
            "globs": _clean_string_list(raw_input.get("globs") or default_args.get("globs"), -1, 500),
            "context_lines": _bounded_int(
                raw_input.get("context_lines", default_args.get("context_lines", 0)),
                default=0,
                minimum=0,
                maximum=5,
            ),
            "max_results": _bounded_int(
                raw_input.get("max_results", default_args.get("max_results", 50)),
                default=50,
                minimum=1,
                maximum=200,
            ),
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
                or default_args.get("technical_plan")
                or default_args.get("plan")
                or ""
            ).strip(),
            "risks": _clean_string_list(raw_input.get("risks") or default_args.get("risks"), -1, 500),
            "verification_commands": _clean_string_list(
                raw_input.get("verification_commands") or default_args.get("verification_commands"),
                -1,
                500,
            ),
            "assumptions": _clean_string_list(
                raw_input.get("assumptions") or default_args.get("assumptions"),
                -1,
                500,
            ),
        }
    if action_name == "ExitPlanMode":
        return {
            "evaluation": str(raw_input.get("evaluation") or default_args.get("evaluation") or "").strip(),
            "approved": _as_bool(raw_input.get("approved", default_args.get("approved", False))),
            "remaining_uncertainties": _clean_string_list(
                raw_input.get("remaining_uncertainties")
                or data.get("uncertainty_questions")
                or default_args.get("remaining_uncertainties"),
                -1,
                500,
            ),
            "next_step": str(raw_input.get("next_step") or default_args.get("next_step") or "").strip(),
        }
    return {}


def _candidate_specs(
    legal_specs: list[ActionSpec],
    candidate_actions: list[str] | None,
) -> list[ActionSpec]:
    legal_by_name = {spec.name: spec for spec in legal_specs}
    names = [spec.name for spec in legal_specs] if candidate_actions is None else candidate_actions
    specs: list[ActionSpec] = []
    for name in _clean_string_list(names, -1, None):
        spec = legal_by_name.get(name)
        if spec is not None:
            specs.append(spec)
    return specs


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


def _missing_required_action_args(
    state: AgentState,
    action_name: str,
    action_args: dict[str, Any],
) -> list[str]:
    """
        如果缺参数则向 llm 提示
    """
    required = _required_action_fields(state, action_name)
    missing: list[str] = []
    for field in required:
        value = action_args.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
            continue
        if isinstance(value, list) and not value:
            missing.append(field)
    return missing


def _required_action_fields(state: AgentState, action_name: str) -> list[str]:
    for item in state.get("tool_manifest", []):
        if not isinstance(item, dict) or str(item.get("name") or "") != action_name:
            continue
        schema = item.get("input_schema")
        if not isinstance(schema, dict):
            return []
        required = schema.get("required")
        if not isinstance(required, list):
            return []
        return [str(field).strip() for field in required if str(field).strip()]
    return []


def _normalize_read_file_target(
    state: AgentState,
    requested_path: str,
    default_path: str,
) -> str:
    """
        获取下一个读取的文件
    """
    execution_targets = execution_target_files(state)
    required_target = choose_read_file_target(
        state,
        requested_path=requested_path,
        default_path=default_path,
    )
    if required_target and not is_full_read(state, required_target):
        return required_target
    candidate_files = [
        str(path).strip()
        for path in state.get("candidate_files", []) or []
        if str(path).strip()
    ]
    if execution_targets:
        if requested_path and requested_path in execution_targets:
            return requested_path
        if default_path and default_path in execution_targets:
            return default_path
    reread = edited_files_needing_reread(state)
    if reread:
        if requested_path and requested_path in reread:
            return requested_path
        if default_path and default_path in reread:
            return default_path
        return reread[0]
    read_files = _state_read_files(state)
    unread = [path for path in candidate_files if path not in read_files]
    if unread:
        scoped_unread = [path for path in unread if path in execution_targets] if execution_targets else []
        if scoped_unread:
            if requested_path and requested_path in scoped_unread:
                return requested_path
            if default_path and default_path in scoped_unread:
                return default_path
            return scoped_unread[0]
        if requested_path and requested_path in unread:
            return requested_path
        if default_path and default_path in unread:
            return default_path
        return unread[0]
    return requested_path or default_path


def _state_read_files(state: AgentState) -> set[str]:
    cache = state.get("read_file_cache")
    if isinstance(cache, dict) and cache:
        return {
            str(path).strip()
            for path in cache.keys()
            if str(path).strip()
        }
    files: set[str] = set()
    # 若 cache 没有则直接从 observation 里拿
    for observation in state.get("observations", []):
        if not isinstance(observation, dict):
            continue
        content = observation.get("content", {})
        if isinstance(content, dict) and content.get("file_path"):
            files.add(str(content.get("file_path")))
            continue
        if observation.get("type") == "tool_output" and isinstance(content, dict):
            payload = content.get("content", {})
            if isinstance(payload, dict) and payload.get("file_path"):
                files.add(str(payload.get("file_path")))
    return files
