"""LLM-backed action policy with constrained action selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ext.file_requirements import full_read_requirements
from ext.focus_files import current_focus_files, execution_target_files
from agent_runtime.actions import ActionFactory
from agent_runtime.lifecycle.execution_queue import current_execution_item
from agent_runtime.llm.llm_nodes import LLMJsonNode, publish_user_update
from ext.tool_summaries import read_file_summaries
from agent_runtime.rl.action_space import ActionSpace
from agent_runtime.rl.state_encoder import StateEncoder
from config import LLMConfig
from loguru import logger
from model.agent.actions import Action, ActionSpec
from model.agent.graph import AgentState
from model.llm import ActionChoiceResponse, GuardDecision
from prompts.templates import load_prompt, render_prompt
from utils import _truncate_text, _clamp_float, _clean_string_list


class ActionDecisionError(RuntimeError):
    """Raised when the LLM policy cannot produce an executable legal action."""


@dataclass
class LLMActionPolicy:
    """
        llm 执行决策层
    """
    llm_config: LLMConfig
    action_space: ActionSpace
    action_factory: ActionFactory | None = None
    q_table: dict[str, dict[str, float]] = field(default_factory=dict)
    encoder: StateEncoder | None = None
    q_top_k: int = 3
    deny_threshold: float = -0.5
    MAX_DECISION_ATTEMPTS = 2

    def __post_init__(self) -> None:
        self.encoder = self.encoder or StateEncoder()
        self.action_factory = self.action_factory or ActionFactory()
        self.node = LLMJsonNode(
            name="action_policy",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/action_policy.md"),
            build_prompt=_action_node_prompt,
            fallback=None,
            response_model=ActionChoiceResponse,
            normalize=_normalize_action_choice,
            raise_on_error=True,
        )

    def next_action(self, state: AgentState) -> Action:
        """
            决策下一个动作
        """
        assert self.action_factory is not None
        legal_specs = self.action_space.legal_specs(state)
        if not legal_specs:
            return self.action_factory.create(
                "finish",
                thought="LLM policy found no legal actions.",
            )

        qtable_context = self._guard(legal_specs, state)
        legal_by_name = {spec.name: spec for spec in legal_specs}
        logger.bind(task_id=state.get("task_id")).info(
            "llm action policy requested model={} provider={} legal_actions={} qtable_hard_denied={}",
            self.llm_config.model,
            self.llm_config.provider,
            [spec.name for spec in legal_specs],
            qtable_context.hard_denied,
        )
        decision_feedback: dict[str, Any] = {}
        last_error = "LLM did not produce a valid action."
        # 最多重读执行一次 action
        for attempt in range(1, self.MAX_DECISION_ATTEMPTS + 1):
            data = self.node.run(
                state,
                {
                    "legal_specs": legal_specs,
                    "guard": qtable_context,
                    "decision_feedback": decision_feedback,
                },
                publish_update=False,
            )
            action_name = str(data.get("action", "")).strip()
            reason = str(data.get("reason", "")).strip()
            guard = self._guard(legal_specs, state, action_name)
            selected_spec = legal_by_name.get(action_name)
            if selected_spec is None or action_name not in guard.allow_list:
                rejection_reason = (
                    "illegal_action"
                    if action_name not in legal_by_name
                    else "qtable_hard_denied"
                    if action_name in guard.hard_denied
                    else "guard_rejected"
                )
                last_error = (
                    f"Selected action `{action_name or '<empty>'}` was rejected: "
                    f"{rejection_reason}."
                )
                decision_feedback = {
                    "rejected_action": action_name,
                    "reason": rejection_reason,
                    "legal_actions": list(legal_by_name),
                }
                logger.bind(task_id=state.get("task_id")).warning(
                    "llm action decision rejected attempt={}/{} selected={} reason={}",
                    attempt,
                    self.MAX_DECISION_ATTEMPTS,
                    action_name,
                    rejection_reason,
                )
                continue

            default_args = self.action_factory.default_args(selected_spec, state)
            action_args = default_args
            if selected_spec.name in {
                "apply_code_patch",
                "request_user_input",
                "read_file",
                "search_text",
                "run_shell_command",
                "EnterPlanMode",
                "ExitPlanMode",
            }:
                action_args = self.action_factory.resolve_args(
                    selected_spec,
                    state,
                    proposed_args=_raw_action_input(data),
                    default_args=default_args,
                    reason=reason,
                    confidence=data.get("confidence"),
                    uncertainty_questions=data.get("uncertainty_questions"),
                )
            missing_required = self.action_factory.missing_required_args(
                state,
                selected_spec.name,
                action_args,
            )
            if missing_required:
                last_error = (
                    f"Selected action `{selected_spec.name}` is missing required arguments: "
                    f"{', '.join(missing_required)}."
                )
                decision_feedback = {
                    "rejected_action": selected_spec.name,
                    "reason": "missing_required_arguments",
                    "missing_required_arguments": missing_required,
                    "legal_actions": list(legal_by_name),
                }
                logger.bind(task_id=state.get("task_id")).warning(
                    "llm action decision rejected attempt={}/{} selected={} missing_required={}",
                    attempt,
                    self.MAX_DECISION_ATTEMPTS,
                    selected_spec.name,
                    missing_required,
                )
                continue

            publish_user_update(state, "action_policy", data.get("user_update"))
            return self.action_factory.build(
                selected_spec,
                state,
                resolved_args=action_args,
                thought=reason or self.action_factory.default_thought(selected_spec.name, action_args),
                metadata={
                    "llm_guard": {
                        "selected_action": selected_spec.name,
                        "guard": guard.to_dict(),
                    },
                },
            )

        raise ActionDecisionError(
            f"LLM failed to produce a valid action after {self.MAX_DECISION_ATTEMPTS} attempts. "
            f"{last_error}"
        )

    def _guard(
        self,
        legal_specs: list[ActionSpec],
        state: AgentState,
        selected_action: str | None = None,
    ) -> GuardDecision:
        """
            截断 reward 低于 threshold 的 action
            并记录到上下文中
        """
        assert self.encoder is not None
        state_key = self.encoder.encode(state).key
        q_values = self.q_table.get(state_key, {})
        legal_actions = [spec.name for spec in legal_specs]
        if selected_action is None:
            selected_specs = legal_specs
        else:
            selected_specs = [spec for spec in legal_specs if spec.name == selected_action]
        selected_names = [spec.name for spec in selected_specs]
        if not selected_specs:
            return GuardDecision(
                state_key=state_key,
                q_values={name: float(value) for name, value in q_values.items()},
                legal_actions=legal_actions,
                hard_denied={},
                allow_list=[],
                allow_scores={},
            )
        if not q_values:
            allowed = selected_names[: self.q_top_k]
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
        for spec in selected_specs:
            score = float(q_values.get(spec.name, 0.0))
            if score <= self.deny_threshold:
                hard_denied[spec.name] = score
            else:
                passed_specs.append((spec, score))

        # Q-table only rejects the selected action; it never substitutes another action.
        allowed_scored = passed_specs[: self.q_top_k]
        decision = GuardDecision(
            state_key=state_key,
            q_values={name: float(value) for name, value in q_values.items()},
            legal_actions=legal_actions,
            hard_denied=hard_denied,
            allow_list=[spec.name for spec, _ in allowed_scored],
            allow_scores={spec.name: score for spec, score in allowed_scored},
        )
        logger.bind(task_id=state.get("task_id")).debug(
            "llm qtable guard decision={}",
            decision.to_dict(),
        )
        return decision

def _action_node_prompt(
    state: AgentState,
    context: dict[str, Any],
) -> str:
    legal_specs = context.get("legal_specs") or []
    guard = context.get("guard")
    if not isinstance(guard, GuardDecision):
        guard = GuardDecision(
            state_key="",
            q_values={},
            legal_actions=[getattr(spec, "name", "") for spec in legal_specs],
            hard_denied={},
            allow_list=[getattr(spec, "name", "") for spec in legal_specs],
            allow_scores={},
        )
    return _action_prompt(
        state,
        legal_specs,
        guard,
        decision_feedback=context.get("decision_feedback") or {},
    )


def _normalize_action_choice(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    del state, context
    action = str(data.get("action") or "").strip()
    if not action:
        raise ActionDecisionError("LLM action response did not include an action")
    reason = str(data.get("reason") or "").strip()
    raw_action_input = data.get("action_input")
    if not isinstance(raw_action_input, dict):
        raise ActionDecisionError("LLM action response action_input must be an object")
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
        0.5,
        "invalid confidence",
    )
    logger.debug(f"confidence: {confidence}")
    return {
        "action": action,
        "reason": reason,
        "action_input": raw_action_input,
        "uncertainty_questions": uncertainty_questions,
        "confidence": confidence,
    }


def _action_prompt(
    state: AgentState,
    legal_specs: list[ActionSpec],
    guard: GuardDecision,
    *,
    decision_feedback: dict[str, Any],
) -> str:
    """
        构造每个 action 的 prompt 格式
    """
    legal = _compact_legal_actions(state, legal_specs)
    constraints = _action_constraints(state, guard, legal_specs)
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
        technical_plan=_truncate_text(str(state.get("technical_plan", "")), 5000),
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
        goal_contract=_truncate_text(
            json.dumps(state.get("goal_contract", {}), ensure_ascii=False, default=str),
            1600,
        ),
        progress_ledger=_truncate_text(
            json.dumps(state.get("progress_ledger", {}), ensure_ascii=False, default=str),
            2200,
        ),
        next_obligation=_truncate_text(
            json.dumps(state.get("next_obligation", {}), ensure_ascii=False, default=str),
            1000,
        ),
        verification_capabilities=_truncate_text(
            json.dumps(state.get("verification_capabilities", {}), ensure_ascii=False, default=str),
            1800,
        ),
        attention_focus=_truncate_text(
            json.dumps(state.get("attention_focus", {}), ensure_ascii=False, default=str),
            1800,
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
        pending_resolution=json.dumps(
            state.get("pending_resolution", {}),
            ensure_ascii=False,
            default=str,
        ),
        memory_context=str(state.get("memory_context", ""))[:2500],
        compressed_context=str(state.get("compressed_context", ""))[:2500],
        legal_actions=json.dumps(legal, ensure_ascii=False),
        action_constraints=json.dumps(constraints, ensure_ascii=False, default=str),
        decision_feedback=json.dumps(decision_feedback, ensure_ascii=False, default=str),
    )


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
    legal_specs: list[ActionSpec],
) -> dict[str, Any]:
    return {
        "phase": _action_phase(state),
        "current_execution": current_execution_item(state) or {},
        "focus_files": current_focus_files(state, limit=4),
        "candidate_action_names": [spec.name for spec in legal_specs],
        "guard_allow_list": list(getattr(guard, "allow_list", []) or [])[:6],
        "plan_readiness": _plan_readiness(state),
    }


def _plan_readiness(state: AgentState) -> dict[str, Any]:
    required_reads = full_read_requirements(state)
    missing: list[str] = []
    if required_reads:
        missing.append("full_read_requirements")
    if bool(state.get("verification_required", True)) and not _verification_command_available(state):
        missing.append("verification_command")
    if bool(state.get("editing_enabled", False)) and not current_focus_files(state, limit=1):
        missing.append("patch_targets")
    return {
        "can_exit": bool(state.get("technical_plan")) and not missing,
        "missing": missing,
        "full_read_required_files": [
            str(item.get("file_path") or "").strip()
            for item in required_reads
            if isinstance(item, dict) and str(item.get("file_path") or "").strip()
        ],
        "has_verification_command": _verification_command_available(state),
        "has_patch_targets": bool(current_focus_files(state, limit=1)),
    }


def _verification_command_available(state: AgentState) -> bool:
    capabilities = state.get("verification_capabilities")
    if isinstance(capabilities, dict) and capabilities.get("allowed"):
        return True
    for item in state.get("plan_verification_commands", []) or []:
        if str(item or "").strip():
            return True
    for item in state.get("verification_commands", []) or []:
        if isinstance(item, dict):
            command = str(item.get("command") or "").strip()
        else:
            command = str(item or "").strip()
        if command:
            return True
    return False


def _action_phase(state: AgentState) -> str:
    phase = str(state.get("phase") or "").strip().lower()
    if phase:
        return phase
    if bool(state.get("plan_mode", False)):
        return "plan"
    if bool(state.get("verification_stale", False)):
        return "verify"
    if state.get("execution_queue"):
        return "execute"
    return "explore"




def _raw_action_input(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("action_input")
    return value if isinstance(value, dict) else {}


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
