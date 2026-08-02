"""Build executable actions from policy selections."""

from __future__ import annotations

from typing import Any

from agent_runtime.search_query import SearchQueryPlanner
from agent_runtime.verification.capabilities import recommended_verification_command
from ext.plan_mode_args import (
    build_enter_plan_mode_args,
    build_exit_plan_mode_args,
    collect_plan_focus_files,
)
from model.agent.actions import Action, ActionSpec
from model.agent.graph import AgentState
from utils import _as_bool, _clean_string_list, _is_informative_query


class ActionFactory:
    """Materialize a selected action without deciding which action to select."""

    def __init__(self, query_planner: SearchQueryPlanner | None = None) -> None:
        self.query_planner = query_planner or SearchQueryPlanner()

    def build(
        self,
        spec: ActionSpec,
        state: AgentState,
        *,
        proposed_args: dict[str, Any] | None = None,
        resolved_args: dict[str, Any] | None = None,
        reason: str = "",
        uncertainty_questions: Any = None,
        thought: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        """ 根据上下文构建参数正确的 action"""
        if resolved_args is not None:
            args = resolved_args
        else:
            default_args = self.default_args(spec, state)
            args = default_args
        if resolved_args is None and proposed_args is not None:
            args = self.resolve_args(
                spec,
                state,
                proposed_args=proposed_args,
                default_args=default_args,
                reason=reason,
                uncertainty_questions=uncertainty_questions,
            )
        return self.create(
            spec.name,
            args=args,
            thought=thought or reason or self.default_thought(spec.name, args),
            metadata=metadata,
        )

    def create(
        self,
        name: str,
        *,
        args: dict[str, Any] | None = None,
        thought: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        """Create an action when no ActionSpec exists, such as policy fallback paths."""
        return Action(name=name, args=args or {}, thought=thought, metadata=metadata or {})

    def resolve_args(
        self,
        spec: ActionSpec,
        state: AgentState,
        *,
        proposed_args: dict[str, Any],
        default_args: dict[str, Any] | None = None,
        reason: str = "",
        uncertainty_questions: Any = None,
    ) -> dict[str, Any]:
        """ 根据 action 提取需要的参数"""
        raw_input = proposed_args if isinstance(proposed_args, dict) else {}
        defaults = default_args if default_args is not None else self.default_args(spec, state)
        action_name = spec.name

        if action_name == "apply_code_patch":
            return {
                "changes": raw_input.get("changes", []),
                "assumptions": _clean_string_list(raw_input.get("assumptions"), -1, 500),
                "dry_run": bool(raw_input.get("dry_run", False)),
            }
        if action_name == "request_user_input":
            questions = _clean_string_list(
                raw_input.get("questions") or uncertainty_questions,
                -1,
                500,
            )
            return {
                "reason": str(raw_input.get("reason") or reason or "").strip(),
                "questions": questions[:3],
            }
        if action_name == "read_file":
            requested_path = str(raw_input.get("file_path") or "").strip()
            file_path = self._normalize_read_file_target(
                state,
                requested_path=requested_path,
                default_path="",
            )
            max_chars = raw_input.get("max_chars", defaults.get("max_chars", 8000))
            return {
                "file_path": file_path,
                "max_chars": _bounded_int(
                    max_chars,
                    default=int(defaults.get("max_chars", 8000) or 8000),
                    minimum=1,
                    maximum=200000,
                ),
                "start_line": _optional_bounded_int(raw_input.get("start_line"), minimum=1),
                "end_line": _optional_bounded_int(raw_input.get("end_line"), minimum=1),
            }
        if action_name == "search_code_context":
            return {
                "query": str(raw_input.get("query") or "").strip(),
                "max_results": _bounded_int(
                    raw_input.get("max_results", 10),
                    default=10,
                    minimum=1,
                    maximum=100,
                ),
            }
        if action_name == "search_text":
            pattern = str(
                raw_input.get("pattern")
                or ""
            ).strip()
            return {
                "pattern": pattern,
                "regex": bool(raw_input.get("regex", defaults.get("regex", True))),
                "globs": _clean_string_list(
                    raw_input.get("globs") or defaults.get("globs"), -1, 500
                ),
                "context_lines": _bounded_int(
                    raw_input.get("context_lines", defaults.get("context_lines", 0)),
                    default=0,
                    minimum=0,
                    maximum=5,
                ),
                "max_results": _bounded_int(
                    raw_input.get("max_results", defaults.get("max_results", 50)),
                    default=50,
                    minimum=1,
                    maximum=200,
                ),
            }
        if action_name == "run_tests":
            return {
                "command": str(raw_input.get("command") or "").strip(),
                "timeout": _bounded_int(
                    raw_input.get("timeout", 120),
                    default=120,
                    minimum=1,
                    maximum=1800,
                ),
            }
        if action_name == "run_shell_command":
            return {
                "command": str(raw_input.get("command") or "").strip(),
                "purpose": _clean_purpose(raw_input.get("purpose") or defaults.get("purpose")),
                "verification_kind": _clean_verification_kind(
                    raw_input.get("verification_kind")
                    or defaults.get("verification_kind")
                ),
                "timeout": _bounded_int(
                    raw_input.get("timeout", defaults.get("timeout", 120)),
                    default=120,
                    minimum=1,
                    maximum=1800,
                ),
                "reason": str(
                    raw_input.get("reason") or reason or defaults.get("reason") or ""
                ).strip()[:500],
                "allow_shell": bool(
                    raw_input.get("allow_shell", defaults.get("allow_shell", False))
                ),
            }
        if action_name == "EnterPlanMode":
            return {
                "technical_plan": str(
                    raw_input.get("technical_plan")
                    or defaults.get("technical_plan")
                    or ""
                ).strip(),
                "risks": _clean_string_list(
                    raw_input.get("risks") or defaults.get("risks"), -1, 500
                ),
                "verification_commands": _clean_string_list(
                    raw_input.get("verification_commands") or defaults.get("verification_commands"),
                    -1,
                    500,
                ),
                "assumptions": _clean_string_list(
                    raw_input.get("assumptions") or defaults.get("assumptions"), -1, 500
                ),
            }
        if action_name == "ExitPlanMode":
            return {
                "evaluation": str(
                    raw_input.get("evaluation") or defaults.get("evaluation") or ""
                ).strip(),
                "approved": _as_bool(raw_input.get("approved", defaults.get("approved", False))),
                "remaining_uncertainties": _clean_string_list(
                    raw_input.get("remaining_uncertainties")
                    or uncertainty_questions
                    or defaults.get("remaining_uncertainties"),
                    -1,
                    500,
                ),
                "next_step": str(
                    raw_input.get("next_step") or defaults.get("next_step") or ""
                ).strip(),
            }
        return defaults

    def invalid_args(self, action_name: str, action_args: dict[str, Any]) -> list[str]:
        invalid: list[str] = []
        if action_name == "search_code_context" and not _is_informative_query(
            action_args.get("query")
        ):
            invalid.append("query must contain a searchable word, identifier, filename, or number")
        if action_name == "search_text" and not _is_informative_query(
            action_args.get("pattern")
        ):
            invalid.append("pattern must contain searchable content, not punctuation alone")
        return invalid

    def default_args(self, spec: ActionSpec, state: AgentState) -> dict[str, Any]:
        action_name = spec.name
        if action_name == "search_code_context":
            query_plan = self.query_planner.plan(state)
            return {"query": query_plan.query, "query_plan": query_plan.to_dict()}
        if action_name == "read_file":
            file_path = self._default_read_file_target(state)
            return {
                "file_path": file_path,
                "max_chars": 8000,
                "start_line": None,
                "end_line": None,
            }
        if action_name == "run_tests":
            return {"command": recommended_verification_command(state)}
        if action_name == "search_text":
            query = self.query_planner.plan(state).query
            return {
                "pattern": query,
                "regex": True,
                "globs": [],
                "context_lines": 0,
                "max_results": 50,
            }
        if action_name == "run_shell_command":
            return {
                "command": self._default_execution_command(state),
                "purpose": "verification",
                "verification_kind": "standard",
                "timeout": 120,
                "reason": "Verify the latest code changes before finishing.",
                "allow_shell": False,
            }
        if action_name == "EnterPlanMode":
            return build_enter_plan_mode_args(
                state,
                query=self.query_planner.plan(state).query,
                focus_files=collect_plan_focus_files(state),
                read_files=sorted(_state_read_files(state)),
                default_verification_command=recommended_verification_command(state),
            )
        if action_name == "ExitPlanMode":
            return build_exit_plan_mode_args(
                state,
                focus_files=collect_plan_focus_files(state),
                read_files=sorted(_state_read_files(state)),
                default_verification_command=recommended_verification_command(state),
            )
        return {}

    def missing_required_args(
        self,
        state: AgentState,
        action_name: str,
        action_args: dict[str, Any],
    ) -> list[str]:
        """ 缺失参数时处理"""
        missing: list[str] = []
        for field in _required_action_fields(state, action_name):
            value = action_args.get(field)
            if value is None or isinstance(value, str) and not value.strip():
                missing.append(field)
            elif isinstance(value, list) and not value:
                missing.append(field)
        return missing

    def default_thought(self, action_name: str, args: dict[str, Any]) -> str:
        """ 兜底"""
        if action_name == "search_code_context":
            return f"Policy selected structured code context search: `{args.get('query', '')}`."
        if action_name == "read_file":
            return f"Policy selected file read: `{args.get('file_path', '')}`."
        if action_name in {"run_tests", "run_shell_command"}:
            return f"Policy selected verification command: `{args.get('command', '')}`."
        if action_name == "search_text":
            return f"Policy selected text search: `{args.get('pattern', '')}`."
        return {
            "list_files": "Policy selected repository structure inspection.",
            "EnterPlanMode": "Policy selected plan construction.",
            "ExitPlanMode": "Policy selected plan evaluation.",
            "apply_code_patch": "Policy selected a guarded code patch.",
            "request_user_input": "Policy selected a clarification request.",
            "git_diff": "Policy selected current diff inspection.",
            "finish": "Policy selected task completion.",
        }.get(action_name, f"Policy selected `{action_name}`.")

    def _default_read_file_target(self, state: AgentState) -> str:
        candidates = [
            str(path).strip()
            for path in state.get("candidate_files", []) or []
            if str(path).strip()
        ]
        unread = [
            path
            for path in candidates
            if path not in _state_read_files(state)
        ]
        targets = unread or list(state.get("edited_files", []) or []) or candidates
        return str(targets[0]).strip() if targets else ""

    def _default_execution_command(self, state: AgentState) -> str:
        return recommended_verification_command(state)

    def _normalize_read_file_target(
        self,
        state: AgentState,
        *,
        requested_path: str,
        default_path: str,
    ) -> str:
        pending_resolution = state.get("pending_resolution") or {}
        if str(pending_resolution.get("kind") or "") == "recovery":
            recovery_file = str(pending_resolution.get("target_file") or "").strip()
            if recovery_file:
                return recovery_file

        return requested_path or default_path


def _required_action_fields(state: AgentState, action_name: str) -> list[str]:
    for item in state.get("tool_manifest", []):
        if not isinstance(item, dict) or str(item.get("name") or "") != action_name:
            continue
        schema = item.get("input_schema")
        required = schema.get("required") if isinstance(schema, dict) else None
        if not isinstance(required, list):
            return []
        return [str(field).strip() for field in required if str(field).strip()]
    return []


def _state_read_files(state: AgentState) -> set[str]:
    cache = state.get("read_file_cache")
    if isinstance(cache, dict) and cache:
        return {
            str(path).strip()
            for path, snapshot in cache.items()
            if str(path).strip()
            and isinstance(snapshot, dict)
            and (snapshot.get("spans") or snapshot.get("is_empty"))
        }
    files: set[str] = set()
    for observation in state.get("observations", []):
        if not isinstance(observation, dict):
            continue
        content = observation.get("content", {})
        if isinstance(content, dict) and content.get("file_path"):
            files.add(str(content["file_path"]))
            continue
        if observation.get("type") == "tool_output" and isinstance(content, dict):
            payload = content.get("content", {})
            if isinstance(payload, dict) and payload.get("file_path"):
                files.add(str(payload["file_path"]))
    return files


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_bounded_int(
    value: Any,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    parsed = max(minimum, parsed)
    return min(maximum, parsed) if maximum is not None else parsed


def _clean_purpose(value: Any) -> str:
    purpose = str(value or "diagnostic").strip().lower()
    return purpose if purpose in {"verification", "diagnostic", "search", "build"} else "diagnostic"


def _clean_verification_kind(value: Any) -> str:
    kind = str(value or "standard").strip().lower()
    return kind if kind in {"standard", "smoke"} else "standard"
