"""Capability- and safety-oriented action space.

The action space deliberately does not encode diagnose/implement/verify order.
The LLM owns semantic sequencing; deterministic tool guards enforce safety.
"""

from __future__ import annotations

from typing import Any, Iterable

from agent_runtime.lifecycle.completion import derive_phase
from agent_runtime.search_query import SearchQueryPlanner
from model.agent.actions import ActionSpec
from model.agent.graph import AgentState


class ActionSpace:
    def __init__(self, action_names: Iterable[str] | None = None) -> None:
        self.query_planner = SearchQueryPlanner()
        self._last_limit_events: list[dict[str, Any]] = []
        self.action_names = list(
            action_names
            or [
                "list_files",
                "search_code_context",
                "search_text",
                "read_file",
                "EnterPlanMode",
                "ExitPlanMode",
                "apply_code_patch",
                "request_user_input",
                "run_tests",
                "run_shell_command",
                "git_diff",
                "finish",
            ]
        )

    def legal_specs(self, state: AgentState) -> list[ActionSpec]:
        self._last_limit_events = []
        status = str(state.get("status") or "")
        if status in {"finished", "failed"}:
            return [ActionSpec("finish", "Return the terminal run result.")]

        available = self._available_action_names(state)
        specs: list[ActionSpec] = []
        plan_mode = bool(state.get("plan_mode", False))
        awaiting_user = status == "awaiting_user_input"

        if awaiting_user:
            if "request_user_input" in available:
                specs.append(ActionSpec("request_user_input", "Wait for required user information."))
            return specs

        # 通用 specs
        self._append(specs, available, "list_files", "Inspect repository structure.")
        self._append(specs, available, "search_code_context", "Search structured repository context.")
        self._append(specs, available, "search_text", "Search repository text or symbols.")
        self._append(specs, available, "read_file", "Read the file or line range needed for the next decision.")
        self._append(specs, available, "run_shell_command", "Run an allowed diagnostic or verification command.")
        self._append(specs, available, "run_tests", "Run an allowed project verification command.")
        self._append(specs, available, "request_user_input", "Ask a concrete question only when repository tools cannot resolve it.")

        if plan_mode:
            self._append(specs, available, "ExitPlanMode", "Exit Plan Mode once the implementation plan is actionable.")
            return self._dedupe(specs)

        intent = _task_intent(state)
        can_edit = (
            bool(state.get("editing_enabled", False))
            and intent == "implement"
            and _has_usable_read_cache(state)
        )
        if can_edit and not bool(state.get("plan_mode_approved", False)):
            self._append(specs, available, "EnterPlanMode", "Create an implementation plan before editing.")
        if can_edit and bool(state.get("plan_mode_approved", False)):
            self._append(
                specs,
                available,
                "apply_code_patch",
                "Apply an exact replacement grounded in source read during this run.",
            )

        if bool(state.get("is_git_repo", True)):
            self._append(specs, available, "git_diff", "Inspect the current repository diff.")
        self._append(specs, available, "finish", "Ask the completion judge whether the current task is complete.")
        return self._dedupe(specs)

    def consume_last_limit_events(self) -> list[dict[str, Any]]:
        events = list(self._last_limit_events)
        self._last_limit_events = []
        return events

    def extract_keyword(self, state: AgentState) -> str:
        return self.query_planner.plan(state).query

    def _available_action_names(self, state: AgentState) -> set[str]:
        """ 获取能够使用的 action"""
        configured = set(self.action_names)
        manifest = {
            str(item.get("name") or "").strip()
            for item in state.get("tool_manifest", []) or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        # finish is a runtime action and may not appear in the tool manifest.
        manifest.add("finish")
        return configured.intersection(manifest) if manifest else configured

    @staticmethod
    def _append(
        specs: list[ActionSpec],
        available: set[str],
        name: str,
        description: str,
    ) -> None:
        if name in available:
            specs.append(ActionSpec(name, description))

    @staticmethod
    def _dedupe(specs: list[ActionSpec]) -> list[ActionSpec]:
        result: list[ActionSpec] = []
        seen: set[str] = set()
        for spec in specs:
            if spec.name in seen:
                continue
            seen.add(spec.name)
            result.append(spec)
        return result


def _task_intent(state: AgentState) -> str:
    brief = state.get("task_brief")
    if isinstance(brief, dict):
        intent = str(brief.get("intent") or "").strip().lower()
        if intent in {"diagnose", "implement", "explain", "review"}:
            return intent
    task_type = str(state.get("task_type") or "").strip().upper()
    return "implement" if task_type in {"BUG_FIX", "FEATURE_IMPL"} else "diagnose"


def _has_usable_read_cache(state: AgentState) -> bool:
    cache = state.get("read_file_cache")
    if not isinstance(cache, dict):
        return False
    return any(
        isinstance(snapshot, dict)
        and bool(snapshot.get("spans") or snapshot.get("is_empty"))
        for snapshot in cache.values()
    )


def _runtime_decision_or_evaluate(state: AgentState) -> dict[str, Any]:
    decision = state.get("runtime_decision")
    if isinstance(decision, dict) and decision:
        return decision
    return {"phase": derive_phase(state), "is_complete": False, "blockers": []}
