"""Action space for the debug agent RL policy."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Iterable

from agent_runtime.search_query import SearchQueryPlanner
from model.agent.actions import Action, ActionSpec
from model.agent.graph import AgentState

# ---------------------------------------------------------------------------
# Action categories
# ---------------------------------------------------------------------------
# Pure RL actions are always available (no LLM needed for parameter generation).
PURE_RL_ACTIONS = {
    "search_code_context",
    "read_file",
    "run_tests",
    "git_diff",
    "finish",
}

# LLM-assisted actions require llm_action_inputs_enabled=True because the
# LLM must generate complex / structured arguments.
LLM_ASSISTED_ACTIONS = {
    "search_text",
    "run_shell_command",
    "EnterPlanMode",
    "ExitPlanMode",
    "apply_code_patch",
    "request_user_input",
}

# System actions are *never* selected by the policy.  The executor uses them
# internally (e.g. write_memory on finalize).
SYSTEM_ACTIONS = {
    "write_memory",
}

ACTION_SPACE_VERSION = "action-space-v1"

# Default action names exposed to the policy.
# list_files is intentionally excluded because its default ToolRegistry
# registration is commented-out.
# write_memory is reserved for executor finalize, not for policy selection.
DEFAULT_ACTION_NAMES = sorted(PURE_RL_ACTIONS | LLM_ASSISTED_ACTIONS)


class ActionSpace:
    def __init__(self, action_names: Iterable[str] | None = None) -> None:
        self.query_planner = SearchQueryPlanner()
        self.action_names = list(action_names or DEFAULT_ACTION_NAMES)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def legal_specs(self, state: AgentState) -> list[ActionSpec]:
        """Return ActionSpec objects that are legal in *state*."""
        if state.get("status") in {"finished", "failed"}:
            return [ActionSpec("finish", "Finish terminal task.")]

        specs: list[ActionSpec] = []
        available = self._available_action_names(state)
        called = [call.get("name") for call in state.get("tool_calls", [])]
        candidate_files = state.get("candidate_files") or []
        unread = [path for path in candidate_files if path not in self._read_files(state)]
        verification_required = bool(state.get("verification_required", True))
        plan_mode = bool(state.get("plan_mode", False))
        plan_approved = bool(state.get("plan_mode_approved", False))
        llm_enabled = bool(state.get("llm_action_inputs_enabled", False))
        edit_task = bool(state.get("editing_enabled", False)) and state.get("task_type") in {
            "BUG_FIX",
            "FEATURE_IMPL",
        }
        verification_stale = bool(state.get("verification_stale", False))
        has_read_files = bool(self._read_files(state))

        # -- plan-mode gate -------------------------------------------------
        if plan_mode:
            if (
                "EnterPlanMode" in available
                and llm_enabled
                and not bool(state.get("debug_technical_plan"))
            ):
                specs.append(
                    ActionSpec(
                        "EnterPlanMode",
                        "Create the missing Debug/Refactor technical plan without changing code.",
                    )
                )
            if (
                "ExitPlanMode" in available
                and llm_enabled
                and bool(state.get("debug_technical_plan"))
            ):
                specs.append(
                    ActionSpec(
                        "ExitPlanMode",
                        "Exit planning mode after evaluating the plan as feasible.",
                    )
                )
            if "request_user_input" in available and llm_enabled:
                specs.append(
                    ActionSpec(
                        "request_user_input",
                        "Pause and ask the user concrete questions when required.",
                    )
                )
            return self._finish_safe_specs(specs, allow_finish=False)

        # -- verification-stale gate ----------------------------------------
        if verification_stale:
            stale_specs: list[ActionSpec] = []
            if "read_file" in available and self._edited_files_needing_reread(state):
                stale_specs.append(
                    ActionSpec(
                        "read_file",
                        "Re-read an edited file after the latest patch before verification.",
                    )
                )
            if (
                "run_shell_command" in available
                and llm_enabled
            ):
                stale_specs.append(
                    ActionSpec(
                        "run_shell_command",
                        "Run a verification command for the latest code changes.",
                    )
                )
            elif "run_tests" in available:
                stale_specs.append(ActionSpec("run_tests", "Run verification command."))
            if stale_specs:
                return stale_specs

        # -- search_code_context (pure RL) ----------------------------------
        if "search_code_context" in available and not state.get("code_context"):
            specs.append(ActionSpec("search_code_context", "Search structured code context."))

        # -- search_text (LLM-assisted) ------------------------------------
        if (
            "search_text" in available
            and llm_enabled
            and (not candidate_files or state.get("status") in {"need_more_context", "planning"})
        ):
            specs.append(
                ActionSpec("search_text", "Search repository text with regex or fixed strings.")
            )

        # -- read_file (pure RL) -------------------------------------------
        if "read_file" in available and unread:
            specs.append(ActionSpec("read_file", "Read the next unread candidate file."))

        # -- EnterPlanMode (LLM-assisted) ----------------------------------
        if (
            "EnterPlanMode" in available
            and llm_enabled
            and edit_task
            and not plan_approved
        ):
            specs.append(
                ActionSpec(
                    "EnterPlanMode",
                    "Enter non-mutating planning mode and write a detailed technical plan.",
                )
            )

        # -- request_user_input (LLM-assisted) ------------------------------
        if "request_user_input" in available and llm_enabled:
            specs.append(
                ActionSpec(
                    "request_user_input",
                    "Pause and ask the user concrete questions when required.",
                )
            )

        # -- apply_code_patch (LLM-assisted, gated) -------------------------
        if (
            "apply_code_patch" in available
            and bool(state.get("editing_enabled", False))
            and llm_enabled
            and plan_approved
            and has_read_files
            and not self._has_applied_edit(state)
        ):
            specs.append(
                ActionSpec(
                    "apply_code_patch",
                    "Apply guarded exact-replacement edits to already-read files.",
                )
            )

        # -- run_shell_command (LLM-assisted only) --------------------------
        if (
            "run_shell_command" in available
            and llm_enabled
            and (
                bool(state.get("verification_stale", False))
                or (
                    verification_required
                    and self._read_files(state)
                    and not state.get("test_results")
                )
            )
        ):
            specs.append(
                ActionSpec(
                    "run_shell_command",
                    "Run a guarded command for verification or diagnostics.",
                )
            )

        # -- run_tests (pure RL) --------------------------------------------
        if "run_tests" in available and verification_required and not state.get("test_results"):
            specs.append(ActionSpec("run_tests", "Run verification command."))

        # -- git_diff (pure RL) ---------------------------------------------
        if (
            "git_diff" in available
            and bool(state.get("is_git_repo", True))
            and (not verification_required or state.get("test_results"))
            and not state.get("verification_stale", False)
            and state.get("patch_summary") is None
        ):
            specs.append(ActionSpec("git_diff", "Inspect current git diff."))

        return self._finish_safe_specs(specs, allow_finish=True, state=state)

    def legal_actions(self, state: AgentState) -> list[Action]:
        return [self.to_action(spec, state) for spec in self.legal_specs(state)]

    def to_action(self, spec: ActionSpec, state: AgentState) -> Action:
        """Convert an ActionSpec into a concrete Action with RL-generated args."""
        if spec.name == "search_code_context":
            query_plan = self.query_planner.plan(state)
            return Action(
                "search_code_context",
                {"query": query_plan.query, "query_plan": query_plan.to_dict()},
                thought=f"RL 选择结构化代码上下文搜索：`{query_plan.query}`。",
            )
        if spec.name == "read_file":
            reread = self._edited_files_needing_reread(state)
            unread = [
                path
                for path in state.get("candidate_files", [])
                if path not in self._read_files(state)
            ]
            targets = reread or unread or list(state.get("edited_files", []) or [])
            if not targets:
                targets = list(state.get("candidate_files", []) or [])
            file_path = targets[0] if targets else ""
            return Action(
                "read_file",
                {"file_path": file_path},
                thought=f"RL 选择阅读文件 `{file_path}`。",
            )
        if spec.name == "run_tests":
            command = state.get("verify_command") or "pytest"
            return Action(
                "run_tests",
                {"command": command},
                thought=f"RL 选择运行验证命令 `{command}`。",
            )
        if spec.name == "search_text":
            return Action(
                "search_text",
                thought="LLM 需要提供 regex/fixed-string 搜索参数。",
            )
        if spec.name == "run_shell_command":
            command = self._default_verification_command(state)
            return Action(
                "run_shell_command",
                {
                    "command": command,
                    "purpose": "verification",
                    "timeout": 120,
                    "reason": "Verify the latest code changes before finishing.",
                    "allow_shell": False,
                },
                thought=f"RL 选择运行验证命令 `{command}`。",
            )
        if spec.name == "EnterPlanMode":
            return Action(
                "EnterPlanMode",
                thought="LLM 需要进入 Plan Mode 并给出 Debug/重构技术方案。",
            )
        if spec.name == "ExitPlanMode":
            return Action(
                "ExitPlanMode",
                thought="LLM 需要评估方案可行性后退出 Plan Mode。",
            )
        if spec.name == "apply_code_patch":
            return Action(
                "apply_code_patch",
                thought="LLM 需要提供受限的 exact-replacement 修改内容。",
            )
        if spec.name == "request_user_input":
            return Action(
                "request_user_input",
                thought="当前存在不确定信息，向用户询问具体问题。",
            )
        if spec.name == "git_diff":
            return Action("git_diff", thought="RL 选择检查当前 diff。")
        # write_memory is a system action — never reached via policy
        if spec.name == "write_memory":
            return Action("write_memory", thought="RL 选择写入 reward-gated memory。")
        return Action("finish", thought="RL 选择结束当前任务。")

    def extract_keyword(self, state: AgentState) -> str:
        return self.query_planner.plan(state).query

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_files(self, state: AgentState) -> set[str]:
        files: set[str] = set()
        for observation in state.get("observations", []):
            content = observation.get("content", {})
            if isinstance(content, dict) and content.get("file_path"):
                files.add(str(content["file_path"]))
        return files

    def _can_finish(self, state: AgentState) -> bool:
        return (
            bool(state.get("memory_written"))
            or bool(state.get("error"))
            or self._has_verified_edit(state)
            or (
                state.get("patch_summary") is not None
                and not state.get("verification_stale", False)
            )
            or int(state.get("loop_count", 0)) >= int(state.get("max_loops", 8)) - 1
        )

    def _has_applied_edit(self, state: AgentState) -> bool:
        return any(
            bool(result.get("applied"))
            for result in state.get("edit_results", [])
            if isinstance(result, dict)
        )

    def _has_verified_edit(self, state: AgentState) -> bool:
        return bool(state.get("edited_files")) and not bool(
            state.get("verification_stale", False)
        )

    def _edited_files_needing_reread(self, state: AgentState) -> list[str]:
        edited_files = [
            str(path).strip()
            for path in state.get("edited_files", []) or []
            if str(path).strip()
        ]
        if not edited_files or not bool(state.get("verification_stale", False)):
            return []
        calls = state.get("tool_calls", []) or []
        latest_edit_index = -1
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or call.get("name") != "apply_code_patch":
                continue
            output = call.get("output")
            if isinstance(output, dict) and output.get("applied"):
                latest_edit_index = index
        if latest_edit_index < 0:
            return []
        read_after_edit: set[str] = set()
        for call in calls[latest_edit_index + 1 :]:
            if not isinstance(call, dict) or call.get("name") != "read_file":
                continue
            output = call.get("output")
            call_input = call.get("input")
            if not isinstance(output, dict) or output.get("error"):
                continue
            path = str(output.get("file_path") or "").strip()
            if not path and isinstance(call_input, dict):
                path = str(call_input.get("file_path") or "").strip()
            if path:
                read_after_edit.add(path)
        return [path for path in edited_files if path not in read_after_edit]

    def _default_verification_command(self, state: AgentState) -> str:
        configured = str(state.get("verify_command") or "").strip()
        if configured:
            return configured
        paths = [
            str(path).strip()
            for path in (state.get("edited_files", []) or state.get("candidate_files", []) or [])
            if str(path).strip()
        ]
        for path in paths:
            quoted = shlex.quote(path)
            if path.endswith(".py"):
                if path.rsplit("/", 1)[-1] == "main.py":
                    return f"python {quoted}"
                return f"python -m py_compile {quoted}"
            if path.endswith(".go"):
                return "go test ./..."
        return "pytest"

    def _available_action_names(self, state: AgentState) -> set[str]:
        """Filter action_names to those registered in the tool snapshot.

        Internal actions (`finish`, `write_memory`, `request_user_input`) are
        always available regardless of registry contents.
        """
        registry = state.get("registry_snapshot") or {}
        registered_tools = set(registry.get("tools") or [])
        if not registered_tools:
            return set(self.action_names)
        # write_memory is a system action — never exposed to the policy.
        internal_actions = {"finish", "request_user_input"}
        return set(self.action_names).intersection(registered_tools | internal_actions)

    def _finish_safe_specs(
        self,
        specs: list[ActionSpec],
        *,
        allow_finish: bool,
        state: AgentState | None = None,
    ) -> list[ActionSpec]:
        non_question_specs = [
            spec
            for spec in specs
            if spec.name not in {"request_user_input"}
        ]
        if allow_finish and state is not None and (
            self._can_finish(state) or not non_question_specs
        ):
            specs.append(ActionSpec("finish", "Finish the current run."))
        return specs
