"""Action space for the debug agent RL policy."""

from __future__ import annotations

from typing import Any, Iterable

from agent_runtime.lifecycle.completion import derive_phase, evaluate_completion_transition
from agent_runtime.lifecycle.execution_queue import current_execution_item
from ext.focus_files import (
    edited_files_needing_reread,
    execution_target_files as state_execution_target_files,
)
from ext.file_requirements import (
    choose_read_file_target,
    full_read_requirements,
    is_full_read,
)
from agent_runtime.search_query import SearchQueryPlanner
from agent_runtime.verification.capabilities import recommended_verification_command
from model.agent.actions import ActionSpec
from model.agent.graph import AgentState


class ActionSpace:
    ACTION_WINDOW_SIZE = 6
    ACTION_LIMITS = {
        "apply_code_patch": 2,
        "request_user_input": 1,
        "search_code_context": 2,
        "run_shell_command": 2,
    }
    READ_FILE_LIMIT = 2

    def __init__(self, action_names: Iterable[str] | None = None) -> None:
        self.query_planner = SearchQueryPlanner()
        self._last_limit_events: list[dict[str, Any]] = []
        # 若不指定 action 类型，则默认走全部默认流程
        self.action_names = list(
            action_names
            or [
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
                "write_memory",
                "finish",
            ]
        )

    def legal_specs(self, state: AgentState) -> list[ActionSpec]:
        self._last_limit_events = []
        runtime_decision = _runtime_decision_or_evaluate(state)
        phase = str(runtime_decision.get("phase") or state.get("phase") or derive_phase(state))
        if state.get("status") in {"finished", "failed"} or phase == "complete":
            return [ActionSpec("finish", "Finish terminal task.")]

        specs: list[ActionSpec] = []
        available = self._available_action_names(state)
        called = [call.get("name") for call in state.get("tool_calls", [])]
        candidate_files = state.get("candidate_files") or []
        unread = [path for path in candidate_files if path not in self._read_files(state)]
        full_read_needed = full_read_requirements(state, candidate_files=list(candidate_files))
        verification_required = bool(state.get("verification_required", True))
        plan_mode = phase == "plan"
        plan_approved = bool(state.get("plan_mode_approved", False))
        need_more_context = state.get("status") in {"need_more_context", "planning"}
        current_execution = self._current_execution_item(state)
        pending_resolution = state.get("pending_resolution") or {}
        edit_task = bool(state.get("editing_enabled", False)) and state.get("task_type") in {
            "BUG_FIX",
            "FEATURE_IMPL",
        }
        verification_stale = bool(state.get("verification_stale", False))
        obligation = state.get("next_obligation") if isinstance(state.get("next_obligation"), dict) else {}
        obligation_kind = str(obligation.get("kind") or "").strip()

        if obligation_kind and obligation_kind != "complete":
            obligation_specs = self._obligation_specs(
                state,
                obligation=obligation,
                current_execution=current_execution,
                available=available,
                full_read_needed=full_read_needed,
                unread=unread,
                plan_approved=plan_approved,
            )
            if obligation_specs:
                return self._limit_specs(
                    state,
                    self._finish_safe_specs(obligation_specs, allow_finish=False),
                )

        # 如果当前待解决的类型为 recovery ，则先处理 recovery 的内容
        if str(pending_resolution.get("kind") or "") == "recovery" and str(
            pending_resolution.get("target_file") or ""
        ).strip():
            recovery_file = str(pending_resolution.get("target_file") or "").strip()
            recovery_specs: list[ActionSpec] = []
            if "read_file" in available:
                recovery_specs.append(
                    ActionSpec(
                        "read_file",
                        f"Re-read {recovery_file} to refresh patch anchors after a recoverable edit conflict.",
                    )
                )
            if "search_code_context" in available:
                recovery_specs.append(
                    ActionSpec(
                        "search_code_context",
                        "Refresh structured context for the recoverable patch conflict.",
                    )
                )
            if recovery_specs:
                return self._limit_specs(state, self._finish_safe_specs(recovery_specs, allow_finish=False))

        if plan_mode:
            if full_read_needed:
                read_specs: list[ActionSpec] = []
                required_files = [
                    str(item.get("file_path") or "").strip()
                    for item in full_read_needed
                    if str(item.get("file_path") or "").strip()
                ]
                target = required_files[0] if required_files else "the required file"
                if "read_file" in available:
                    read_specs.append(
                        ActionSpec(
                            "read_file",
                            f"Fully read {target} before continuing the technical plan.",
                        )
                    )
                if "request_user_input" in available:
                    read_specs.append(
                        ActionSpec(
                            "request_user_input",
                            "Pause only if the required file cannot be read or the target is ambiguous.",
                        )
                    )
                if read_specs:
                    return self._limit_specs(
                        state,
                        self._finish_safe_specs(read_specs, allow_finish=False),
                    )
            # EnterPlanMode is the gate into planning. Once a plan exists, keep the
            # model moving toward ExitPlanMode or a concrete user question instead
            # of letting it spend loops rewriting the same plan.
            if (
                "search_code_context" in available
                and (
                    not state.get("code_context")
                    or need_more_context
                    or not self._read_files(state)
                )
            ):
                specs.append(
                    ActionSpec(
                        "search_code_context",
                        "Search structured code context to refine the technical plan.",
                    )
                )
            if (
                "search_text" in available
                and bool(state.get("llm_action_inputs_enabled", False))
                and (not candidate_files or need_more_context)
            ):
                specs.append(
                    ActionSpec(
                        "search_text",
                        "Search repository text to fill missing planning details.",
                    )
                )
            if "read_file" in available and (unread or edited_files_needing_reread(state) or full_read_needed):
                specs.append(
                    ActionSpec(
                        "read_file",
                        "Read a relevant candidate file to complete the technical plan.",
                    )
                )
            if (
                "EnterPlanMode" in available
                and bool(state.get("llm_action_inputs_enabled", False))
                and not bool(state.get("technical_plan"))
            ):
                specs.append(
                    ActionSpec(
                        "EnterPlanMode",
                        "Create the missing Debug/Refactor technical plan without changing code.",
                    )
                )
            if (
                "ExitPlanMode" in available
                and bool(state.get("llm_action_inputs_enabled", False))
                and bool(state.get("technical_plan"))
            ):
                specs.append(
                    ActionSpec(
                        "ExitPlanMode",
                        "Exit planning mode after evaluating the plan as feasible.",
                    )
                )
            if "request_user_input" in available:
                specs.append(
                    ActionSpec(
                        "request_user_input",
                        "Pause and ask the user concrete questions when required.",
                    )
                )
            return self._limit_specs(state, self._finish_safe_specs(specs, allow_finish=False))

        # 如果校验过期并且当权执行队列没有 edit 操作时才会
        if verification_stale and not (current_execution and current_execution.get("kind") in {"patch", "verify"}):
            stale_specs: list[ActionSpec] = []
            if "read_file" in available and edited_files_needing_reread(state):
                stale_specs.append(
                    ActionSpec(
                        "read_file",
                        "Re-read an edited file after the latest patch before verification.",
                    )
                )
            if (
                "run_shell_command" in available
                and bool(state.get("llm_action_inputs_enabled", False))
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
                return self._limit_specs(state, stale_specs)

        if plan_approved and current_execution:
            current_kind = str(current_execution.get("kind") or "")
            target_files = self._execution_target_files(current_execution)
            relevant_unread = [path for path in target_files if path not in self._read_files(state)]
            reread = [
                path
                for path in edited_files_needing_reread(state)
                if not target_files or path in target_files
            ]
            scoped_specs: list[ActionSpec] = []
            if current_kind == "patch":
                if (
                    "search_code_context" in available
                    and (
                        not state.get("code_context")
                        or need_more_context
                    )
                ):
                    scoped_specs.append(ActionSpec("search_code_context", "Refresh structured context for the current patch target."))
                target_full_read_needed = [
                    item
                    for item in full_read_requirements(state, candidate_files=target_files)
                    if str(item.get("file_path") or "") in target_files
                ]
                global_full_read_needed = full_read_requirements(
                    state,
                    candidate_files=list(state.get("candidate_files", []) or []),
                )
                if "read_file" in available and (
                    reread
                    or relevant_unread
                    or target_full_read_needed
                    or global_full_read_needed
                ):
                    scoped_specs.append(
                        ActionSpec(
                            "read_file",
                            "Read the current patch target file or any globally required full-read file before patching.",
                        )
                    )
                if (
                    "apply_code_patch" in available
                    and bool(state.get("editing_enabled", False))
                    and bool(state.get("llm_action_inputs_enabled", False))
                    and self._targets_ready_for_patch(state, target_files)
                ):
                    scoped_specs.append(ActionSpec("apply_code_patch", "Apply the next patch task from the approved execution queue."))
                if "request_user_input" in available:
                    scoped_specs.append(ActionSpec("request_user_input", "Pause and ask the user concrete questions when required."))
                return self._limit_specs(state, self._finish_safe_specs(scoped_specs, allow_finish=False))
            if current_kind == "verify":
                verify_specs: list[ActionSpec] = []
                if "read_file" in available and reread:
                    verify_specs.append(ActionSpec("read_file", "Re-read the latest edited file before verification."))
                if (
                    "run_shell_command" in available
                    and bool(state.get("llm_action_inputs_enabled", False))
                ):
                    verify_specs.append(ActionSpec("run_shell_command", "Run the next verification command from the approved execution queue."))
                elif "run_tests" in available:
                    verify_specs.append(ActionSpec("run_tests", "Run verification command."))
                if "request_user_input" in available:
                    verify_specs.append(ActionSpec("request_user_input", "Pause and ask the user concrete questions when required."))
                base_specs = verify_specs or self._finish_safe_specs([], allow_finish=False, state=state)
                return self._limit_specs(state, base_specs)

        # 根据动作名称列表补充 action spec
        if "list_files" in available and "list_files" not in called:
            specs.append(ActionSpec("list_files", "List repository files."))

        if (
            "search_code_context" in available
            and (
                not state.get("code_context")
                or need_more_context
            )
        ):
            specs.append(ActionSpec("search_code_context", "Search structured code context."))

        if (
            "search_text" in available
            and bool(state.get("llm_action_inputs_enabled", False))
            and (not candidate_files or need_more_context)
        ):
            specs.append(ActionSpec("search_text", "Search repository text with regex or fixed strings."))

        if "read_file" in available and (unread or full_read_needed):
            specs.append(ActionSpec("read_file", "Read the next unread candidate file."))

        if (
            "EnterPlanMode" in available
            and bool(state.get("llm_action_inputs_enabled", False))
            and edit_task
            and not plan_approved
        ):
            specs.append(
                ActionSpec(
                    "EnterPlanMode",
                    "Enter non-mutating planning mode and write a detailed technical plan.",
                )
            )

        if "request_user_input" in available:
            specs.append(
                ActionSpec(
                    "request_user_input",
                    "Pause and ask the user concrete questions when required.",
                )
            )

        if (
            "apply_code_patch" in available
            and bool(state.get("editing_enabled", False))
            and bool(state.get("llm_action_inputs_enabled", False))
            and plan_approved
            and self._read_files(state)
            and not full_read_needed
            and not self._has_applied_edit(state)
        ):
            specs.append(
                ActionSpec(
                    "apply_code_patch",
                    "Apply guarded exact-replacement edits to already-read files.",
                )
            )

        if (
            "run_shell_command" in available
            and bool(state.get("llm_action_inputs_enabled", False))
            and (
                bool(state.get("verification_stale", False))
                or (verification_required and self._read_files(state) and not state.get("test_results"))
            )
        ):
            specs.append(ActionSpec("run_shell_command", "Run a guarded command for verification or diagnostics."))

        if "run_tests" in available and verification_required and not state.get("test_results"):
            specs.append(ActionSpec("run_tests", "Run verification command."))

        if (
            "git_diff" in available
            and bool(state.get("is_git_repo", True))
            and (not verification_required or state.get("test_results"))
            and not state.get("verification_stale", False)
            and state.get("patch_summary") is None
        ):
            specs.append(ActionSpec("git_diff", "Inspect current git diff."))

        return self._limit_specs(state, self._finish_safe_specs(specs, allow_finish=True, state=state))

    def consume_last_limit_events(self) -> list[dict[str, Any]]:
        events = list(self._last_limit_events)
        self._last_limit_events = []
        return events

    def extract_keyword(self, state: AgentState) -> str:
        return self.query_planner.plan(state).query

    # 从 observation 中获取文件路径
    def _read_files(self, state: AgentState) -> set[str]:
        cache = state.get("read_file_cache")
        if isinstance(cache, dict) and cache:
            return {
                str(path).strip()
                for path in cache.keys()
                if str(path).strip()
            }
        files: set[str] = set()
        for observation in state.get("observations", []):
            content = observation.get("content", {})
            if isinstance(content, dict) and content.get("file_path"):
                files.add(str(content["file_path"]))
        return files

    def _can_finish(self, state: AgentState) -> bool:
        decision = _runtime_decision_or_evaluate(state)
        return bool(decision.get("is_complete"))

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

    def _obligation_specs(
        self,
        state: AgentState,
        *,
        obligation: dict[str, Any],
        current_execution: dict[str, Any] | None,
        available: set[str],
        full_read_needed: list[dict[str, Any]],
        unread: list[str],
        plan_approved: bool,
    ) -> list[ActionSpec]:
        specs: list[ActionSpec] = []
        capability = str(obligation.get("required_capability") or "")
        target_files = self._execution_target_files(current_execution)
        target_unread = [path for path in target_files if path not in self._read_files(state)]
        target_full_read_needed = full_read_requirements(
            state,
            candidate_files=target_files,
        ) if target_files else []
        queue_target_ready = (
            bool(target_files)
            and not target_unread
            and not target_full_read_needed
        ) or (
            not target_files
            and bool(self._read_files(state))
            and not full_read_needed
        )
        if capability == "read_code":
            if "read_file" in available and (unread or full_read_needed):
                specs.append(ActionSpec("read_file", "Read focused files required to complete diagnosis."))
            if "run_shell_command" in available and bool(state.get("llm_action_inputs_enabled", False)):
                specs.append(ActionSpec("run_shell_command", "Run an allowed verification command to collect failure evidence."))
            if "search_code_context" in available and not state.get("code_context"):
                specs.append(ActionSpec("search_code_context", "Search structured code context for diagnostic targets."))
            return specs
        if capability == "patch":
            if "read_file" in available and (
                target_unread
                or target_full_read_needed
                or (not target_files and (unread or full_read_needed))
                or edited_files_needing_reread(state)
            ):
                description = (
                    f"Read the current queue target before implementation: {target_files[0]}."
                    if target_files
                    else "Read exact target source before implementation."
                )
                specs.append(ActionSpec("read_file", description))
            if (
                "EnterPlanMode" in available
                and bool(state.get("llm_action_inputs_enabled", False))
                and not plan_approved
            ):
                specs.append(ActionSpec("EnterPlanMode", "Create or refine the implementation plan for the unresolved obligation."))
            if (
                "apply_code_patch" in available
                and bool(state.get("editing_enabled", False))
                and bool(state.get("llm_action_inputs_enabled", False))
                and plan_approved
                and queue_target_ready
            ):
                specs.append(ActionSpec("apply_code_patch", "Apply the code change required by the goal contract."))
            if "request_user_input" in available:
                specs.append(ActionSpec("request_user_input", "Ask only if the required implementation behavior is ambiguous."))
            return specs
        if capability == "verification":
            if "read_file" in available and edited_files_needing_reread(state):
                specs.append(ActionSpec("read_file", "Re-read edited files before verification."))
            if "run_shell_command" in available and bool(state.get("llm_action_inputs_enabled", False)):
                specs.append(ActionSpec("run_shell_command", "Run an allowed verification command selected from project capabilities."))
            elif "run_tests" in available:
                specs.append(ActionSpec("run_tests", "Run an allowed verification command."))
            return specs
        return specs

    def _default_verification_command(self, state: AgentState) -> str:
        return recommended_verification_command(state)

    def _current_execution_item(self, state: AgentState) -> dict[str, Any] | None:
        return current_execution_item(state)

    def _execution_target_files(self, item: dict[str, Any] | None) -> list[str]:
        if not isinstance(item, dict):
            return []
        return state_execution_target_files({"execution_queue": [item]})

    def _execution_command(self, item: dict[str, Any] | None, state: AgentState) -> str:
        if isinstance(item, dict):
            for command in item.get("commands", []) or []:
                text = str(command or "").strip()
                if text:
                    return text
        return self._default_verification_command(state)

    def _targets_ready_for_patch(self, state: AgentState, target_files: list[str]) -> bool:
        if not target_files:
            return bool(self._read_files(state))
        read_files = self._read_files(state)
        for path in target_files:
            if path not in read_files:
                return False
            if not is_full_read(state, path):
                return False
        return True

    def _limit_specs(self, state: AgentState, specs: list[ActionSpec]) -> list[ActionSpec]:
        if len(specs) <= 1:
            self._last_limit_events = []
            return specs
        counts = self._recent_action_counts(state)
        allowed: list[ActionSpec] = []
        blocked: list[dict[str, Any]] = []
        for spec in specs:
            signature = self._spec_signature(spec, state)
            limit = self._limit_for_signature(spec.name, signature)
            count = counts.get(signature, 0)
            if 0 < limit <= count:
                blocked.append(
                    {
                        "action": spec.name,
                        "signature": signature,
                        "count": count,
                        "limit": limit,
                        "window_size": self.ACTION_WINDOW_SIZE,
                    }
                )
                continue
            allowed.append(spec)
        if not allowed:
            self._last_limit_events = []
            return specs
        self._last_limit_events = blocked
        return allowed

    def _recent_action_counts(self, state: AgentState) -> dict[str, int]:
        history = state.get("action_history", []) or []
        counts: dict[str, int] = {}
        for item in history[-self.ACTION_WINDOW_SIZE:]:
            if not isinstance(item, dict):
                continue
            signature = str(item.get("signature") or "").strip()
            if signature:
                counts[signature] = counts.get(signature, 0) + 1
        return counts

    def _limit_for_signature(self, action_name: str, signature: str) -> int:
        if action_name == "read_file":
            return self.READ_FILE_LIMIT
        return int(self.ACTION_LIMITS.get(action_name, 0) or 0)

    def _spec_signature(self, spec: ActionSpec, state: AgentState) -> str:
        if spec.name == "read_file":
            current_execution = self._current_execution_item(state)
            execution_targets = self._execution_target_files(current_execution) if current_execution else []
            targets = execution_targets or list(state.get("candidate_files", []) or [])
            file_path = choose_read_file_target(
                state,
                default_path=targets[0] if targets else "",
            )
            return f"read_file:{file_path or '<unknown>'}"
        if spec.name == "apply_code_patch":
            current_execution = self._current_execution_item(state)
            targets = self._execution_target_files(current_execution)
            if not targets:
                targets = list(state.get("edited_files", []) or []) or list(state.get("candidate_files", []) or [])[:2]
            return f"apply_code_patch:{'|'.join(sorted(targets)) or '<unknown>'}"
        if spec.name == "run_shell_command":
            current_execution = self._current_execution_item(state)
            command = self._execution_command(current_execution, state) if current_execution else self._default_verification_command(state)
            return f"run_shell_command:{command}"
        if spec.name == "search_code_context":
            query = self.query_planner.plan(state).query
            return f"search_code_context:{query or '<empty>'}"
        if spec.name == "request_user_input":
            pending = (state.get("pending_resolution") or {}).get("details") or {}
            missing = ",".join(
                sorted(
                    str(item).strip()
                    for item in pending.get("missing_required_args", []) or []
                    if str(item).strip()
                )
            )
            return f"request_user_input:{missing or 'generic'}"
        return f"action:{spec.name}"

    def _available_action_names(self, state: AgentState) -> set[str]:
        """
            从快照加载需要的工具
        """
        registry = state.get("registry_snapshot") or {}
        registered_tools = set(registry.get("tools") or [])
        if not registered_tools:
            return set(self.action_names)
        internal_actions = {"finish", "write_memory", "request_user_input"}
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
        if allow_finish and state is not None and (self._can_finish(state) or not non_question_specs):
            specs.append(ActionSpec("finish", "Finish the current run."))
        return specs


def _runtime_decision_or_evaluate(state: AgentState) -> dict[str, Any]:
    """ 提取 state 中的 runtime_decision"""
    decision = state.get("runtime_decision")
    if isinstance(decision, dict) and decision:
        return decision
    return evaluate_completion_transition(state)
