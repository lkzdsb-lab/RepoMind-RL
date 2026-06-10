"""Action space for the debug agent RL policy."""

from __future__ import annotations

from loguru import logger
from typing import Any, Iterable

from ext.focus_files import (
    current_execution_item as state_current_execution_item,
    edited_files_needing_reread,
    execution_target_files as state_execution_target_files,
)
from ext.file_requirements import (
    choose_read_file_target,
    full_read_requirements,
    is_full_read,
    recommended_read_file_args,
)
from ext.plan_mode_args import (
    build_enter_plan_mode_args,
    build_exit_plan_mode_args,
    collect_plan_focus_files,
)
from agent_runtime.verification import infer_lightweight_verification_command
from agent_runtime.search_query import SearchQueryPlanner
from model.agent.actions import Action, ActionSpec
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
        if state.get("status") in {"finished", "failed"}:
            return [ActionSpec("finish", "Finish terminal task.")]

        specs: list[ActionSpec] = []
        available = self._available_action_names(state)
        called = [call.get("name") for call in state.get("tool_calls", [])]
        candidate_files = state.get("candidate_files") or []
        unread = [path for path in candidate_files if path not in self._read_files(state)]
        full_read_needed = full_read_requirements(state, candidate_files=list(candidate_files))
        verification_required = bool(state.get("verification_required", True))
        plan_mode = bool(state.get("plan_mode", False))
        plan_approved = bool(state.get("plan_mode_approved", False))
        need_more_context = state.get("status") in {"need_more_context", "planning"}
        current_execution = self._current_execution_item(state)
        edit_task = bool(state.get("editing_enabled", False)) and state.get("task_type") in {
            "BUG_FIX",
            "FEATURE_IMPL",
        }
        verification_stale = bool(state.get("verification_stale", False))

        if plan_mode:
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
                and bool(state.get("llm_action_inputs_enabled", False))
                and bool(state.get("debug_technical_plan"))
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
        if verification_stale and not (current_execution and current_execution.get("kind") == "patch"):
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

    def legal_actions(self, state: AgentState) -> list[Action]:
        return [self.to_action(spec, state) for spec in self.legal_specs(state)]

    def consume_last_limit_events(self) -> list[dict[str, Any]]:
        events = list(self._last_limit_events)
        self._last_limit_events = []
        return events

    # 将 action 语义 convert to llm 的思考
    def to_action(self, spec: ActionSpec, state: AgentState) -> Action:
        """ 根据不同的 action spec 返回不同的 action"""
        if spec.name == "search_code_context":
            query_plan = self.query_planner.plan(state)
            return Action(
                "search_code_context",
                {"query": query_plan.query, "query_plan": query_plan.to_dict()},
                thought=f"RL 选择结构化代码上下文搜索：`{query_plan.query}`。",
            )
        if spec.name == "read_file":
            current_execution = self._current_execution_item(state)
            execution_targets = self._execution_target_files(current_execution) if current_execution else []
            reread = edited_files_needing_reread(state)
            if execution_targets:
                reread = [path for path in reread if path in execution_targets]
            required_full_reads = [
                str(item.get("file_path") or "").strip()
                for item in full_read_requirements(
                    state,
                    candidate_files=execution_targets or list(state.get("candidate_files", [])),
                )
                if str(item.get("file_path") or "").strip()
            ]
            unread = [
                path
                for path in (execution_targets or list(state.get("candidate_files", [])))
                if path not in self._read_files(state)
            ]
            targets = reread or required_full_reads or unread or list(state.get("edited_files", []) or [])
            if not targets and execution_targets:
                targets = execution_targets
            if not targets:
                targets = list(state.get("candidate_files", []) or [])
            file_path = choose_read_file_target(
                state,
                default_path=targets[0] if targets else "",
            )
            read_args = recommended_read_file_args(state, file_path)
            return Action(
                "read_file",
                {"file_path": file_path, "max_chars": read_args["max_chars"]},
                thought=f"RL 选择阅读文件 `{file_path}`。",
            )
        if spec.name == "run_tests":
            command = self._default_verification_command(state)
            return Action(
                "run_tests",
                {"command": command},
                thought=f"RL 选择运行验证命令 `{command}`。",
            )
        if spec.name == "search_text":
            query_plan = self.query_planner.plan(state)
            return Action(
                "search_text",
                {
                    "pattern": query_plan.query,
                    "regex": True,
                    "globs": [],
                    "context_lines": 0,
                    "max_results": 50,
                },
                thought=f"LLM 可在默认搜索词 `{query_plan.query}` 基础上补充 regex/fixed-string 搜索参数。",
            )
        if spec.name == "run_shell_command":
            current_execution = self._current_execution_item(state)
            command = self._execution_command(current_execution, state) if current_execution else self._default_verification_command(state)
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
        if spec.name == "list_files":
            return Action("list_files", thought="RL 选择读取仓库结构。")
        if spec.name == "EnterPlanMode":
            args = self._default_enter_plan_mode_args(state)
            logger.debug(
                "EnterPlanMode",
                f"technical_plan: {args['technical_plan']}",
            )
            return Action(
                "EnterPlanMode",
                args,
                thought="LLM 需要进入 Plan Mode 并给出 Debug/重构技术方案。",
            )
        if spec.name == "ExitPlanMode":
            args = self._default_exit_plan_mode_args(state)
            return Action(
                "ExitPlanMode",
                args,
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
        if spec.name == "write_memory":
            return Action("write_memory", thought="RL 选择写入 reward-gated memory。")
        return Action("finish", thought="RL 选择结束当前任务。")

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

    def _default_verification_command(self, state: AgentState) -> str:
        return infer_lightweight_verification_command(
            str(state.get("repo_path") or "."),
            configured=str(state.get("verify_command") or ""),
            changed_files=list(state.get("edited_files", []) or []),
            candidate_files=list(state.get("candidate_files", []) or []),
        )

    def _current_execution_item(self, state: AgentState) -> dict[str, Any] | None:
        return state_current_execution_item(state)

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
            count = max(
                counts.get(signature, 0),
                counts.get(f"action:{spec.name}", 0),
            )
            if limit > 0 and count >= limit:
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
            action = str(item.get("action") or "").strip()
            if signature:
                counts[signature] = counts.get(signature, 0) + 1
            if action:
                key = f"action:{action}"
                counts[key] = counts.get(key, 0) + 1
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
            pending = state.get("pending_action_requirements", {}) or {}
            missing = ",".join(
                sorted(
                    str(item).strip()
                    for item in pending.get("missing_required_args", []) or []
                    if str(item).strip()
                )
            )
            return f"request_user_input:{missing or 'generic'}"
        return f"action:{spec.name}"

    def _default_enter_plan_mode_args(self, state: AgentState) -> dict[str, Any]:
        """ 特殊处理 enter plan mode 的 arg"""
        return build_enter_plan_mode_args(
            state,
            query=self.query_planner.plan(state).query,
            focus_files=collect_plan_focus_files(state),
            read_files=sorted(self._read_files(state)),
            default_verification_command=self._default_verification_command(state),
        )

    def _default_exit_plan_mode_args(self, state: AgentState) -> dict[str, Any]:
        """ 特殊处理 exit plan mode 的 arg"""
        return build_exit_plan_mode_args(
            state,
            focus_files=collect_plan_focus_files(state),
            read_files=sorted(self._read_files(state)),
            default_verification_command=self._default_verification_command(state),
        )

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
