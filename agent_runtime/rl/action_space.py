"""Action space for the debug agent RL policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent_runtime.search_query import SearchQueryPlanner
from model.agent.actions import Action, ActionSpec
from model.agent.graph import AgentState


class ActionSpace:
    def __init__(self, action_names: Iterable[str] | None = None) -> None:
        self.query_planner = SearchQueryPlanner()
        # 若不指定 action 类型，则默认走全部默认流程
        self.action_names = list(
            action_names
            or [
                "list_files",
                "search_code_context",
                "read_file",
                "run_tests",
                "git_diff",
                "write_memory",
                "finish",
            ]
        )

    def legal_specs(self, state: AgentState) -> list[ActionSpec]:
        if state.get("status") in {"finished", "failed"}:
            return [ActionSpec("finish", "Finish terminal task.")]

        specs: list[ActionSpec] = []
        called = [call.get("name") for call in state.get("tool_calls", [])]
        candidate_files = state.get("candidate_files") or []
        unread = [path for path in candidate_files if path not in self._read_files(state)]

        # 根据动作名称列表补充 action spec
        if "list_files" in self.action_names and "list_files" not in called:
            specs.append(ActionSpec("list_files", "List repository files."))

        if "search_code_context" in self.action_names and not state.get("code_context"):
            specs.append(ActionSpec("search_code_context", "Search structured code context."))

        if "read_file" in self.action_names and unread:
            specs.append(ActionSpec("read_file", "Read the next unread candidate file."))

        if "run_tests" in self.action_names and not state.get("test_results"):
            specs.append(ActionSpec("run_tests", "Run verification command."))

        if (
            "git_diff" in self.action_names
            and state.get("test_results")
            and state.get("patch_summary") is None
        ):
            specs.append(ActionSpec("git_diff", "Inspect current git diff."))

        if (
            "write_memory" in self.action_names
            and state.get("patch_summary") is not None
            and not state.get("memory_written")
        ):
            specs.append(ActionSpec("write_memory", "Write reward-gated memory."))

        if self._can_finish(state) or not specs:
            specs.append(ActionSpec("finish", "Finish the current run."))

        return specs

    def legal_actions(self, state: AgentState) -> list[Action]:
        return [self.to_action(spec, state) for spec in self.legal_specs(state)]

    # 将 action 语义 convert to llm 的思考
    def to_action(self, spec: ActionSpec, state: AgentState) -> Action:
        if spec.name == "search_code_context":
            query_plan = self.query_planner.plan(state)
            return Action(
                "search_code_context",
                {"query": query_plan.query, "query_plan": query_plan.to_dict()},
                thought=f"RL 选择结构化代码上下文搜索：`{query_plan.query}`。",
            )
        if spec.name == "read_file":
            unread = [
                path
                for path in state.get("candidate_files", [])
                if path not in self._read_files(state)
            ]
            return Action(
                "read_file",
                {"file_path": unread[0]},
                thought=f"RL 选择阅读候选文件 `{unread[0]}`。",
            )
        if spec.name == "run_tests":
            command = state.get("verify_command") or "pytest"
            return Action(
                "run_tests",
                {"command": command},
                thought=f"RL 选择运行验证命令 `{command}`。",
            )
        if spec.name == "list_files":
            return Action("list_files", thought="RL 选择读取仓库结构。")
        if spec.name == "git_diff":
            return Action("git_diff", thought="RL 选择检查当前 diff。")
        if spec.name == "write_memory":
            return Action("write_memory", thought="RL 选择写入 reward-gated memory。")
        return Action("finish", thought="RL 选择结束当前任务。")

    def extract_keyword(self, state: AgentState) -> str:
        return self.query_planner.plan(state).query

    # 从 observation 中获取文件路径
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
            or int(state.get("loop_count", 0)) >= int(state.get("max_loops", 8)) - 1
        )
