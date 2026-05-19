"""
First-version debug policy.

This is intentionally deterministic. It gives the project a runnable baseline
and a clean replacement point for a future LLM planner, contextual bandit, or
DQN controller.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.search_query import SearchQueryPlanner
from model.agent.actions import Action
from model.agent.graph import AgentState


@dataclass
class HeuristicDebugPolicy:
    """
        启发式学习策略
        为 llm 失效后的降级策略
        fallback 后续考虑删除
    """
    default_query: str = "TODO"

    def __post_init__(self) -> None:
        self.query_planner = SearchQueryPlanner(default_query=self.default_query)

    def make_initial_plan(self, state: AgentState) -> list[str]:
        verify_command = state.get("verify_command") or "pytest"
        return [
            "解析 issue，提取代码搜索关键词",
            "读取仓库结构并搜索相关代码",
            "阅读候选文件建立上下文",
            f"运行验证命令：{verify_command}",
            "查看 git diff 并汇总当前补丁状态",
            "按 reward gate 写入分层记忆，必要时沉淀到 skill",
        ]

    # llm 决策
    def next_action(self, state: AgentState) -> Action:
        loop_count = state.get("loop_count", 0)
        max_loops = state.get("max_loops", 8)

        if state.get("status") in {"finished", "failed"}:
            return Action("finish", thought="任务已经到达终态。")

        if loop_count == 0:
            return Action(
                "list_files",
                thought="先读取仓库结构，建立初始上下文。",
            )

        if loop_count == 1:
            query_plan = self.query_planner.plan(state)
            return Action(
                "search_code_context",
                {"query": query_plan.query, "query_plan": query_plan.to_dict()},
                thought=f"用查询 `{query_plan.query}` 搜索结构化代码上下文。",
            )

        candidate_files = state.get("candidate_files") or []
        read_files = self._read_files_from_state(state)
        unread = [path for path in candidate_files if path not in read_files]
        if unread:
            return Action(
                "read_file",
                {"file_path": unread[0]},
                thought=f"阅读候选文件 `{unread[0]}`。",
            )

        if not state.get("test_results"):
            command = state.get("verify_command") or "pytest"
            return Action(
                "run_tests",
                {"command": command},
                thought=f"运行验证命令 `{command}`。",
            )

        if state.get("patch_summary") is None:
            return Action(
                "git_diff",
                thought="检查当前工作区 diff，判断是否已有补丁。",
            )

        if not state.get("memory_written"):
            return Action(
                "write_memory",
                thought="将本次任务轨迹沉淀为第一版 episodic memory。",
            )

        if loop_count >= max_loops:
            return Action("finish", thought="达到最大循环次数，结束本轮执行。")

        return Action("finish", thought="已完成第一版可执行流程。")

    def extract_keyword(self, state: AgentState) -> str:
        return self.query_planner.plan(state).query

    def _read_files_from_state(self, state: AgentState) -> set[str]:
        files: set[str] = set()
        for observation in state.get("observations", []):
            content = observation.get("content", {})
            if isinstance(content, dict) and content.get("file_path"):
                files.add(content["file_path"])
        return files
