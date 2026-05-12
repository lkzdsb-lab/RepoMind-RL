"""
RepoMind-RL first-version agent executor.

The executor owns orchestration only. Planning/action selection, tool execution,
memory persistence, and trajectory recording are separate collaborators so each
layer can evolve independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from agent_runtime.memory.cards import MemoryCard
from agent_runtime.memory.store import JsonlMemoryStore
from agent_runtime.policy import HeuristicDebugPolicy
from agent_runtime.registry import RegistryManager, RegistrySnapshot
from agent_runtime.tool_registry import ToolRegistry
from agent_runtime.trajectory import TrajectoryRecorder
from model.agent.graph import AgentState, DebugAgentConfig, AgentRunResult
from model.agent.actions import Action


class DebugAgent:
    def __init__(
        self,
        config: DebugAgentConfig,
        policy: HeuristicDebugPolicy | None = None,
        tools: ToolRegistry | None = None,
        registry: RegistryManager | None = None,
        memory_store: JsonlMemoryStore | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or HeuristicDebugPolicy()
        self.registry_manager = registry or RegistryManager(
            tools=tools,
            manifest_dir=config.manifest_dir,
        )
        self._active_registry: RegistrySnapshot | None = None
        self.memory_store = memory_store or JsonlMemoryStore(
            Path(config.repo_path) / config.memory_path
        )
        self.recorder = recorder or TrajectoryRecorder()

    def run(self, title: str, description: str = "") -> AgentRunResult:
        self._active_registry = self.registry_manager.snapshot()
        state = self._initial_state(title=title, description=description)
        state = self._understand_task(state)
        state = self._retrieve_memories(state)
        state = self._make_plan(state)

        while state.get("loop_count", 0) < state.get("max_loops", self.config.max_loops):
            action = self.policy.next_action(state)
            state = self._record_action_selection(state, action)

            if action.name == "finish":
                state = self._finalize(state)
                break

            state = self._execute_action(state, action)
            state = self._observe(state)

        else:
            state = {
                **state,
                "status": "failed",
                "error": "Reached max_loops before finishing.",
            }
            state = self.recorder.append(
                state,
                node="finalize",
                thought="达到最大循环次数，第一版 agent 停止执行。",
            )

        trace_path = self.recorder.save(state, self.config.trace_dir)
        return AgentRunResult(state=state, trace_path=trace_path.as_posix())

    def _initial_state(self, title: str, description: str) -> AgentState:
        registry = self._registry()
        return AgentState(
            task_id=str(uuid4()),
            task_type="BUG_FIX",
            title=title,
            description=description,
            registry_snapshot={
                "tools": registry.names("tools"),
                "nodes": registry.names("nodes"),
                "prompts": registry.names("prompts"),
                "skills": registry.names("skills"),
            },
            repo_path=self.config.repo_path,
            branch="",
            verify_command=self.config.verify_command,
            plan=[],
            current_step="created",
            candidate_files=[],
            observations=[],
            tool_calls=[],
            test_results=[],
            trajectory=[],
            patch=None,
            patch_summary=None,
            next_action=None,
            next_action_input=None,
            loop_count=0,
            max_loops=self.config.max_loops,
            status="created",
            error=None,
        )

    def _understand_task(self, state: AgentState) -> AgentState:
        state = {**state, "status": "running", "current_step": "understand_task"}
        return self.recorder.append(
            state,
            node="understand_task",
            thought=f"理解任务：{state.get('title', '')}",
        )

    def _retrieve_memories(self, state: AgentState) -> AgentState:
        query = f"{state.get('title', '')} {state.get('description', '')}"
        memories = self.memory_store.search(query)
        observations = state.get("observations", []) + [
            {"type": "retrieved_memories", "content": memories}
        ]
        state = {**state, "observations": observations, "retrieved_memories": memories}
        return self.recorder.append(
            state,
            node="retrieve_memory",
            thought=f"检索到 {len(memories)} 条历史记忆。",
            observation={"count": len(memories), "memories": memories},
        )

    def _make_plan(self, state: AgentState) -> AgentState:
        plan = self.policy.make_initial_plan(state)
        state = {
            **state,
            "plan": plan,
            "current_step": "select_action",
        }
        return self.recorder.append(
            state,
            node="make_plan",
            thought="生成第一版调试计划。",
            observation={"plan": plan},
        )

    def _record_action_selection(self, state: AgentState, action: Action) -> AgentState:
        state = {
            **state,
            "next_action": action.name,
            "next_action_input": action.args,
            "current_step": action.name,
        }
        return self.recorder.append(
            state,
            node="select_action",
            thought=action.thought,
            action=action.name,
            action_input=action.args,
        )

    def _execute_action(self, state: AgentState, action: Action) -> AgentState:
        if action.name == "write_memory":
            output = self._write_memory(state)
        else:
            output = self._registry().run_tool(
                action.name,
                self.config.repo_path,
                action.args,
            )

        state = self._apply_tool_output(state, action, output)
        state = {
            **state,
            "loop_count": state.get("loop_count", 0) + 1,
        }
        return self.recorder.append(
            state,
            node="execute_action",
            thought=f"执行动作：{action.name}",
            action=action.name,
            action_input=action.args,
            observation=output,
        )

    def _apply_tool_output(
        self,
        state: AgentState,
        action: Action,
        output: Dict[str, Any],
    ) -> AgentState:
        updates: Dict[str, Any] = {}

        tool_spec = self._registry().get_tool(action.name)
        if tool_spec and tool_spec.reducer:
            updates.update(tool_spec.reducer(state, output))
        elif action.name == "write_memory":
            updates["memory_written"] = True

        tool_calls = state.get("tool_calls", []) + [
            {
                "name": action.name,
                "input": action.args,
                "output": output,
                "error": output.get("error"),
            }
        ]
        observations = state.get("observations", []) + [
            {
                "type": "tool_output",
                "tool": action.name,
                "content": output,
            }
        ]

        if output.get("error"):
            updates["error"] = output["error"]

        return {
            **state,
            **updates,
            "tool_calls": tool_calls,
            "observations": observations,
        }

    def _observe(self, state: AgentState) -> AgentState:
        latest = state.get("tool_calls", [{}])[-1]
        return self.recorder.append(
            state,
            node="observe",
            thought=f"整理 `{latest.get('name', 'unknown')}` 的结果。",
            observation={"latest_tool": latest.get("name"), "error": latest.get("error")},
        )

    def _write_memory(self, state: AgentState) -> dict:
        latest_test = (state.get("test_results") or [{}])[-1]
        passed = latest_test.get("exit_code") == 0
        memory_type = "episodic" if passed else "anti_pattern"
        status = "verified" if passed else "draft"
        evidence = []
        if latest_test:
            evidence.append(f"verify_command={latest_test.get('command', '')}")
            evidence.append(f"exit_code={latest_test.get('exit_code')}")
        if state.get("patch"):
            evidence.append("git_diff_present=true")

        card = MemoryCard(
            type=memory_type,
            scope=state.get("repo_path", ""),
            trigger=state.get("title", ""),
            content=self._memory_content(state),
            evidence=evidence,
            reward_credit=1.0 if passed else -0.2,
            status=status,
        )
        return self.memory_store.append(card)

    def _memory_content(self, state: AgentState) -> str:
        candidates = ", ".join(state.get("candidate_files", [])[:5]) or "none"
        tests = state.get("test_results") or []
        latest_exit = tests[-1].get("exit_code") if tests else "not_run"
        return (
            f"Task: {state.get('title', '')}\n"
            f"Candidate files: {candidates}\n"
            f"Latest test exit code: {latest_exit}\n"
            f"Patch summary: {state.get('patch_summary') or 'no patch'}"
        )

    def _tests_passed(self, state: AgentState) -> bool:
        tests = state.get("test_results") or []
        return bool(tests and tests[-1].get("exit_code") == 0)

    def _finalize(self, state: AgentState) -> AgentState:
        status = "finished" if not state.get("error") else "failed"
        state = {**state, "status": status, "current_step": "finished"}
        return self.recorder.append(
            state,
            node="finalize",
            thought="任务执行结束，输出最终结果。",
            observation={
                "status": status,
                "candidate_files": state.get("candidate_files", []),
                "patch_summary": state.get("patch_summary"),
            },
        )

    def _registry(self) -> RegistrySnapshot:
        if self._active_registry is None:
            self._active_registry = self.registry_manager.snapshot()
        return self._active_registry
