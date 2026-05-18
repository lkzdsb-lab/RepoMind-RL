"""
RepoMind-RL first-version agent executor.

The executor owns orchestration only. Planning/action selection, tool execution,
memory persistence, and trajectory recording are separate collaborators so each
layer can evolve independently.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from agent_runtime.context import ContextCompressionManager
from agent_runtime.logging_config import configure_from_agent_config
from agent_runtime.memory.manager import LayeredMemoryManager
from agent_runtime.memory.store import JsonlMemoryStore
from agent_runtime.policy import HeuristicDebugPolicy
from agent_runtime.registry import RegistryManager, RegistrySnapshot
from agent_runtime.rl import (
    ActionSpace,
    QLearningDebugPolicy,
    QLearningTrainer,
    ReplayBuffer,
    RewardFunction,
    StateEncoder,
    Transition,
)
from agent_runtime.rl.trainer import QTableStore
from agent_runtime.tool_registry import ToolRegistry
from agent_runtime.trajectory import TrajectoryRecorder
from model.agent.graph import AgentState, AgentRunResult
from model.agent.actions import Action
from config import DebugAgentConfig
from loguru import logger


class DebugAgent:
    """
        主 agent，后续考虑 subagent 去跑主流程外的 mcp 以及 skill 等服务
        具备 tools 注册、memory 管理与存储、上下文管理以及持久话、任务并行等功能
    """
    def __init__(
        self,
        config: DebugAgentConfig,
        policy: HeuristicDebugPolicy | None = None,
        tools: ToolRegistry | None = None,
        registry: RegistryManager | None = None,
        memory_manager: LayeredMemoryManager | None = None,
        context_manager: ContextCompressionManager | None = None,
        memory_store: JsonlMemoryStore | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        configure_from_agent_config(config)
        self.config = config
        self.rl_enabled = config.rl_enabled
        self.rl_encoder = StateEncoder()
        self.rl_action_space = ActionSpace()
        self.rl_reward = RewardFunction()
        repo_path = Path(config.repo_path)
        self.rl_q_store = QTableStore(repo_path / config.rl_q_table_path)
        self.rl_q_table = self.rl_q_store.load() if self.rl_enabled else {}
        self.rl_replay = (
            ReplayBuffer(repo_path / config.rl_replay_path, max_size=config.rl_replay_max_size)
            if self.rl_enabled
            else None
        )
        self.rl_trainer = (
            QLearningTrainer(
                self.rl_q_table,
                self.rl_action_space,
                learning_rate=config.rl_learning_rate,
                discount=config.rl_discount,
            )
            if self.rl_enabled
            else None
        )
        self.policy = policy or self._default_policy()
        self.registry_manager = registry or RegistryManager(
            tools=tools,
            manifest_dir=config.manifest_dir,
        )
        self._active_registry: RegistrySnapshot | None = None
        self.memory_manager = memory_manager or LayeredMemoryManager.from_config(config)
        if memory_store is not None:
            self.memory_manager.mid_store = memory_store
        self.context_manager = context_manager or ContextCompressionManager.from_config(config)
        self.recorder = recorder or TrajectoryRecorder()
        logger.info(
            "debug agent initialized repo_path={} max_loops={} rl_enabled={} manifest_dir={}",
            config.repo_path,
            config.max_loops,
            self.rl_enabled,
            config.manifest_dir,
        )

    def run(self, title: str, description: str = "") -> AgentRunResult:
        started_at = time.perf_counter()
        self._active_registry = self.registry_manager.snapshot()
        state = self._initial_state(title=title, description=description)
        run_logger = logger.bind(task_id=state.get("task_id"), repo_path=self.config.repo_path)
        run_logger.info(
            "agent run started title={} description_present={}",
            title,
            bool(description),
        )
        state = self._understand_task(state)
        state = self._retrieve_memories(state)
        state = self._prepare_context(state)
        state = self._make_plan(state)

        while state.get("loop_count", 0) < state.get("max_loops", self.config.max_loops):
            state = self._prepare_context(state)
            action = self.policy.next_action(state)
            state = self._record_action_selection(state, action)

            if action.name == "finish":
                prev_state = state
                state = self._finalize(state)
                state = self._record_rl_transition(prev_state, action, state, {}, done=True)
                break

            state = self._execute_action(state, action)
            state = self._observe(state)

        else:
            state = {
                **state,
                "status": "failed",
                "error": "Reached max_loops before finishing.",
            }
            run_logger.warning(
                "agent run reached max loops max_loops={} tool_calls={}",
                self.config.max_loops,
                len(state.get("tool_calls", [])),
            )
            state = self.recorder.append(
                state,
                node="finalize",
                thought="达到最大循环次数，第一版 agent 停止执行。",
            )

        trace_path = self.recorder.save(state, self.config.trace_dir)
        logger.bind(task_id=state.get("task_id")).info(
            "agent run finished status={} loops={} tool_calls={} trace_path={} elapsed_ms={:.1f}",
            state.get("status"),
            state.get("loop_count"),
            len(state.get("tool_calls", [])),
            trace_path.as_posix(),
            (time.perf_counter() - started_at) * 1000,
        )
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
            code_context={},
            observations=[],
            tool_calls=[],
            test_results=[],
            trajectory=[],
            selected_skills=[],
            skill_context=[],
            retrieved_memories=[],
            memory_context="",
            context_items=[],
            context_digest={},
            compressed_context="",
            short_term_memories=[],
            promoted_memories=[],
            consolidated_skills=[],
            memory_written=False,
            rl_enabled=self.rl_enabled,
            rl_transitions=[],
            rl_last_reward={},
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
        memory_pack = self.memory_manager.retrieve(query, state, self._registry())
        memories = memory_pack.to_dict()
        memory_context = memory_pack.render_for_prompt()
        skill_context = [result.to_dict() for result in memory_pack.skill]
        selected_skills = [
            result.card.skill_name
            for result in memory_pack.skill
            if result.card.skill_name
        ]
        observations = state.get("observations", []) + [
            {"type": "retrieved_memories", "content": memories}
        ]
        state = {
            **state,
            "observations": observations,
            "retrieved_memories": memories,
            "memory_context": memory_context,
            "skill_context": skill_context,
            "selected_skills": selected_skills,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "memory retrieved total={} short={} mid={} long={} skill={} selected_skills={}",
            len(memory_pack.all_results()),
            len(memory_pack.short_term),
            len(memory_pack.mid_term),
            len(memory_pack.long_term),
            len(memory_pack.skill),
            selected_skills,
        )
        return self.recorder.append(
            state,
            node="retrieve_memory",
            thought=f"检索到 {len(memory_pack.all_results())} 条分层记忆。",
            observation={
                "count": len(memory_pack.all_results()),
                "memories": memories,
                "memory_context": memory_context,
            },
        )

    def _prepare_context(self, state: AgentState) -> AgentState:
        new_state = self.context_manager.prepare(state)
        if new_state.get("context_digest") != state.get("context_digest"):
            logger.bind(task_id=state.get("task_id")).info(
                "context compressed method={} context_items={}",
                new_state.get("context_digest", {}).get("compression_method"),
                len(new_state.get("context_items", [])),
            )
            return self.recorder.append(
                new_state,
                node="compress_context",
                thought="上下文过长，已生成压缩摘要供后续 LLM 使用。",
                observation={
                    "compression_method": new_state.get("context_digest", {}).get(
                        "compression_method"
                    ),
                    "context_items": len(new_state.get("context_items", [])),
                },
            )
        return new_state

    def _make_plan(self, state: AgentState) -> AgentState:
        plan = self.policy.make_initial_plan(state)
        state = {
            **state,
            "plan": plan,
            "current_step": "select_action",
        }
        logger.bind(task_id=state.get("task_id")).info("initial plan created steps={}", len(plan))
        return self.recorder.append(
            state,
            node="make_plan",
            thought="生成第一版调试计划。",
            observation={"plan": plan},
        )

    def _record_action_selection(self, state: AgentState, action: Action) -> AgentState:
        logger.bind(task_id=state.get("task_id"), action=action.name).info(
            "action selected args={}",
            action.args,
        )
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
        action_logger = logger.bind(task_id=state.get("task_id"), action=action.name)
        started_at = time.perf_counter()
        action_logger.info("action execution started")
        try:
            if action.name == "write_memory":
                output = self._write_memory(state)
            else:
                action_args = dict(action.args)
                if action.name in {"build_codebase_context", "search_code_context"}:
                    action_args.setdefault("index_path", self.config.code_context_index_path)
                output = self._registry().run_tool(
                    action.name,
                    self.config.repo_path,
                    action_args,
                )
        except Exception as exc:
            action_logger.exception("action execution raised exception")
            output = {"error": str(exc), "exception_type": exc.__class__.__name__}
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if output.get("error"):
            action_logger.warning(
                "action execution completed with error elapsed_ms={:.1f} error={}",
                elapsed_ms,
                output.get("error"),
            )
        else:
            action_logger.info("action execution completed elapsed_ms={:.1f}", elapsed_ms)

        prev_state = state
        state = self._apply_tool_output(state, action, output)
        state = {
            **state,
            "loop_count": state.get("loop_count", 0) + 1,
        }
        state = self._record_rl_transition(prev_state, action, state, output, done=False)
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
            updates["promoted_memories"] = output.get("promoted", [])
            updates["consolidated_skills"] = output.get("consolidated", [])

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
            logger.bind(task_id=state.get("task_id"), action=action.name).warning(
                "tool output contains error error={}",
                output["error"],
            )

        new_state = {
            **state,
            **updates,
            "tool_calls": tool_calls,
            "observations": observations,
        }
        if action.name != "write_memory":
            new_state = self.memory_manager.add_short_term(
                new_state,
                trigger=f"tool:{action.name}",
                content=self._short_term_tool_content(action.name, output),
                tags=[action.name],
            )
        return new_state

    def _observe(self, state: AgentState) -> AgentState:
        latest = state.get("tool_calls", [{}])[-1]
        return self.recorder.append(
            state,
            node="observe",
            thought=f"整理 `{latest.get('name', 'unknown')}` 的结果。",
            observation={"latest_tool": latest.get("name"), "error": latest.get("error")},
        )

    def _write_memory(self, state: AgentState) -> dict:
        result = self.memory_manager.record_task_memory(state, self._registry()).to_dict()
        logger.bind(task_id=state.get("task_id")).info(
            "memory written written={} promoted={} consolidated={}",
            len(result.get("written", [])),
            len(result.get("promoted", [])),
            len(result.get("consolidated", [])),
        )
        return result

    def _record_rl_transition(
        self,
        prev_state: AgentState,
        action: Action,
        next_state: AgentState,
        output: Dict[str, Any],
        done: bool,
    ) -> AgentState:
        if not self.rl_enabled:
            return next_state

        prev_encoded = self.rl_encoder.encode(prev_state)
        next_encoded = self.rl_encoder.encode(next_state)
        reward = self.rl_reward.compute(prev_state, action, next_state, output)
        transition = Transition(
            state_key=prev_encoded.key,
            action=action.name,
            action_args=action.args,
            reward=reward.reward,
            reward_reasons=reward.reasons,
            next_state_key=next_encoded.key,
            done=done,
            state_features=prev_encoded.features,
            next_state_features=next_encoded.features,
            task_id=next_state.get("task_id", ""),
        )
        if self.rl_replay is not None:
            self.rl_replay.append(transition)
        if self.rl_trainer is not None:
            self.rl_trainer.update(transition)
            if self.rl_replay is not None:
                self.rl_trainer.train_batch(
                    self.rl_replay.sample(self.config.rl_train_batch_size)
                )
            self.rl_q_store.save(self.rl_q_table)

        transitions = next_state.get("rl_transitions", []) + [transition.to_dict()]
        logger.bind(task_id=next_state.get("task_id"), action=action.name).debug(
            "rl transition recorded reward={:.3f} reasons={}",
            reward.reward,
            reward.reasons,
        )
        return {
            **next_state,
            "rl_transitions": transitions,
            "rl_last_reward": reward.to_dict(),
        }

    def _short_term_tool_content(self, tool_name: str, output: Dict[str, Any]) -> str:
        if output.get("error"):
            return f"{tool_name} failed: {output.get('error')}"
        if tool_name == "search_code":
            matches = output.get("matches", [])
            return f"{tool_name} returned {len(matches)} matches."
        if tool_name == "search_code_context":
            return (
                f"{tool_name} returned {len(output.get('files', []))} files, "
                f"{len(output.get('functions', []))} functions, "
                f"{len(output.get('api_routes', []))} routes."
            )
        if tool_name == "read_file":
            return f"{tool_name} read {output.get('file_path', 'unknown file')}."
        if tool_name == "run_tests":
            return f"{tool_name} exit_code={output.get('exit_code')} command={output.get('command')}"
        if tool_name == "git_diff":
            diff = output.get("diff", "")
            return f"{tool_name} returned {len(diff.splitlines())} diff lines."
        return f"{tool_name} output keys: {', '.join(sorted(output.keys()))}"

    def _finalize(self, state: AgentState) -> AgentState:
        status = "finished" if not state.get("error") else "failed"
        state = {**state, "status": status, "current_step": "finished"}
        if self.rl_enabled:
            terminal = self.rl_reward.terminal_reward(state)
            state = {
                **state,
                "rl_last_reward": terminal.to_dict(),
            }
        logger.bind(task_id=state.get("task_id")).info(
            "finalizing run status={} error={}",
            status,
            state.get("error"),
        )
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

    def _default_policy(self):
        if not self.rl_enabled:
            return HeuristicDebugPolicy()
        return QLearningDebugPolicy(
            q_table=self.rl_q_table,
            epsilon=self.config.rl_epsilon,
            encoder=self.rl_encoder,
            action_space=self.rl_action_space,
        )
