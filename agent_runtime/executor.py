"""
RepoMind-RL first-version agent executor.

The executor owns orchestration only. Planning/action selection, tool execution,
memory persistence, and trajectory recording are separate collaborators so each
layer can evolve independently.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from agent_runtime.codebase_context.retrieval import (
    CodeContextQueryPlanner,
    CodeContextReranker,
    CONTEXT_LIST_KEYS,
    DisabledCodeContextQueryPlanner,
    DisabledCodeContextReranker,
    LLMCodeContextQueryPlanner,
    LLMCodeContextReranker,
    merge_code_context_outputs,
)
from agent_runtime.context import ContextCompressionManager
from agent_runtime.logging_config import configure_from_agent_config
from agent_runtime.llm.llm_policy import LLMActionPolicy
from agent_runtime.llm.completion_judge import (
    CompletionJudge,
    LLMCompletionJudge,
    RuleBasedCompletionJudge,
)
from agent_runtime.llm.final_reporter import (
    FinalReporter,
    LLMFinalReporter,
    RuleBasedFinalReporter,
)
from agent_runtime.llm.observation import DisabledObserver, LLMObserver, Observer
from agent_runtime.llm.task_analysis import DisabledTaskAnalyzer, LLMTaskAnalyzer, TaskAnalyzer
from agent_runtime.memory.manager import LayeredMemoryManager
from agent_runtime.memory.retrieval import (
    DisabledMemoryQueryPlanner,
    DisabledMemoryReranker,
    LLMMemoryQueryPlanner,
    LLMMemoryReranker,
    MemoryQueryPlanner,
    MemoryReranker,
    merge_memory_packs,
)
from agent_runtime.memory.store import JsonlMemoryStore
from agent_runtime.planning import HeuristicPlanner, LLMPlanner, Planner
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
from agent_runtime.skill_selection import DisabledSkillSelector, LLMSkillSelector, SkillSelector
from agent_runtime.rl.trainer import QTableStore
from agent_runtime.tool_registry import ToolRegistry
from agent_runtime.trajectory import TrajectoryRecorder
from model.agent.graph import AgentState, AgentRunResult
from model.agent.actions import Action
from config import CompressionConfig, DebugAgentConfig, LLMConfig
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
        planner: Planner | None = None,
        task_analyzer: TaskAnalyzer | None = None,
        observer: Observer | None = None,
        final_reporter: FinalReporter | None = None,
        completion_judge: CompletionJudge | None = None,
        memory_query_planner: MemoryQueryPlanner | None = None,
        memory_reranker: MemoryReranker | None = None,
        code_context_query_planner: CodeContextQueryPlanner | None = None,
        code_context_reranker: CodeContextReranker | None = None,
        skill_selector: SkillSelector | None = None,
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
        self.rl_q_table = (
            self.rl_q_store.load()
            if self.rl_enabled or (config.action_policy_mode or "").strip().lower() == "llm"
            else {}
        )
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
        self.planner = planner or self._default_planner()
        self.task_analyzer = task_analyzer or self._default_task_analyzer()
        self.observer = observer or self._default_observer()
        self.final_reporter = final_reporter or self._default_final_reporter()
        self.completion_judge = completion_judge or self._default_completion_judge()
        self.memory_query_planner = memory_query_planner or self._default_memory_query_planner()
        self.memory_reranker = memory_reranker or self._default_memory_reranker()
        self.code_context_query_planner = (
            code_context_query_planner or self._default_code_context_query_planner()
        )
        self.code_context_reranker = code_context_reranker or self._default_code_context_reranker()
        self.registry_manager = registry or RegistryManager(
            tools=tools,
            manifest_dir=config.manifest_dir,
        )
        self.skill_selector = skill_selector or self._default_skill_selector()
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
        if state.get("status") == "failed":
            state = self._finalize(state)
            trace_path = self.recorder.save(state, self.config.trace_dir)
            run_logger.error(
                "agent run aborted during task analysis error={} trace_path={}",
                state.get("error"),
                trace_path.as_posix(),
            )
            return AgentRunResult(state=state, trace_path=trace_path.as_posix())
        state = self._select_skills(state)
        if state.get("status") == "failed":
            state = self._finalize(state)
            trace_path = self.recorder.save(state, self.config.trace_dir)
            run_logger.error(
                "agent run aborted during skill selection error={} trace_path={}",
                state.get("error"),
                trace_path.as_posix(),
            )
            return AgentRunResult(state=state, trace_path=trace_path.as_posix())
        state = self._retrieve_memories(state)
        if state.get("status") == "failed":
            state = self._finalize(state)
            trace_path = self.recorder.save(state, self.config.trace_dir)
            run_logger.error(
                "agent run aborted during memory retrieval error={} trace_path={}",
                state.get("error"),
                trace_path.as_posix(),
            )
            return AgentRunResult(state=state, trace_path=trace_path.as_posix())
        state = self._prepare_context(state)
        state = self._make_plan(state)

        return self._run_action_loop(state, started_at)

    def resume(self, state: AgentState, user_answer: str = "") -> AgentRunResult:
        started_at = time.perf_counter()
        self._active_registry = self.registry_manager.snapshot()
        if user_answer:
            state = self._inject_user_input(state, user_answer)
        else:
            state = {
                **state,
                "status": "running",
                "current_step": "select_action",
            }
        logger.bind(task_id=state.get("task_id"), repo_path=self.config.repo_path).info(
            "agent run resumed user_answer_present={} prior_inputs={}",
            bool(user_answer),
            len(state.get("user_inputs", [])),
        )
        return self._run_action_loop(state, started_at)

    def _run_action_loop(
        self,
        state: AgentState,
        started_at: float,
    ) -> AgentRunResult:
        run_logger = logger.bind(task_id=state.get("task_id"), repo_path=self.config.repo_path)
        while state.get("loop_count", 0) < state.get("max_loops", self.config.max_loops):
            state = self._prepare_context(state)
            action = self.policy.next_action(state)
            state = self._record_action_selection(state, action)

            if action.name == "finish":
                prev_state = state
                state, should_stop, done, output = self._handle_finish_action(state)
                state = self._record_rl_transition(prev_state, action, state, output, done=done)
                if should_stop:
                    break
                continue

            state = self._execute_action(state, action)
            if state.get("status") == "failed":
                break
            state = self._observe(state)
            if state.get("status") == "failed":
                break

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
        if state.get("status") != "awaiting_user_input" and state.get("current_step") != "finished":
            state = self._finalize(state)

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
        project_profile = _build_project_profile(self.config.repo_path)
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
            project_profile=project_profile,
            branch="",
            verification_required=True,
            verification_reason="Task analysis has not run yet.",
            verify_command=self.config.verify_command,
            task_analysis={},
            plan=[],
            current_step="created",
            candidate_files=[],
            code_context={},
            selected_code_context={},
            code_context_query_plan={},
            code_context_rerank={},
            observations=[],
            llm_observations=[],
            tool_calls=[],
            test_results=[],
            trajectory=[],
            completion_judgement={},
            pending_user_questions=[],
            needs_user_input_reason="",
            user_inputs=[],
            selected_skills=[],
            skill_selection={},
            skill_context=[],
            retrieved_memories={},
            selected_memories={},
            memory_query_plan={},
            memory_rerank={},
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
            llm_guard_events=[],
            patch=None,
            patch_summary=None,
            final_report={},
            next_action=None,
            next_action_input=None,
            loop_count=0,
            max_loops=self.config.max_loops,
            status="created",
            error=None,
        )

    def _understand_task(self, state: AgentState) -> AgentState:
        try:
            analysis = self.task_analyzer.analyze(state)
        except Exception as exc:
            logger.bind(task_id=state.get("task_id")).exception(
                "task analysis failed without fallback"
            )
            failed_state = {
                **state,
                "status": "failed",
                "current_step": "understand_task",
                "error": f"Task analysis failed: {exc}",
            }
            return self.recorder.append(
                failed_state,
                node="understand_task",
                thought="任务分析失败，未启用降级策略。",
                observation={"error": str(exc), "fallback": False},
            )
        task_type = str(analysis.get("task_type") or state.get("task_type") or "BUG_FIX").upper()
        if task_type not in {"BUG_FIX", "FEATURE_IMPL", "DIAGNOSE"}:
            task_type = "BUG_FIX"
        verification_required = bool(analysis.get("verification_required", True))
        verification_reason = str(analysis.get("verification_reason") or "")
        task_category = str(analysis.get("task_category") or state.get("task_category") or "")
        observations = state.get("observations", []) + [
            {"type": "task_analysis", "content": analysis}
        ]
        state = {
            **state,
            "status": "running",
            "current_step": "understand_task",
            "task_type": task_type,
            "verification_required": verification_required,
            "verification_reason": verification_reason,
            "task_category": task_category,
            "task_analysis": analysis,
            "observations": observations,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "task analyzed type={} verification_required={} category={} entities={} source={}",
            task_type,
            verification_required,
            task_category,
            analysis.get("entities", []),
            analysis.get("source"),
        )
        return self.recorder.append(
            state,
            node="understand_task",
            thought=f"理解任务：{state.get('title', '')}",
            observation={"task_analysis": analysis},
        )

    def _select_skills(self, state: AgentState) -> AgentState:
        try:
            selection = self.skill_selector.select(state, self._registry().skills)
        except Exception as exc:
            logger.bind(task_id=state.get("task_id")).exception(
                "skill selection failed without fallback"
            )
            failed_state = {
                **state,
                "status": "failed",
                "current_step": "select_skills",
                "error": f"Skill selection failed: {exc}",
            }
            return self.recorder.append(
                failed_state,
                node="select_skills",
                thought="Skill 选择失败，未启用降级策略。",
                observation={"error": str(exc), "fallback": False},
            )

        selected_skills = _merge_unique(
            state.get("selected_skills", []),
            selection.selected_skills,
        )
        skill_context = _merge_skill_context(
            state.get("skill_context", []),
            _selected_skill_context(selected_skills, self._registry().skills),
        )
        selection_payload = selection.to_dict()
        observations = state.get("observations", []) + [
            {
                "type": "skill_selection",
                "content": selection_payload,
            }
        ]
        state = {
            **state,
            "current_step": "select_skills",
            "selected_skills": selected_skills,
            "skill_context": skill_context,
            "skill_selection": selection_payload,
            "observations": observations,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "skills selected source={} count={} selected={}",
            selection.source,
            len(selected_skills),
            selected_skills,
        )
        return self.recorder.append(
            state,
            node="select_skills",
            thought=f"选择 {len(selected_skills)} 个 skill 作为本轮流程约束。",
            observation=selection_payload,
        )

    def _retrieve_memories(self, state: AgentState) -> AgentState:
        try:
            query_plan = self.memory_query_planner.plan(state)
            packs = [
                self.memory_manager.retrieve(
                    query,
                    state,
                    self._registry(),
                    limit=self.config.memory_query_limit,
                    touch=False,
                )
                for query in query_plan.queries
            ]
            candidate_pack = merge_memory_packs(packs)
            memory_pack, rerank_decision = self.memory_reranker.rerank(
                state,
                query_plan,
                candidate_pack,
            )
            self.memory_manager.touch_retrieved(memory_pack, state)
        except Exception as exc:
            logger.bind(task_id=state.get("task_id")).exception(
                "memory retrieval failed without fallback"
            )
            failed_state = {
                **state,
                "status": "failed",
                "current_step": "retrieve_memory",
                "error": f"Memory retrieval failed: {exc}",
            }
            return self.recorder.append(
                failed_state,
                node="retrieve_memory",
                thought="记忆检索失败，未启用降级策略。",
                observation={"error": str(exc), "fallback": False},
            )

        candidate_memories = candidate_pack.to_dict()
        selected_memories = memory_pack.to_dict()
        memory_context = memory_pack.render_for_prompt()
        skill_context = _merge_skill_context(
            state.get("skill_context", []),
            [result.to_dict() for result in memory_pack.skill],
        )
        memory_selected_skills = [
            result.card.skill_name
            for result in memory_pack.skill
            if result.card.skill_name
        ]
        selected_skills = _merge_unique(state.get("selected_skills", []), memory_selected_skills)
        observations = state.get("observations", []) + [
            {
                "type": "retrieved_memories",
                "content": {
                    "query_plan": query_plan.to_dict(),
                    "candidate_count": len(candidate_pack.all_results()),
                    "selected_count": len(memory_pack.all_results()),
                    "rerank": rerank_decision.to_dict(),
                },
            }
        ]
        state = {
            **state,
            "observations": observations,
            "retrieved_memories": candidate_memories,
            "selected_memories": selected_memories,
            "memory_query_plan": query_plan.to_dict(),
            "memory_rerank": rerank_decision.to_dict(),
            "memory_context": memory_context,
            "skill_context": skill_context,
            "selected_skills": selected_skills,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "memory retrieved queries={} candidates={} selected={} short={} mid={} long={} skill={} selected_skills={}",
            len(query_plan.queries),
            len(candidate_pack.all_results()),
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
            thought=f"检索到 {len(candidate_pack.all_results())} 条候选记忆，选中 {len(memory_pack.all_results())} 条。",
            observation={
                "query_plan": query_plan.to_dict(),
                "candidate_count": len(candidate_pack.all_results()),
                "selected_count": len(memory_pack.all_results()),
                "selected_memories": selected_memories,
                "rerank": rerank_decision.to_dict(),
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
        plan = self.planner.make_plan(state)
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
            observation=action.metadata or None,
        )

    def _execute_action(self, state: AgentState, action: Action) -> AgentState:
        """
            执行
        """
        action_logger = logger.bind(task_id=state.get("task_id"), action=action.name)
        started_at = time.perf_counter()
        action_logger.info("action execution started")
        try:
            if action.name == "run_tests" and not _verification_required(state):
                output = {
                    "command": action.args.get("command", self.config.verify_command),
                    "skipped": True,
                    "reason": "verification_required=false",
                }
            elif action.name == "write_memory":
                output = self._write_memory(state)
            elif action.name == "search_code_context":
                output = self._search_code_context(state, action)
            else:
                action_args = dict(action.args)
                if action.name == "build_codebase_context":
                    action_args.setdefault("index_path", self.config.code_context_index_path)
                output = self._registry().run_tool(
                    action.name,
                    self.config.repo_path,
                    action_args,
                )
        except Exception as exc:
            action_logger.exception("action execution raised exception")
            output = {"error": str(exc), "exception_type": exc.__class__.__name__}
            if action.name == "search_code_context" and (
                (self.config.code_context_query_planner_mode or "").strip().lower() == "llm"
                or (self.config.code_context_reranker_mode or "").strip().lower() == "llm"
            ):
                output["fatal"] = True
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

    def _search_code_context(self, state: AgentState, action: Action) -> Dict[str, Any]:
        action_args = dict(action.args)
        action_args.setdefault("index_path", self.config.code_context_index_path)
        base_query = str(action_args.get("query", "")).strip()
        query_plan = self.code_context_query_planner.plan(state, base_query)
        queries = query_plan.queries or ([base_query] if base_query else [])
        if not queries:
            return {
                "error": "Code context query planner returned no queries.",
                "query_plan": query_plan.to_dict(),
            }

        outputs: list[dict[str, Any]] = []
        for query in queries:
            query_args = dict(action_args)
            query_args["query"] = query
            query_args["limit"] = int(self.config.code_context_query_limit)
            outputs.append(
                self._registry().run_tool(
                    action.name,
                    self.config.repo_path,
                    query_args,
                )
            )

        merged = merge_code_context_outputs(outputs)
        merged["queries"] = queries
        merged["query"] = " | ".join(queries)
        merged["query_plan"] = query_plan.to_dict()
        if merged.get("error"):
            return merged

        candidate_count = _code_context_candidate_count(merged)
        if candidate_count == 0:
            logger.bind(task_id=state.get("task_id")).warning(
                "code context search returned no candidates queries={} query_plan_source={} base_query={}",
                queries,
                query_plan.source,
                base_query,
            )

        selected_context, rerank_decision = self.code_context_reranker.rerank(
            state,
            query_plan,
            merged,
        )
        merged["selected_code_context"] = selected_context
        merged["code_context_rerank"] = rerank_decision.to_dict()
        selected_count = _code_context_candidate_count(selected_context)
        if candidate_count > 0 and selected_count == 0:
            logger.bind(task_id=state.get("task_id")).warning(
                "code context reranker selected no candidates queries={} query_plan_source={} candidate_count={} reranker_source={}",
                queries,
                query_plan.source,
                candidate_count,
                rerank_decision.source,
            )
        logger.bind(task_id=state.get("task_id")).info(
            "code context searched queries={} candidate_count={} candidates_files={} selected_count={} selected_ids={}",
            len(queries),
            candidate_count,
            len(merged.get("files", []) or []),
            selected_count,
            rerank_decision.selected_ids,
        )
        return merged

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
            if output.get("fatal"):
                updates["status"] = "failed"
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
        try:
            observation = self.observer.observe(state)
        except Exception as exc:
            logger.bind(task_id=state.get("task_id")).exception(
                "observation failed without fallback"
            )
            failed_state = {
                **state,
                "status": "failed",
                "current_step": "observe",
                "error": f"Observation failed: {exc}",
            }
            return self.recorder.append(
                failed_state,
                node="observe",
                thought="观察结果生成失败，未启用降级策略。",
                observation={
                    "latest_tool": latest.get("name"),
                    "error": str(exc),
                    "fallback": False,
                },
            )
        llm_observations = state.get("llm_observations", [])
        observations = state.get("observations", [])
        if observation.get("source") != "disabled":
            llm_observations = llm_observations + [observation]
            observations = observations + [observation]
        state = {
            **state,
            "llm_observations": llm_observations,
            "observations": observations,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "observation synthesized latest_tool={} status={} confidence={}",
            observation.get("latest_tool"),
            observation.get("status"),
            observation.get("confidence"),
        )
        return self.recorder.append(
            state,
            node="observe",
            thought=f"整理 `{latest.get('name', 'unknown')}` 的结果。",
            observation=observation,
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
            if output.get("skipped"):
                return f"{tool_name} skipped reason={output.get('reason')}"
            return f"{tool_name} exit_code={output.get('exit_code')} command={output.get('command')}"
        if tool_name == "git_diff":
            diff = output.get("diff", "")
            return f"{tool_name} returned {len(diff.splitlines())} diff lines."
        return f"{tool_name} output keys: {', '.join(sorted(output.keys()))}"

    def _handle_finish_action(
        self,
        state: AgentState,
    ) -> tuple[AgentState, bool, bool, Dict[str, Any]]:
        """
            处理终止状态，由 llm 判断是否结束，并且进行 loop 限制
        """
        judgement = self.completion_judge.judge(state)
        output = {"completion_judgement": judgement}
        state = self._record_completion_judgement(state, judgement)
        decision = str(judgement.get("decision") or "complete").strip().lower()
        if decision == "needs_user_input":
            state = self._await_user_input(state, judgement)
            return state, True, False, output
        if decision == "continue":
            attempts = int(state.get("completion_judge_continue_count", 0)) + 1
            state = {
                **state,
                "current_step": "select_action",
                "completion_judge_continue_count": attempts,
            }
            # todo 考虑配置化
            if attempts >= 2:
                logger.bind(task_id=state.get("task_id")).warning(
                    "completion judge requested continue repeatedly; finalizing to avoid loop"
                )
                state = self._finalize(state)
                return state, True, True, output
            return state, False, False, output
        state = self._finalize(state)
        return state, True, True, output

    def _record_completion_judgement(
        self,
        state: AgentState,
        judgement: dict[str, Any],
    ) -> AgentState:
        observations = state.get("observations", []) + [
            {
                "type": "completion_judgement",
                "content": judgement,
            }
        ]
        state = {
            **state,
            "completion_judgement": judgement,
            "observations": observations,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "completion judged decision={} confidence={} source={}",
            judgement.get("decision"),
            judgement.get("confidence"),
            judgement.get("source"),
        )
        return self.recorder.append(
            state,
            node="completion_judge",
            thought="判断当前状态是否足够结束，或是否需要用户补充信息。",
            observation=judgement,
        )

    def _await_user_input(
        self,
        state: AgentState,
        judgement: dict[str, Any],
    ) -> AgentState:
        questions = _clean_string_list(judgement.get("questions"), limit=3, max_chars=300)
        reason = str(judgement.get("reason") or "").strip()
        if not questions:
            questions = ["请补充当前任务缺失的具体目标、约束或期望判断标准。"]
        state = {
            **state,
            "status": "awaiting_user_input",
            "current_step": "awaiting_user_input",
            "pending_user_questions": questions,
            "needs_user_input_reason": reason,
        }
        logger.bind(task_id=state.get("task_id")).info(
            "agent awaiting user input questions={} reason={}",
            questions,
            reason,
        )
        return self.recorder.append(
            state,
            node="await_user_input",
            thought="当前信息不足，暂停并等待用户补充。",
            observation={
                "reason": reason,
                "questions": questions,
            },
        )

    def _inject_user_input(self, state: AgentState, answer: str) -> AgentState:
        text = str(answer or "").strip()
        pending_questions = _clean_string_list(
            state.get("pending_user_questions"),
            limit=5,
            max_chars=500,
        )
        input_item = {
            "questions": pending_questions,
            "answer": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        user_inputs = state.get("user_inputs", []) + [input_item]
        observations = state.get("observations", []) + [
            {
                "type": "user_input",
                "content": input_item,
            }
        ]
        description = str(state.get("description") or "")
        clarification = _render_user_clarification(input_item)
        if clarification:
            description = f"{description}\n\n{clarification}".strip()
        loop_count = int(state.get("loop_count", 0))
        max_loops = max(
            int(state.get("max_loops", self.config.max_loops)),
            loop_count + int(self.config.max_loops),
        )
        state = {
            **state,
            "description": description,
            "status": "running",
            "current_step": "user_input_received",
            "pending_user_questions": [],
            "needs_user_input_reason": "",
            "user_inputs": user_inputs,
            "observations": observations,
            "final_report": {},
            "error": None,
            "max_loops": max_loops,
        }
        return self.recorder.append(
            state,
            node="user_input_received",
            thought="收到用户补充信息，并写回当前 agent state。",
            observation=input_item,
        )

    def _finalize(self, state: AgentState) -> AgentState:
        status = "finished" if not state.get("error") else "failed"
        state = {**state, "status": status, "current_step": "finished"}
        if self.rl_enabled:
            terminal = self.rl_reward.terminal_reward(state)
            state = {
                **state,
                "rl_last_reward": terminal.to_dict(),
            }
        try:
            final_report = self.final_reporter.report(state)
        except Exception as exc:
            logger.bind(task_id=state.get("task_id")).exception("final report generation failed")
            final_report = RuleBasedFinalReporter().report(state)
            final_report["fallback_reason"] = str(exc)
        state = {**state, "final_report": final_report}
        logger.bind(task_id=state.get("task_id")).info(
            "finalizing run status={} error={} final_report_source={}",
            status,
            state.get("error"),
            final_report.get("source"),
        )
        return self.recorder.append(
            state,
            node="finalize",
            thought="任务执行结束，输出最终结果。",
            observation={
                "status": status,
                "candidate_files": state.get("candidate_files", []),
                "patch_summary": state.get("patch_summary"),
                "final_report": final_report,
            },
        )

    def _registry(self) -> RegistrySnapshot:
        if self._active_registry is None:
            self._active_registry = self.registry_manager.snapshot()
        return self._active_registry

    # 后面都是降级策略，项目完全实现后考虑删除。
    def _default_policy(self):
        mode = (self.config.action_policy_mode or "").strip().lower()
        if mode == "llm":
            return LLMActionPolicy(
                llm_config=_resolve_llm_config(self.config.llm_config, self.config.action_llm_config),
                action_space=self.rl_action_space,
                q_table=self.rl_q_table,
                encoder=self.rl_encoder,
                fallback=HeuristicDebugPolicy(),
            )
        if not self.rl_enabled and mode not in {"rl"}:
            return HeuristicDebugPolicy()
        return QLearningDebugPolicy(
            q_table=self.rl_q_table,
            epsilon=self.config.rl_epsilon,
            encoder=self.rl_encoder,
            action_space=self.rl_action_space,
        )

    def _default_planner(self) -> Planner:
        mode = (self.config.planner_mode or "").strip().lower()
        if mode == "llm":
            return LLMPlanner(
                llm_config=_resolve_llm_config(self.config.llm_config, self.config.plan_llm_config),
                fallback=HeuristicPlanner(),
            )
        return HeuristicPlanner()

    def _default_task_analyzer(self) -> TaskAnalyzer:
        mode = (self.config.task_analyzer_mode or "").strip().lower()
        if mode == "llm":
            return LLMTaskAnalyzer(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.task_analysis_llm_config,
                ),
            )
        return DisabledTaskAnalyzer()

    def _default_observer(self) -> Observer:
        mode = (self.config.observer_mode or "").strip().lower()
        if mode == "llm":
            return LLMObserver(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.observer_llm_config,
                ),
            )
        return DisabledObserver()

    def _default_final_reporter(self) -> FinalReporter:
        mode = (self.config.final_reporter_mode or "").strip().lower()
        fallback = RuleBasedFinalReporter()
        if mode == "llm":
            return LLMFinalReporter(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.final_reporter_llm_config,
                ),
                fallback=fallback,
            )
        return fallback

    def _default_completion_judge(self) -> CompletionJudge:
        mode = (self.config.completion_judge_mode or "").strip().lower()
        fallback = RuleBasedCompletionJudge()
        llm_config = _resolve_llm_config(
            self.config.llm_config,
            self.config.completion_judge_llm_config,
        )
        if mode == "auto" and not _llm_config_enabled(llm_config):
            return fallback
        if mode in {"auto", "llm"}:
            return LLMCompletionJudge(
                llm_config=llm_config,
                fallback=fallback,
            )
        return fallback

    def _default_memory_query_planner(self) -> MemoryQueryPlanner:
        mode = (self.config.memory_query_planner_mode or "").strip().lower()
        if mode == "llm":
            return LLMMemoryQueryPlanner(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.memory_query_llm_config,
                ),
            )
        return DisabledMemoryQueryPlanner()

    def _default_memory_reranker(self) -> MemoryReranker:
        mode = (self.config.memory_reranker_mode or "").strip().lower()
        if mode == "llm":
            return LLMMemoryReranker(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.memory_rerank_llm_config,
                ),
                selected_limit=self.config.memory_selected_limit,
                candidate_limit=self.config.memory_rerank_candidate_limit,
            )
        return DisabledMemoryReranker(selected_limit=self.config.memory_selected_limit)

    def _default_code_context_query_planner(self) -> CodeContextQueryPlanner:
        mode = (self.config.code_context_query_planner_mode or "").strip().lower()
        if mode == "llm":
            return LLMCodeContextQueryPlanner(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.code_context_query_llm_config,
                ),
            )
        return DisabledCodeContextQueryPlanner()

    def _default_code_context_reranker(self) -> CodeContextReranker:
        mode = (self.config.code_context_reranker_mode or "").strip().lower()
        if mode == "llm":
            return LLMCodeContextReranker(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.code_context_rerank_llm_config,
                ),
                selected_limit=self.config.code_context_selected_limit,
                candidate_limit=self.config.code_context_rerank_candidate_limit,
            )
        return DisabledCodeContextReranker(selected_limit=self.config.code_context_selected_limit)

    def _default_skill_selector(self) -> SkillSelector:
        mode = (self.config.skill_selector_mode or "").strip().lower()
        if mode == "llm":
            return LLMSkillSelector(
                llm_config=_resolve_llm_config(
                    self.config.llm_config,
                    self.config.skill_selector_llm_config,
                ),
                selected_limit=self.config.skill_selected_limit,
            )
        return DisabledSkillSelector()


def _resolve_llm_config(base: LLMConfig, override: LLMConfig) -> LLMConfig:
    default = LLMConfig()

    def resolve_str(field: str) -> str:
        value = getattr(override, field)
        default_value = getattr(default, field)
        return value if value and value != default_value else getattr(base, field)

    return LLMConfig(
        provider=resolve_str("provider"),
        model=resolve_str("model"),
        api_base=resolve_str("api_base"),
        api_key_env=resolve_str("api_key_env"),
        timeout=override.timeout if override.timeout != default.timeout else base.timeout,
        temperature=(
            override.temperature
            if override.temperature != default.temperature
            else base.temperature
        ),
        max_output_chars=(
            override.max_output_chars
            if override.max_output_chars != default.max_output_chars
            else base.max_output_chars
        ),
    )


def _llm_config_enabled(value: LLMConfig) -> bool:
    provider = str(value.provider).strip().lower()
    return provider not in {"", "disabled", "none"} and bool(str(value.model).strip())


def _verification_required(state: AgentState) -> bool:
    return bool(state.get("verification_required", True))


def _code_context_candidate_count(context: Dict[str, Any]) -> int:
    return sum(
        len(items)
        for key in CONTEXT_LIST_KEYS
        for items in [context.get(key, [])]
        if isinstance(items, list)
    )


def _selected_skill_context(
    selected_skills: list[str],
    skills: Any,
    max_resource_chars: int = 4000,
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for skill_name in selected_skills:
        spec = skills.get(skill_name) if hasattr(skills, "get") else None
        if spec is None:
            continue
        resources = []
        for resource_path in getattr(spec, "resources", [])[:2]:
            path = Path(str(resource_path))
            if not path.is_file():
                resources.append({"path": str(resource_path), "error": "resource not found"})
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                resources.append({"path": path.as_posix(), "error": str(exc)})
                continue
            resources.append(
                {
                    "path": path.as_posix(),
                    "content": text[:max_resource_chars],
                    "truncated": len(text) > max_resource_chars,
                }
            )
        context.append(
            {
                "source": "registry",
                "skill_name": skill_name,
                "description": getattr(spec, "description", ""),
                "resources": resources,
            }
        )
    return context


def _merge_skill_context(*groups: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            key = str(
                item.get("skill_name")
                or item.get("memory_id")
                or item.get("id")
                or item
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _merge_unique(*groups: Any) -> list[str]:
    values: list[str] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            value = str(item).strip()
            if value and value not in values:
                values.append(value)
    return values


def _build_project_profile(repo_path: str, max_files: int = 3000) -> dict[str, Any]:
    """
        启动时先统计项目使用语言分布情况
    """
    repo = Path(repo_path or ".").resolve()
    config = CompressionConfig()
    language_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    language_extensions: dict[str, Counter[str]] = defaultdict(Counter)
    sample_files: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    if not repo.exists() or not repo.is_dir():
        return {
            "repo_path": repo.as_posix(),
            "primary_language": "",
            "languages": [],
            "file_count": 0,
            "truncated": False,
            "error": "repo_path is not a directory",
        }
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(repo).parts
        if set(rel_parts).intersection(config.IGNORED_DIRS):
            continue
        suffix = path.suffix.lower()
        language = config.INDEXED_EXTENSIONS.get(suffix)
        if not language:
            continue
        rel = path.relative_to(repo).as_posix()
        language_counts[language] += 1
        extension_counts[suffix] += 1
        language_extensions[language][suffix] += 1
        if len(sample_files[language]) < 5:
            sample_files[language].append(rel)
        scanned += 1
        if scanned >= max_files:
            break
    total = sum(language_counts.values())
    languages = []
    for language, count in language_counts.most_common():
        languages.append(
            {
                "language": language,
                "file_count": count,
                "ratio": round(count / total, 4) if total else 0.0,
                "extensions": dict(language_extensions[language].most_common()),
                "sample_files": sample_files[language],
            }
        )
    primary_language = languages[0]["language"] if languages else ""
    return {
        "repo_path": repo.as_posix(),
        "primary_language": primary_language,
        "languages": languages[:8],
        "file_count": total,
        "extension_counts": dict(extension_counts.most_common(12)),
        "truncated": scanned >= max_files,
    }


def _clean_string_list(value: Any, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:max_chars])
        if len(cleaned) >= limit:
            break
    return cleaned


def _render_user_clarification(input_item: dict[str, Any]) -> str:
    answer = str(input_item.get("answer") or "").strip()
    if not answer:
        return ""
    questions = _clean_string_list(input_item.get("questions"), limit=5, max_chars=500)
    lines = ["User clarification:"]
    for question in questions:
        lines.append(f"- Question: {question}")
    lines.append(f"- Answer: {answer}")
    return "\n".join(lines)
