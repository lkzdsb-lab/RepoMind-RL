"""
RepoMind-RL first-version agent executor.

The executor owns orchestration only. Planning/action selection, tool execution,
memory persistence, and trajectory recording are separate collaborators so each
layer can evolve independently.
"""

from __future__ import annotations

import json
import subprocess
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
from agent_runtime.context.events import latest_tool_event, should_llm_observe_event
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
from agent_runtime.llm.observation import (
    DisabledObserver,
    LLMObserver,
    Observer,
    build_action_limit_observation,
)
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
from agent_runtime.skill_context import build_selected_skill_context
from agent_runtime.tool_registry import ToolRegistry
from agent_runtime.trajectory import TrajectoryRecorder
from agent_runtime.user_updates import UserUpdateSink, set_user_update_sink
from model.agent.graph import AgentState, AgentRunResult
from model.agent.actions import Action
from config import CompressionConfig, DebugAgentConfig, LLMConfig
from loguru import logger
from model.agent.tools import tool_spec_prompt_dict
from utils import _clean_string_list


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
        user_update_sink: UserUpdateSink | None = None,
    ) -> None:
        configure_from_agent_config(config)
        self.config = config
        self.user_update_sink = user_update_sink
        set_user_update_sink(user_update_sink)
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
        """
            回顾对话，更新参数
        """
        started_at = time.perf_counter()
        self._active_registry = self.registry_manager.snapshot()
        state = {
            **state,
            "llm_action_inputs_enabled": _llm_action_inputs_enabled(self.config),
            "is_git_repo": bool(state.get("is_git_repo", _is_git_repo(self.config.repo_path))),
            "tool_manifest": _registry_tool_manifest(self._active_registry),
            "plan_mode": bool(state.get("plan_mode", False)),
            "plan_mode_entered": bool(state.get("plan_mode_entered", False)),
            "plan_mode_approved": bool(state.get("plan_mode_approved", False)),
            "debug_technical_plan": state.get("debug_technical_plan", ""),
            "plan_verification_commands": state.get("plan_verification_commands", []),
            "plan_mode_evaluation": state.get("plan_mode_evaluation", ""),
            "plan_mode_events": state.get("plan_mode_events", []),
            "execution_queue": state.get("execution_queue", []),
            "pending_action_requirements": state.get("pending_action_requirements", {}),
            "user_updates": state.get("user_updates", []),
            "last_user_update": state.get("last_user_update"),
            "llm_calls": state.get("llm_calls", []),
            "llm_token_usage": state.get("llm_token_usage", _empty_llm_token_usage()),
            "llm_errors": state.get("llm_errors", []),
            "editing_enabled": self.config.editing_enabled,
            "editing_config": _editing_config_dict(self.config),
            "edit_results": state.get("edit_results", []),
            "edited_files": state.get("edited_files", []),
            "change_summaries": state.get("change_summaries", []),
            "last_change_summary": state.get("last_change_summary", {}),
            "command_results": state.get("command_results", []),
            "verification_commands": state.get("verification_commands", []),
            "verification_stale": bool(state.get("verification_stale", False)),
            "last_edit_at_loop": int(state.get("last_edit_at_loop", -1)),
            "last_verified_edit_loop": int(state.get("last_verified_edit_loop", -1)),
            "context_events": state.get("context_events", []),
            "distilled_events": state.get("distilled_events", []),
            "working_context": state.get("working_context", ""),
            "archive_context": state.get("archive_context", ""),
            "context_sections": state.get("context_sections", {}),
            "memory_candidates": state.get("memory_candidates", []),
            "compressed_context_item_ids": state.get("compressed_context_item_ids", []),
            "require_step_approval": self.config.require_step_approval,
            "pending_step_approval": state.get("pending_step_approval", {}),
            "step_approval_history": state.get("step_approval_history", []),
            "action_history": state.get("action_history", []),
            "action_limit_events": state.get("action_limit_events", []),
        }
        if user_answer and _has_pending_step_approval(state):
            state, approved_action = self._handle_step_approval_response(state, user_answer)
            if approved_action is not None:
                state, should_stop = self._run_approved_action_once(state, approved_action)
                if should_stop:
                    if (
                        state.get("status") != "awaiting_user_input"
                        and state.get("current_step") != "finished"
                    ):
                        state = self._finalize(state)
                    return self._finish_run(state, started_at)
            return self._run_action_loop(state, started_at)
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
        """
            运行 pipeline
        """
        run_logger = logger.bind(task_id=state.get("task_id"), repo_path=self.config.repo_path)
        while state.get("loop_count", 0) < state.get("max_loops", self.config.max_loops):
            state = self._prepare_context(state)
            action = self.policy.next_action(state)
            limit_events = self.rl_action_space.consume_last_limit_events()
            if limit_events:
                action = Action(
                    action.name,
                    action.args,
                    thought=action.thought,
                    metadata={**dict(action.metadata), "action_limit_events": limit_events},
                )
            state = self._record_action_selection(state, action)

            if self._requires_step_approval(state, action):
                state = self._await_step_approval(state, action)
                break

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
            if state.get("status") == "awaiting_user_input":
                break
            if action.name == "write_memory" and not state.get("error"):
                state = self._finalize(state)
                break
            if self._should_observe_latest_tool(state):
                state = self._observe(state)
                if state.get("status") == "failed":
                    break

        else:
            if _can_finalize_at_loop_limit(state):
                run_logger.warning(
                    "agent run reached max loops but state is finalizable max_loops={} tool_calls={}",
                    self.config.max_loops,
                    len(state.get("tool_calls", [])),
                )
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

        return self._finish_run(state, started_at)

    def _finish_run(self, state: AgentState, started_at: float) -> AgentRunResult:
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
            tool_manifest=_registry_tool_manifest(registry),
            repo_path=self.config.repo_path,
            is_git_repo=_is_git_repo(self.config.repo_path),
            project_profile=project_profile,
            branch="",
            verification_required=True,
            verification_reason="Task analysis has not run yet.",
            verify_command=self.config.verify_command,
            task_analysis={},
            plan=[],
            current_step="created",
            candidate_files=[],
            read_file_cache={},
            read_file_order=[],
            code_context={},
            selected_code_context={},
            code_context_query_plan={},
            code_context_rerank={},
            observations=[],
            llm_observations=[],
            llm_calls=[],
            llm_token_usage=_empty_llm_token_usage(),
            llm_errors=[],
            user_updates=[],
            last_user_update=None,
            tool_calls=[],
            test_results=[],
            command_results=[],
            verification_commands=[],
            verification_stale=False,
            last_edit_at_loop=-1,
            last_verified_edit_loop=-1,
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
            compressed_context_item_ids=[],
            context_events=[],
            distilled_events=[],
            working_context="",
            archive_context="",
            context_sections={},
            memory_candidates=[],
            short_term_memories=[],
            promoted_memories=[],
            consolidated_skills=[],
            memory_written=False,
            rl_enabled=self.rl_enabled,
            rl_transitions=[],
            rl_last_reward={},
            llm_guard_events=[],
            action_history=[],
            action_limit_events=[],
            llm_action_inputs_enabled=_llm_action_inputs_enabled(self.config),
            plan_mode=False,
            plan_mode_entered=False,
            plan_mode_approved=False,
            debug_technical_plan="",
            plan_verification_commands=[],
            plan_mode_evaluation="",
            plan_mode_events=[],
            execution_queue=[],
            editing_enabled=self.config.editing_enabled,
            editing_config=_editing_config_dict(self.config),
            edit_results=[],
            edited_files=[],
            change_summaries=[],
            last_change_summary={},
            require_step_approval=self.config.require_step_approval,
            pending_step_approval={},
            step_approval_history=[],
            patch=None,
            patch_summary=None,
            final_report={},
            next_action=None,
            next_action_input=None,
            pending_action_requirements={},
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
        pending_requirements = dict(state.get("pending_action_requirements") or {})
        deferred = action.metadata.get("deferred_action") if isinstance(action.metadata, dict) else None
        limit_events = []
        if isinstance(action.metadata, dict):
            raw_limit_events = action.metadata.get("action_limit_events")
            if isinstance(raw_limit_events, list):
                limit_events = [event for event in raw_limit_events if isinstance(event, dict)]
        if isinstance(deferred, dict):
            pending_requirements = deferred
        elif pending_requirements:
            pending_requirements = {}
        action_history = list(state.get("action_history", []) or [])
        action_history.append(self._action_history_entry(state, action))
        observations = list(state.get("observations", []) or [])
        llm_observations = list(state.get("llm_observations", []) or [])
        action_limit_events = list(state.get("action_limit_events", []) or [])
        if limit_events:
            limit_observation = build_action_limit_observation(action.name, limit_events)
            observations.append(limit_observation)
            llm_observations = _store_lru_observation(
                llm_observations,
                limit_observation,
                limit=max(1, int(getattr(self.config, "observer_store_limit", 12))),
            )
            action_limit_events.extend(limit_events)
        state = {
            **state,
            "next_action": action.name,
            "next_action_input": action.args,
            "pending_action_requirements": pending_requirements,
            "action_history": action_history,
            "action_limit_events": action_limit_events,
            "observations": observations,
            "llm_observations": llm_observations,
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

    def _action_history_entry(self, state: AgentState, action: Action) -> dict[str, Any]:
        return {
            "action": action.name,
            "signature": _action_signature(action, state),
            "loop_count": int(state.get("loop_count", 0)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _execute_action(self, state: AgentState, action: Action) -> AgentState:
        """
            执行 action
        """
        action_logger = logger.bind(task_id=state.get("task_id"), action=action.name)
        started_at = time.perf_counter()
        action_logger.info("action execution started")
        try:
            deferred = action.metadata.get("deferred_action") if isinstance(action.metadata, dict) else None
            if isinstance(deferred, dict):
                output = {
                    "skipped": True,
                    "needs_more_context": True,
                    "deferred_action": deferred.get("action"),
                    "missing_required_args": deferred.get("missing_required_args", []),
                    "message": deferred.get("message") or "Action deferred until required arguments are inferred from repository context.",
                }
            elif action.name == "run_tests" and not _verification_required(state):
                output = {
                    "command": action.args.get("command", self.config.verify_command),
                    "skipped": True,
                    "reason": "verification_required=false",
                }
            elif action.name == "write_memory":
                output = self._write_memory(state)
            elif action.name == "request_user_input":
                questions = _clean_string_list(
                    action.args.get("questions"), limit=3, max_chars=300
                )
                reason = str(action.args.get("reason") or "").strip()
                if questions:
                    output = {
                        "needs_user_input": True,
                        "reason": reason,
                        "questions": questions,
                    }
                else:
                    output = {
                        "needs_user_input": False,
                        "needs_more_context": True,
                        "skipped": True,
                        "reason": reason,
                        "questions": [],
                        "message": (
                            "request_user_input skipped because no concrete "
                            "questions were provided."
                        ),
                    }
            elif action.name == "search_code_context":
                output = self._search_code_context(state, action)
            elif self._is_blocked_by_plan_mode(state, action):
                output = {
                    "error": "Code-changing actions require an approved plan. Call EnterPlanMode, then ExitPlanMode after the plan is evaluated.",
                    "needs_more_context": True,
                    "plan_mode": state.get("plan_mode"),
                    "plan_mode_approved": state.get("plan_mode_approved"),
                }
            else:
                action_args = dict(action.args)
                if action.name == "build_codebase_context":
                    action_args.setdefault("index_path", self.config.code_context_index_path)
                if action.name == "apply_code_patch":
                    action_args["_guard"] = self._edit_guard(state)
                output = self._registry().run_tool(
                    action.name,
                    self.config.repo_path,
                    action_args,
                    allowed_permissions=self._allowed_tool_permissions(),
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
        state = self.recorder.append(
            state,
            node="execute_action",
            thought=f"执行动作：{action.name}",
            action=action.name,
            action_input=action.args,
            observation=output,
        )
        if output.get("needs_user_input"):
            state = self._await_user_input(state, output)
        return state

    def _requires_step_approval(self, state: AgentState, action: Action) -> bool:
        if not bool(self.config.require_step_approval):
            return False
        if not bool(state.get("require_step_approval", self.config.require_step_approval)):
            return False
        if _has_pending_step_approval(state):
            return False
        return action.name != "request_user_input"

    def _await_step_approval(self, state: AgentState, action: Action) -> AgentState:
        pending = {
            "action": action.name,
            "args": dict(action.args),
            "thought": action.thought,
            "metadata": dict(action.metadata),
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "loop_count": state.get("loop_count", 0),
        }
        question = (
            f"是否批准执行下一步 `{action.name}`？回复 `approve`/`yes`/`同意` 执行；"
            "输入其他内容会作为补充说明写回上下文并重新规划。"
        )
        reason = _render_step_approval_reason(action)
        state = {
            **state,
            "status": "awaiting_user_input",
            "current_step": "awaiting_step_approval",
            "pending_user_questions": [question],
            "needs_user_input_reason": reason,
            "pending_step_approval": pending,
        }
        logger.bind(task_id=state.get("task_id"), action=action.name).info(
            "agent awaiting step approval"
        )
        return self.recorder.append(
            state,
            node="await_step_approval",
            thought="等待用户批准下一步 action。",
            action=action.name,
            action_input=action.args,
            observation={
                "reason": reason,
                "questions": [question],
                "pending_step_approval": pending,
            },
        )

    def _handle_step_approval_response(
        self,
        state: AgentState,
        answer: str,
    ) -> tuple[AgentState, Action | None]:
        """
            如果 step_approvel_response == true
        """
        pending = state.get("pending_step_approval")
        if not isinstance(pending, dict) or not pending.get("action"):
            return self._inject_user_input(state, answer), None

        # 1. 获取用户反馈以及断电前的上下文
        approved = _approval_answer_is_approved(answer)
        state = _record_step_approval_response(state, answer, approved)
        if approved:
            action = _action_from_pending_step_approval(pending)
            state = {
                **state,
                "status": "running",
                "current_step": "step_approval_approved",
                "pending_user_questions": [],
                "needs_user_input_reason": "",
                "pending_step_approval": {},
                "error": None,
            }
            logger.bind(task_id=state.get("task_id"), action=action.name).info(
                "step approval granted"
            )
            return self.recorder.append(
                state,
                node="step_approval_approved",
                thought="用户批准执行下一步 action。",
                action=action.name,
                action_input=action.args,
                observation={"approved": True, "answer": str(answer or "").strip()},
            ), action

        # 2. 如果用户输入了附加的 feature，则注入到 state 中
        state = self._inject_user_input(state, answer)
        state = {
            **state,
            "pending_step_approval": {},
            "current_step": "step_approval_feedback_received",
        }
        logger.bind(task_id=state.get("task_id"), action=pending.get("action")).info(
            "step approval not granted; treating response as user feedback"
        )
        return self.recorder.append(
            state,
            node="step_approval_feedback",
            thought="用户未批准下一步，反馈已写回上下文。",
            action=str(pending.get("action") or ""),
            action_input=pending.get("args") if isinstance(pending.get("args"), dict) else {},
            observation={"approved": False, "answer": str(answer or "").strip()},
        ), None

    def _run_approved_action_once(
        self,
        state: AgentState,
        action: Action,
    ) -> tuple[AgentState, bool]:
        if action.name == "finish":
            prev_state = state
            state, should_stop, done, output = self._handle_finish_action(state)
            state = self._record_rl_transition(prev_state, action, state, output, done=done)
            return state, should_stop

        state = self._execute_action(state, action)
        if state.get("status") in {"failed", "awaiting_user_input"}:
            return state, True
        if action.name == "write_memory" and not state.get("error"):
            return self._finalize(state), True
        if self._should_observe_latest_tool(state):
            state = self._observe(state)
            if state.get("status") == "failed":
                return state, True
        return state, False

    def _edit_guard(self, state: AgentState) -> dict[str, Any]:
        """
            获取 llm calls 操作过的所有文件后的回复
        """
        read_contents: dict[str, str] = {}
        cache = state.get("read_file_cache") or {}
        if isinstance(cache, dict):
            for file_path, snapshot in cache.items():
                if not isinstance(snapshot, dict):
                    continue
                content = snapshot.get("content")
                if str(file_path).strip() and isinstance(content, str):
                    read_contents[str(file_path).strip()] = content
        if not read_contents:
            for call in state.get("tool_calls", []):
                if not isinstance(call, dict) or call.get("name") != "read_file":
                    continue
                output = call.get("output")
                if not isinstance(output, dict) or output.get("error"):
                    continue
                file_path = str(output.get("file_path") or "").strip()
                content = output.get("content")
                if file_path and isinstance(content, str):
                    read_contents[file_path] = content
        return {
            "editing_enabled": self.config.editing_enabled,
            "allowed_files": sorted(read_contents),
            "read_contents": read_contents,
            "max_files": self.config.editing_max_files,
            "max_changed_lines": self.config.editing_max_changed_lines,
            "max_file_bytes": self.config.editing_max_file_bytes,
            "require_read_before_write": self.config.editing_require_read_before_write,
            "confidence_threshold": self.config.editing_confidence_threshold,
            "allow_create": self.config.editing_allow_create,
        }

    def _allowed_tool_permissions(self) -> list[str]:
        """
            限制 llm action
        """
        permissions = {"repo:read", "repo:command", "agent:plan"}
        if self.config.editing_enabled:
            permissions.add("repo:write")
        return sorted(permissions)

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
                    allowed_permissions=self._allowed_tool_permissions(),
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
        """ 根据 tool 执行的结果更新 state"""
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
            if output.get("needs_more_context"):
                updates["status"] = "need_more_context"
            else:
                updates["error"] = output["error"]
            if output.get("fatal"):
                updates["status"] = "failed"
            logger.bind(task_id=state.get("task_id"), action=action.name).warning(
                "tool output contains error error={}",
                output["error"],
            )
        elif _tool_output_is_success(output):
            if state.get("error"):
                updates["error"] = None
            if state.get("status") == "need_more_context" and "status" not in updates:
                updates["status"] = "running"

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
        if observation.get("source") != "disabled" and bool(observation.get("store", True)):
            llm_observations = _store_lru_observation(
                llm_observations,
                observation,
                limit=max(1, int(getattr(self.config, "observer_store_limit", 12))),
            )
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
        if tool_name == "search_text":
            matches = output.get("matches", [])
            return f"{tool_name} pattern={output.get('pattern')} matches={len(matches)}."
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
        if tool_name == "run_shell_command":
            return (
                f"{tool_name} purpose={output.get('purpose')} "
                f"exit_code={output.get('exit_code')} command={output.get('command')}"
            )
        if tool_name == "apply_code_patch":
            return (
                f"{tool_name} applied={output.get('applied')} "
                f"files={output.get('changed_files', [])} "
                f"changed_lines={output.get('changed_line_count', 0)}"
            )
        if tool_name == "request_user_input":
            return f"{tool_name} questions={output.get('questions', [])}"
        if tool_name == "EnterPlanMode":
            return f"{tool_name} entered={output.get('entered')} plan_chars={len(str(output.get('technical_plan') or ''))}"
        if tool_name == "ExitPlanMode":
            return f"{tool_name} exited={output.get('exited')} approved={output.get('approved')}"
        if tool_name == "git_diff":
            if output.get("skipped"):
                return f"{tool_name} skipped reason={output.get('reason')}"
            diff = output.get("diff", "")
            return f"{tool_name} returned {len(diff.splitlines())} diff lines."
        return f"{tool_name} output keys: {', '.join(sorted(output.keys()))}"

    def _should_observe_latest_tool(self, state: AgentState) -> bool:
        mode = (self.config.observer_mode or "").strip().lower()
        if mode == "disabled":
            return False
        calls = state.get("tool_calls") or []
        if not calls:
            return False
        latest = calls[-1]
        if not isinstance(latest, dict):
            return False
        output = latest.get("output")
        if not isinstance(output, dict):
            return False
        tool_name = str(latest.get("name") or "")

        if output.get("needs_user_input"):
            return False
        event = latest_tool_event(state)
        if event is not None:
            return should_llm_observe_event(event)
        if output.get("fatal"):
            return True
        if output.get("error"):
            return not bool(output.get("unsupported") or output.get("skipped"))
        if output.get("needs_more_context"):
            return True
        if tool_name in {"run_shell_command", "run_tests"}:
            return output.get("exit_code") not in (None, 0)
        if tool_name == "search_code_context":
            return _code_context_candidate_count(output) == 0
        if tool_name == "search_text":
            return not output.get("matches")
        return False

    def _handle_finish_action(
        self,
        state: AgentState,
    ) -> tuple[AgentState, bool, bool, Dict[str, Any]]:
        """
            处理终止状态，由 llm 判断是否结束并总结，并且进行 loop 限制
        """
        if state.get("plan_mode"):
            judgement = {
                "decision": "continue",
                "reason": "Agent is still in Plan Mode and must call ExitPlanMode before finishing.",
                "questions": [],
                "suggested_next_action": "ExitPlanMode",
                "confidence": 1.0,
                "source": "rule_gate",
            }
            output = {"completion_judgement": judgement}
            state = self._record_completion_judgement(state, judgement)
            state = {
                **state,
                "current_step": "select_action",
                "completion_judge_continue_count": int(
                    state.get("completion_judge_continue_count", 0)
                )
                + 1,
                "loop_count": int(state.get("loop_count", 0)) + 1,
            }
            return state, False, False, output
        if _requires_post_edit_verification(state):
            judgement = {
                "decision": "continue",
                "reason": "Code changes have not been verified after the latest edit.",
                "questions": [],
                "suggested_next_action": "run_shell_command",
                "confidence": 1.0,
                "source": "rule_gate",
            }
            output = {"completion_judgement": judgement}
            state = self._record_completion_judgement(state, judgement)
            state = {
                **state,
                "current_step": "select_action",
                "completion_judge_continue_count": int(
                    state.get("completion_judge_continue_count", 0)
                )
                + 1,
                "loop_count": int(state.get("loop_count", 0)) + 1,
            }
            return state, False, False, output
        judgement = self.completion_judge.judge(state)
        output = {"completion_judgement": judgement}
        state = self._record_completion_judgement(state, judgement)
        decision = str(judgement.get("decision") or "complete").strip().lower()
        if decision == "needs_user_input":
            state = self._await_user_input(state, judgement)
            if state.get("status") != "awaiting_user_input":
                state = {
                    **state,
                    "current_step": "select_action",
                    "completion_judge_continue_count": int(
                        state.get("completion_judge_continue_count", 0)
                    )
                    + 1,
                }
                return state, False, False, output
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
            state = {
                **state,
                "status": "need_more_context",
                "current_step": "select_action",
                "pending_user_questions": [],
                "needs_user_input_reason": reason,
            }
            logger.bind(task_id=state.get("task_id")).warning(
                "skipped awaiting user input because no concrete questions were provided reason={}",
                reason,
            )
            return self.recorder.append(
                state,
                node="await_user_input_skipped",
                thought="没有明确问题，跳过用户询问并继续收集上下文。",
                observation={
                    "reason": reason,
                    "questions": [],
                    "skipped": True,
                },
            )
        if _is_duplicate_user_question_set(state, questions):
            state = {
                **state,
                "status": "need_more_context",
                "current_step": "select_action",
                "pending_user_questions": [],
                "needs_user_input_reason": reason,
            }
            logger.bind(task_id=state.get("task_id")).info(
                "skipped duplicate user question set questions={} reason={}",
                questions,
                reason,
            )
            return self.recorder.append(
                state,
                node="await_user_input_skipped_duplicate",
                thought="重复问题集已被抑制，继续尝试其他动作。",
                observation={
                    "reason": reason,
                    "questions": questions,
                    "skipped": True,
                    "duplicate": True,
                },
            )
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
        """
            将用户附加 feature 注入对话
        """
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
        # 这一步是在去除因为用户行为影响整个的 max_loop
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
        """
            对话结束流程
        """
        status = "finished" if not state.get("error") else "failed"
        state = {**state, "status": status, "current_step": "finished"}
        if status == "finished":
            state = self._write_memory_on_finalize(state)
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
        final_report = _attach_runtime_usage_to_report(final_report, state)
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
                "llm_token_usage": state.get("llm_token_usage", {}),
                "llm_errors": state.get("llm_errors", [])[-5:],
            },
        )

    def _write_memory_on_finalize(self, state: AgentState) -> AgentState:
        if state.get("memory_written") or not _has_meaningful_task_result(state):
            return state
        try:
            output = self._write_memory(state)
        except Exception as exc:
            logger.bind(task_id=state.get("task_id")).warning(
                "finalize memory write failed error={}",
                exc,
            )
            observations = state.get("observations", []) + [
                {
                    "type": "memory_write_failed",
                    "tool": "write_memory",
                    "content": {"error": str(exc)},
                }
            ]
            return {**state, "observations": observations}

        tool_calls = state.get("tool_calls", []) + [
            {
                "name": "write_memory",
                "input": {"trigger": "finalize"},
                "output": output,
                "error": output.get("error") if isinstance(output, dict) else None,
            }
        ]
        observations = state.get("observations", []) + [
            {
                "type": "tool_output",
                "tool": "write_memory",
                "content": output,
            }
        ]
        return {
            **state,
            "memory_written": True,
            "promoted_memories": output.get("promoted", []) if isinstance(output, dict) else [],
            "consolidated_skills": output.get("consolidated", []) if isinstance(output, dict) else [],
            "tool_calls": tool_calls,
            "observations": observations,
        }

    def _registry(self) -> RegistrySnapshot:
        if self._active_registry is None:
            self._active_registry = self.registry_manager.snapshot()
        return self._active_registry

    def _is_blocked_by_plan_mode(self, state: AgentState, action: Action) -> bool:
        if action.name != "apply_code_patch":
            return False
        return bool(state.get("plan_mode")) or not bool(state.get("plan_mode_approved"))

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
                use_delta=bool(getattr(self.config, "observer_use_delta", True)),
                full_state_on_severe=bool(getattr(self.config, "observer_full_state_on_severe", True)),
                write_threshold=float(getattr(self.config, "observer_write_threshold", 0.35)),
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


def _requires_post_edit_verification(state: AgentState) -> bool:
    if state.get("error"):
        return False
    if not state.get("edited_files"):
        return False
    return bool(state.get("verification_stale", False))


def _has_meaningful_task_result(state: AgentState) -> bool:
    """
        判断是否有必要写 memory
    """
    return bool(
        state.get("edited_files")
        or state.get("patch_summary") is not None
        or state.get("test_results")
        or state.get("verification_commands")
    )


def _can_finalize_at_loop_limit(state: AgentState) -> bool:
    if state.get("error") or state.get("plan_mode"):
        return False
    if _requires_post_edit_verification(state):
        return False
    return _has_meaningful_task_result(state)


def _is_git_repo(repo_path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_path or ".",
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _code_context_candidate_count(context: Dict[str, Any]) -> int:
    return sum(
        len(items)
        for key in CONTEXT_LIST_KEYS
        for items in [context.get(key, [])]
        if isinstance(items, list)
    )


def _tool_output_is_success(output: Dict[str, Any]) -> bool:
    if output.get("error") or output.get("fatal"):
        return False
    if output.get("needs_more_context") or output.get("needs_user_input"):
        return False
    if output.get("skipped"):
        return False
    if "ok" in output:
        return bool(output.get("ok"))
    exit_code = output.get("exit_code")
    if isinstance(exit_code, int):
        return exit_code == 0
    status = str(output.get("status") or "").strip().lower()
    if status:
        return status in {"success", "complete", "finished"}
    return True


def _empty_llm_token_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "by_node": {},
    }


def _attach_runtime_usage_to_report(
    final_report: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    """
        上报 token usage 等信息
    """
    report = dict(final_report or {})
    usage = state.get("llm_token_usage")
    if isinstance(usage, dict):
        report["llm_token_usage"] = usage
    errors = state.get("llm_errors")
    if isinstance(errors, list) and errors:
        report["llm_errors"] = errors[-5:]
    return report


def _has_pending_step_approval(state: AgentState) -> bool:
    pending = state.get("pending_step_approval")
    return isinstance(pending, dict) and bool(pending.get("action"))


def _approval_answer_is_approved(answer: str) -> bool:
    """
        阶段性
        获取 user 的指令
    """
    text = str(answer or "").strip().lower()
    if not text:
        return False
    text = text.strip(" .!！。")
    approvals = {
        "approve",
        "approved",
        "yes",
        "y",
        "ok",
        "okay",
        "go",
        "go ahead",
        "continue",
        "run",
        "execute",
        "同意",
        "批准",
        "确认",
        "可以",
        "继续",
        "执行",
        "好的",
        "好",
        "是",
        "允许",
    }
    if text in approvals:
        return True
    prefixes = (
        "approve",
        "approved",
        "yes",
        "ok",
        "go ahead",
        "continue",
        "run it",
        "execute",
        "同意",
        "批准",
        "确认",
        "可以",
        "继续",
        "执行",
    )
    return text.startswith(prefixes)


def _action_from_pending_step_approval(pending: dict[str, Any]) -> Action:
    args = pending.get("args")
    metadata = pending.get("metadata")
    return Action(
        name=str(pending.get("action") or ""),
        args=args if isinstance(args, dict) else {},
        thought=str(pending.get("thought") or ""),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _record_step_approval_response(
    state: AgentState,
    answer: str,
    approved: bool,
) -> AgentState:
    pending = state.get("pending_step_approval")
    if not isinstance(pending, dict):
        pending = {}
    item = {
        "pending_step_approval": pending,
        "answer": str(answer or "").strip(),
        "approved": approved,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    history = state.get("step_approval_history")
    if not isinstance(history, list):
        history = []
    observations = state.get("observations", []) + [
        {
            "type": "step_approval_response",
            "content": item,
        }
    ]
    return {
        **state,
        "step_approval_history": history + [item],
        "observations": observations,
    }


def _render_step_approval_reason(action: Action) -> str:
    args = _compact_action_args(action.args)
    if args:
        return f"下一步准备执行 `{action.name}`，参数：{args}"
    return f"下一步准备执行 `{action.name}`。"


def _compact_action_args(args: dict[str, Any], max_chars: int = 800) -> str:
    if not args:
        return ""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except TypeError:
        text = str(args)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _registry_tool_manifest(registry: Any) -> list[dict[str, Any]]:
    if registry is None:
        return []
    names = registry.names("tools") if hasattr(registry, "names") else []
    manifest: list[dict[str, Any]] = []
    for name in names:
        spec = registry.get_tool(name) if hasattr(registry, "get_tool") else None
        if spec is None:
            continue
        manifest.append(tool_spec_prompt_dict(spec))
    return manifest


def _selected_skill_context(
    selected_skills: list[str],
    skills: Any,
) -> list[dict[str, Any]]:
    return build_selected_skill_context(selected_skills, skills)


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


def _normalize_question_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _question_set_signature(questions: list[str]) -> str:
    return "|".join(_normalize_question_text(item) for item in questions if _normalize_question_text(item))


def _is_duplicate_user_question_set(state: AgentState, questions: list[str]) -> bool:
    signature = _question_set_signature(questions)
    if not signature:
        return False
    pending = _question_set_signature(
        _clean_string_list(state.get("pending_user_questions"), limit=5, max_chars=500)
    )
    if pending and pending == signature:
        return True
    for item in state.get("user_inputs", []) or []:
        if not isinstance(item, dict):
            continue
        previous = _question_set_signature(
            _clean_string_list(item.get("questions"), limit=5, max_chars=500)
        )
        if previous and previous == signature:
            return True
    return False


def _action_signature(action: Action, state: AgentState | None = None) -> str:
    if action.name == "read_file":
        file_path = str(action.args.get("file_path") or "").strip()
        if not file_path and isinstance(state, dict):
            current = _current_execution_item(state)
            if isinstance(current, dict):
                for path in current.get("target_files", []) or []:
                    file_path = str(path).strip()
                    if file_path:
                        break
        return f"read_file:{file_path or '<unknown>'}"
    if action.name == "apply_code_patch":
        changes = action.args.get("changes")
        targets: list[str] = []
        if isinstance(changes, list):
            for change in changes:
                if not isinstance(change, dict):
                    continue
                file_path = str(change.get("file_path") or "").strip()
                if file_path and file_path not in targets:
                    targets.append(file_path)
        if not targets:
            targets = [
                str(path).strip()
                for path in action.args.get("target_files", []) or []
                if str(path).strip()
            ]
        if not targets and isinstance(state, dict):
            current = _current_execution_item(state)
            if isinstance(current, dict):
                targets = [
                    str(path).strip()
                    for path in current.get("target_files", []) or []
                    if str(path).strip()
                ]
        return f"apply_code_patch:{'|'.join(sorted(targets)) or '<unknown>'}"
    if action.name == "run_shell_command":
        command = str(action.args.get("command") or "").strip()
        return f"run_shell_command:{command or '<unknown>'}"
    if action.name == "search_code_context":
        query = str(action.args.get("query") or "").strip()
        return f"search_code_context:{query or '<empty>'}"
    if action.name == "request_user_input":
        questions = _clean_string_list(action.args.get("questions"), limit=3, max_chars=300)
        signature = _question_set_signature(questions)
        if signature:
            return f"request_user_input:{signature}"
        missing = ",".join(
            sorted(
                str(item).strip()
                for item in action.metadata.get("missing_required_args", []) or []
                if str(item).strip()
            )
        )
        return f"request_user_input:{missing or 'generic'}"
    return f"action:{action.name}"


def _current_execution_item(state: AgentState) -> dict[str, Any] | None:
    for item in state.get("execution_queue", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "pending") == "pending":
            return item
    return None


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


def _editing_config_dict(config: DebugAgentConfig) -> dict[str, Any]:
    return {
        "enabled": config.editing_enabled,
        "max_files": config.editing_max_files,
        "max_changed_lines": config.editing_max_changed_lines,
        "max_file_bytes": config.editing_max_file_bytes,
        "require_read_before_write": config.editing_require_read_before_write,
        "confidence_threshold": config.editing_confidence_threshold,
        "allow_create": config.editing_allow_create,
    }


def _llm_action_inputs_enabled(config: DebugAgentConfig) -> bool:
    return (config.action_policy_mode or "").strip().lower() == "llm"


def _store_lru_observation(
    existing: list[dict[str, Any]],
    observation: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    items = [dict(item) for item in existing if isinstance(item, dict)]
    signature = _observation_lru_signature(observation)
    items = [item for item in items if _observation_lru_signature(item) != signature]
    items.append(dict(observation))
    return items[-limit:]


def _observation_lru_signature(observation: dict[str, Any]) -> str:
    return "|".join(
        [
            str(observation.get("event_id") or ""),
            str(observation.get("latest_tool") or ""),
            str(observation.get("status") or ""),
            str(observation.get("summary") or ""),
        ]
    )


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
