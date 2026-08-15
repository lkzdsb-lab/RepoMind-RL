"""
    File name: graph.py
    Author: kunze.li
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any, Optional, Literal

# 工具调用格式
class ToolCall(TypedDict, total=False):
    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    error: Optional[str]

# 结果返回格式
class TestResult(TypedDict, total=False):
    command: str
    exit_code: int
    stdout: str
    stderr: str

# 每一个步骤的记录格式
class TrajectoryStep(TypedDict, total=False):
    step_id: int
    node: str
    thought: str
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[Dict[str, Any]]

# 记录 agent 活动状态格式
class AgentState(TypedDict, total=False):
    task_id: str  # 后续考虑是否添加 trace id 跟踪任务流程
    task_type: Literal["BUG_FIX", "FEATURE_IMPL", "DIAGNOSE"]
    title: str
    description: str
    registry_snapshot: Dict[str, List[str]]
    tool_manifest: List[Dict[str, Any]]
    # 引入 skill
    task_category: str
    verification_required: bool
    verification_reason: str
    task_analysis: Dict[str, Any]
    selected_skills: list[str]
    skill_selection: Dict[str, Any]
    skill_context: list[dict]

    repo_path: str # 仓库路径
    is_git_repo: bool
    project_profile: Dict[str, Any] # 项目语言和文件概况
    branch: str # 仓库分支
    verify_command: str # 权限认证相关命令

    plan: List[str] # llm 给的 plan
    current_step: str

    candidate_files: List[str] # llm 想要调用的 files
    # todo 考虑使用本地缓存实现
    read_file_cache: Dict[str, Dict[str, Any]]
    read_file_order: List[str]
    code_context: Dict[str, Any]
    selected_code_context: Dict[str, Any]
    code_context_query_plan: Dict[str, Any]
    code_context_rerank: Dict[str, Any]
    observations: List[Dict[str, Any]] # 执行流程记录
    llm_observations: List[Dict[str, Any]]
    llm_calls: List[Dict[str, Any]]
    llm_token_usage: Dict[str, Any]
    llm_errors: List[Dict[str, Any]]
    user_updates: List[Dict[str, Any]]
    last_user_update: Optional[Dict[str, Any]]
    tool_calls: List[ToolCall] # llm 使用过的 tools
    test_results: List[TestResult]
    command_results: List[Dict[str, Any]]
    verification_commands: List[Dict[str, Any]]
    verification_stale: bool
    last_edit_at_loop: int
    last_verified_edit_loop: int
    trajectory: List[TrajectoryStep] # 整个任务的结果集

    # 需要用户补充信息时的暂停/恢复状态
    completion_judgement: Dict[str, Any]
    pending_user_questions: List[str]
    needs_user_input_reason: str
    completion_judge_continue_count: int
    user_inputs: List[Dict[str, Any]]
    require_step_approval: bool
    pending_step_approval: Dict[str, Any]
    step_approval_history: List[Dict[str, Any]]

    # 查询过的记忆记录
    retrieved_memories: Dict[str, Any]
    selected_memories: Dict[str, Any]
    memory_query_plan: Dict[str, Any]
    memory_rerank: Dict[str, Any]
    memory_context: str

    # 上下文压缩
    context_items: List[Dict[str, Any]]
    context_digest: Dict[str, Any]
    compressed_context: str
    compressed_context_item_ids: List[str]
    context_events: List[Dict[str, Any]]
    distilled_events: List[Dict[str, Any]]
    working_context: str
    archive_context: str
    context_sections: Dict[str, List[str]]
    memory_candidates: List[Dict[str, Any]]
    attention_focus: Dict[str, Any]

    # 记忆持久话相关
    short_term_memories: List[Dict[str, Any]]
    promoted_memories: List[Dict[str, Any]]
    consolidated_skills: List[Dict[str, Any]]
    memory_written: bool
    """ 是否已经写过 memory 标记"""

    # rl 模块相关
    rl_enabled: bool
    rl_transitions: List[Dict[str, Any]]
    rl_last_reward: Dict[str, Any]
    llm_guard_events: List[Dict[str, Any]]
    action_history: List[Dict[str, Any]]
    action_limit_events: List[Dict[str, Any]]

    llm_action_inputs_enabled: bool
    plan_mode: bool
    plan_mode_entered: bool
    plan_mode_approved: bool
    debug_technical_plan: str
    plan_verification_commands: List[str]
    plan_mode_evaluation: str
    plan_mode_events: List[Dict[str, Any]]
    execution_queue: List[Dict[str, Any]]
    """ plan 后的执行队列"""
    editing_enabled: bool
    editing_config: Dict[str, Any]
    edit_results: List[Dict[str, Any]]
    edited_files: List[str]
    change_summaries: List[Dict[str, Any]]
    change_events: List[Dict[str, Any]]
    patch: Optional[str] # 修改文件的块
    patch_summary: Optional[str]
    final_report: Dict[str, Any]

    next_action: Optional[str] # 下一个步骤
    next_action_input: Optional[Dict[str, Any]]
    pending_resolution: Dict[str, Any]
    phase: str
    runtime_decision: Dict[str, Any]

    loop_count: int
    max_loops: int

    status: Literal[
        "created",
        "running",
        "need_more_context",
        "planning",
        "patching",
        "testing",
        "awaiting_user_input",
        "finished",
        "failed",
    ]

    error: Optional[str]


@dataclass
class AgentRunResult:
    state: AgentState
    trace_path: str
