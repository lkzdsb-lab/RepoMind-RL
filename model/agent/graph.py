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
    # 引入 skill
    task_category: str
    verification_required: bool
    verification_reason: str
    task_analysis: Dict[str, Any]
    selected_skills: list[str]
    skill_selection: Dict[str, Any]
    skill_context: list[dict]

    repo_path: str # 仓库路径
    project_profile: Dict[str, Any] # 项目语言和文件概况
    branch: str # 仓库分支
    verify_command: str # 权限认证相关命令

    plan: List[str] # llm 给的 plan
    current_step: str

    candidate_files: List[str] # llm 想要调用的 files
    code_context: Dict[str, Any]
    selected_code_context: Dict[str, Any]
    code_context_query_plan: Dict[str, Any]
    code_context_rerank: Dict[str, Any]
    observations: List[Dict[str, Any]] # 执行流程记录
    llm_observations: List[Dict[str, Any]]
    tool_calls: List[ToolCall] # llm 使用过的 tools
    test_results: List[TestResult]
    trajectory: List[TrajectoryStep] # 整个任务的结果集

    # 需要用户补充信息时的暂停/恢复状态
    completion_judgement: Dict[str, Any]
    pending_user_questions: List[str]
    needs_user_input_reason: str
    completion_judge_continue_count: int
    user_inputs: List[Dict[str, Any]]

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

    # 记忆持久话相关
    short_term_memories: List[Dict[str, Any]]
    promoted_memories: List[Dict[str, Any]]
    consolidated_skills: List[Dict[str, Any]]
    memory_written: bool

    # rl 模块相关
    rl_enabled: bool
    rl_transitions: List[Dict[str, Any]]
    rl_last_reward: Dict[str, Any]
    llm_guard_events: List[Dict[str, Any]]

    patch: Optional[str] # 修改文件的块
    patch_summary: Optional[str]
    final_report: Dict[str, Any]

    next_action: Optional[str] # 下一个步骤
    next_action_input: Optional[Dict[str, Any]]

    loop_count: int
    max_loops: int

    status: Literal[
        "created",
        "running",
        "need_more_context",
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
