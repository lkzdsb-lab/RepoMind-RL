"""
    File name: graph.py
    Author: kunze.li
"""
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
    # 引入 skill
    task_category: str
    selected_skills: list[str]
    skill_context: list[dict]

    repo_path: str # 仓库路径
    branch: str # 仓库分支
    verify_command: str # 权限认证相关命令

    plan: List[str] # llm 给的 plan
    current_step: str

    candidate_files: List[str] # llm 想要调用的 files
    observations: List[Dict[str, Any]] # 执行流程记录
    tool_calls: List[ToolCall] # llm 使用过的 tools
    test_results: List[TestResult]
    trajectory: List[TrajectoryStep] # 整个任务的结果集

    patch: Optional[str] # 修改文件的块
    patch_summary: Optional[str]

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
        "finished",
        "failed",
    ]

    error: Optional[str]