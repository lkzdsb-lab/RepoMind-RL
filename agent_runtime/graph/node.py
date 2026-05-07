"""
    file name: node.py
    Author: kunze.li
"""
from typing import Dict, Any

from model.agent.graph import AgentState, TrajectoryStep
from tools.code_tools.code import search_code
from tools.code_tools.file import list_files, read_file
from tools.go_tools.go_test import run_go_test
from tools.git_tools.diff import git_diff
from config import FileConfig

# 添加结果记录到当前对话 trace
def append_trajectory(
    state: AgentState,
    node: str,
    thought: str,
    action: str | None = None,
    action_input: Dict[str, Any] | None = None,
    observation: Dict[str, Any] | None = None,
) -> AgentState:
    trajectory = state.get("trajectory", [])
    step = {
        "step_id": len(trajectory) + 1,
        "node": node,
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": observation,
    }
    step = TrajectoryStep(**step)
    return {
        **state,
        "trajectory": trajectory + [step],
    }

# 理解任务节点，获取任务的类型以及状态并初始化
def understand_task_node(state: AgentState) -> AgentState:
    new_state = {
        **state,
        "status": "running",
        "loop_count": state.get("loop_count", 0),
        "max_loops": state.get("max_loops", 6),
        "observations": state.get("observations", []),
        "tool_calls": state.get("tool_calls", []),
        "test_results": state.get("test_results", []),
        "candidate_files": state.get("candidate_files", []),
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="understand_task",
        thought=f"理解任务：{state.get('title', '')}",
    )

# 读取文件，建立上下文
def retrieve_context_node(state: AgentState) -> AgentState:
    output = list_files(state["repo_path"], max_files=FileConfig.MAX_READ_AMOUNT)

    observations = state.get("observations", []) + [
        {
            "type": "repo_files",
            "content": output,
        }
    ]

    new_state = {
        **state,
        "observations": observations,
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="retrieve_context",
        thought="读取仓库文件结构，建立初始上下文。",
        action="list_files",
        action_input={"repo_path": state["repo_path"]},
        observation=output,
    )

# 创建 plan 节点
def make_plan_node(state: AgentState) -> AgentState:
    title = state.get("title", "")
    description = state.get("description", "")

    plan = [
        "根据任务描述提取关键词",
        "搜索相关 Go 代码",
        "阅读候选文件",
        "运行测试确认失败",
        "生成或建议补丁",
        "再次运行测试验证",
    ]

    new_state = {
        **state,
        "plan": plan,
        "current_step": "search_related_code",
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="make_plan",
        thought=f"为任务制定调试计划。任务标题：{title}。任务描述：{description}",
    )

# 选择 tools 节点
# todo 这里的 workflow 后期需要让 llm 自行判断使用哪个工具， 并且自行判断是否结束
def select_action_node(state: AgentState) -> AgentState:
    loop_count = state.get("loop_count", 0)

    if loop_count == 0:
        action = "search_code"
        action_input = {"query": extract_keyword(state)}
    elif loop_count == 1 and state.get("candidate_files"):
        action = "read_file"
        action_input = {"file_path": state["candidate_files"][0]}
    elif loop_count == 2:
        action = "run_go_test"
        action_input = {"command": state.get("verify_command", "go test ./...")}
    else:
        action = "git_diff"
        action_input = {}

    new_state = {
        **state,
        "next_action": action,
        "next_action_input": action_input,
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="select_action",
        thought=f"选择下一步动作：{action}",
        action=action,
        action_input=action_input,
    )

# 执行 tools 节点
# todo 这里后期也要优化
def execute_action_node(state: AgentState) -> AgentState:
    action = state["next_action"]
    action_input = state.get("next_action_input", {})
    repo_path = state["repo_path"]

    if action == "search_code":
        output = search_code(repo_path, action_input["query"])
        candidate_files = extract_files_from_grep(output.get("matches", []))
        new_state = {
            **state,
            "candidate_files": candidate_files,
        }

    elif action == "read_file":
        output = read_file(repo_path, action_input["file_path"])
        new_state = state

    elif action == "run_go_test":
        output = run_go_test(repo_path, action_input.get("command", "go test ./..."))
        test_results = state.get("test_results", []) + [output]
        new_state = {
            **state,
            "test_results": test_results,
        }

    elif action == "git_diff":
        output = git_diff(repo_path)
        new_state = state

    else:
        output = {"error": f"Unknown action: {action}"}
        new_state = {
            **state,
            "error": output["error"],
        }

    tool_calls = state.get("tool_calls", []) + [
        {
            "name": action,
            "input": action_input,
            "output": output,
            "error": output.get("error"),
        }
    ]

    observations = state.get("observations", []) + [
        {
            "type": "tool_output",
            "tool": action,
            "content": output,
        }
    ]

    new_state = {
        **new_state,
        "tool_calls": tool_calls,
        "observations": observations,
        "loop_count": state.get("loop_count", 0) + 1,
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="execute_action",
        thought=f"执行工具：{action}",
        action=action,
        action_input=action_input,
        observation=output,
    )

# 回顾信息节点
def observe_node(state: AgentState) -> AgentState:
    return append_trajectory(
        state,
        node="observe",
        thought="整理最近一次工具调用结果，准备判断下一步。",
    )

# 修改代码节点
def generate_patch_node(state: AgentState) -> AgentState:
    # 第一版先不真正改文件，只生成 patch 意图。
    # 后续这里接 LLM，让它根据上下文生成 unified diff。
    new_state = {
        **state,
        "status": "patching",
        "patch_summary": "第一版暂未自动改代码。后续在此节点接入 LLM 生成 patch。",
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="generate_patch",
        thought="根据已收集上下文生成候选补丁。",
    )

# 测试代码节点
def run_tests_node(state: AgentState) -> AgentState:
    command = state.get("verify_command", "go test ./...")
    output = run_go_test(state["repo_path"], command)

    test_results = state.get("test_results", []) + [output]

    new_state = {
        **state,
        "status": "testing",
        "test_results": test_results,
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="run_tests",
        thought="运行测试验证当前代码状态。",
        action="run_go_test",
        action_input={"command": command},
        observation=output,
    )

# 评价节点
def reflect_node(state: AgentState) -> AgentState:
    latest_test = state.get("test_results", [{}])[-1]
    passed = latest_test.get("exit_code") == 0

    thought = "测试通过，可以结束。" if passed else "测试未通过，需要继续分析。"

    new_state = {
        **state,
        "status": "finished" if passed else "need_more_context",
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="reflect",
        thought=thought,
    )

# 结束节点
def finalize_node(state: AgentState) -> AgentState:
    new_state = {
        **state,
        "status": "finished",
    }
    new_state = AgentState(**new_state)

    return append_trajectory(
        new_state,
        node="finalize",
        thought="任务执行结束，输出最终结果。",
    )

# 提取关键信息
def extract_keyword(state: AgentState) -> str:
    text = f"{state.get('title', '')} {state.get('description', '')}"

    keyword_candidates = [
        "DeleteComment",
        "Comment",
        "comment",
        "Post",
        "post",
        "Like",
        "User",
        "Auth",
    ]

    for keyword in keyword_candidates:
        if keyword.lower() in text.lower():
            return keyword

    return "TODO"

# 从 command line 提取文件信息
def extract_files_from_grep(matches: list[str]) -> list[str]:
    files = []

    for line in matches:
        # grep -RIn 输出类似：./internal/comment/service.go:12:func DeleteComment...
        if line.startswith("./"):
            path = line.split(":", 1)[0]
            path = path[2:]
            if path not in files:
                files.append(path)

    return files[:10]