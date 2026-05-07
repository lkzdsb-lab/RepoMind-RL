from model.agent.graph import AgentState


def route_after_observe(state: AgentState) -> str:
    if state.get("error"):
        return "finalize"

    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 6)

    test_results = state.get("test_results", [])
    if test_results and test_results[-1].get("exit_code") == 0:
        return "finalize"

    if loop_count >= max_loops:
        return "finalize"

    # 第一版：跑过至少 3 轮后进入 patch 节点
    if loop_count >= 3 and not state.get("patch_summary"):
        return "generate_patch"

    return "select_action"


def route_after_reflect(state: AgentState) -> str:
    if state.get("status") == "finished":
        return "finalize"

    if state.get("loop_count", 0) >= state.get("max_loops", 6):
        return "finalize"

    return "select_action"