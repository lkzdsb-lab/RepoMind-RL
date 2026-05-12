"""
    file name: graphAdapter.py
    Author: kunze.li
    description: The whole graph
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent_runtime.registry import RegistryManager, RegistrySnapshot
from model.agent.graph import AgentState
from agent_runtime.graph.node import (
    understand_task_node,
    retrieve_context_node,
    make_plan_node,
    select_action_node,
    execute_action_node,
    observe_node,
    generate_patch_node,
    run_tests_node,
    reflect_node,
    finalize_node,
)
from agent_runtime.graph.router import route_after_observe, route_after_reflect

# 后续这些
def build_graph(registry_snapshot: RegistrySnapshot | None = None):
    registry_snapshot = registry_snapshot or RegistryManager().snapshot()
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("understand_task", understand_task_node)
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("make_plan", make_plan_node)
    builder.add_node("select_action", select_action_node)
    builder.add_node("execute_action", execute_action_node)
    builder.add_node("observe", observe_node)
    builder.add_node("generate_patch", generate_patch_node)
    builder.add_node("run_tests", run_tests_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("finalize", finalize_node)
    _add_registered_nodes(builder, registry_snapshot)
    # 注册边
    builder.add_edge(START, "understand_task")
    builder.add_edge("understand_task", "retrieve_context")
    builder.add_edge("retrieve_context", "make_plan")
    builder.add_edge("make_plan", "select_action")
    builder.add_edge("select_action", "execute_action")
    builder.add_edge("execute_action", "observe")

    # 增加从一个 node 到任何其他 node 的连接
    builder.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "select_action": "select_action",
            "generate_patch": "generate_patch",
            "finalize": "finalize",
        },
    )

    builder.add_edge("generate_patch", "run_tests")
    builder.add_edge("run_tests", "reflect")

    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "select_action": "select_action",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)
    # checkpointer 提供上下问切换的能力，允许从任何一个节点重放
    memory_saver = InMemorySaver()
    graph = builder.compile(checkpointer=memory_saver)

    return graph


def _add_registered_nodes(
    builder: StateGraph,
    registry_snapshot: RegistrySnapshot,
) -> None:
    builtins = {
        "understand_task",
        "retrieve_context",
        "make_plan",
        "select_action",
        "execute_action",
        "observe",
        "generate_patch",
        "run_tests",
        "reflect",
        "finalize",
    }

    for node in registry_snapshot.nodes.values():
        if node.name not in builtins:
            builder.add_node(node.name, node.handler)

        for edge in node.metadata.get("edges", []):
            source = edge.get("from")
            target = edge.get("to")
            if source and target:
                builder.add_edge(_graph_endpoint(source), _graph_endpoint(target))


def _graph_endpoint(name: str):
    if name == "START":
        return START
    if name == "END":
        return END
    return name
