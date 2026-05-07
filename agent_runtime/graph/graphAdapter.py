"""
    file name: graphAdapter.py
    Author: kunze.li
    description: The whole graph
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT
from langgraph_sdk.schema import _TypedDictLikeV1
from pydantic import BaseModel

from model.agent.graph import AgentState
from typing import TypedDict, Dict, Any
from graph.node import (
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
from graph.router import route_after_observe, route_after_reflect


def build_graph(ctx: ContextT, agent_state: AgentState):
    builder = StateGraph(state_schema=agent_state, context_schema=ctx)

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

    builder.add_edge(START, "understand_task")
    builder.add_edge("understand_task", "retrieve_context")
    builder.add_edge("retrieve_context", "make_plan")
    builder.add_edge("make_plan", "select_action")
    builder.add_edge("select_action", "execute_action")
    builder.add_edge("execute_action", "observe")

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

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph