from types import MappingProxyType

from agent_runtime.memory.cards import MemoryCard
from agent_runtime.memory.manager import LayeredMemoryManager
from agent_runtime.memory.store import JsonlMemoryStore, LocalVectorMemoryStore
from agent_runtime.registry import RegistrySnapshot
from agent_runtime.search_query import SearchQueryPlanner


def _empty_registry() -> RegistrySnapshot:
    return RegistrySnapshot(
        tools=MappingProxyType({}),
        nodes=MappingProxyType({}),
        prompts=MappingProxyType({}),
        skills=MappingProxyType({}),
    )


def test_jsonl_store_crud_feedback_and_bad_line_tolerance(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")
    store = JsonlMemoryStore(path)
    card = MemoryCard(
        type="episodic",
        scope="repo",
        trigger="order status",
        content="Payment callback updates order status.",
        tags=["order", "payment"],
    )

    stored = store.append_card(card)

    assert store.get_card(stored.memory_id) is not None
    assert len(store.list_cards()) == 1
    assert store.search_cards("payment order")

    updated = store.record_reuse_feedback(stored.memory_id, success=False)
    assert updated is not None
    assert updated.reuse_failure == 1
    assert updated.conflict_score == 0.1

    deprecated = store.deprecate_card(stored.memory_id, reason="stale")
    assert deprecated is not None
    assert deprecated.status == "deprecated"
    assert store.search_cards("payment order") == []


def test_record_task_memory_promotes_success_and_updates_reuse_feedback(tmp_path):
    mid_store = JsonlMemoryStore(tmp_path / "mid.jsonl")
    long_store = LocalVectorMemoryStore(tmp_path / "long.jsonl")
    manager = LayeredMemoryManager(
        mid_store=mid_store,
        long_store=long_store,
        skill_dir=tmp_path / "skills",
    )
    old = mid_store.append_card(
        MemoryCard(
            type="episodic",
            scope="repo",
            trigger="order status",
            content="Order payment callback should update status.",
            tags=["order", "payment"],
        )
    )
    state = {
        "task_id": "task-1",
        "repo_path": "repo",
        "title": "order status not updated",
        "description": "payment callback",
        "candidate_files": ["orders/service.py"],
        "tool_calls": [{"name": "run_tests"}],
        "test_results": [{"command": "pytest", "exit_code": 0}],
        "patch": "diff",
        "patch_summary": "changed order status update",
        "retrieved_memories": {
            "mid_term": [{**old.to_dict(), "score": 1.0, "source": "mid_term"}],
            "long_term": [],
        },
        "selected_skills": [],
        "code_context": {"files": [{}], "functions": [], "api_routes": [], "db_models": []},
    }

    result = manager.record_task_memory(state, _empty_registry())

    assert result.written[0].type == "episodic"
    assert result.promoted
    assert any(card.type == "semantic" for card in result.promoted)
    assert result.feedback and result.feedback[0]["reuse_success"] == 1


def test_search_query_ignores_anti_pattern_memory():
    planner = SearchQueryPlanner(default_query="TODO")
    state = {
        "title": "order payment status",
        "description": "",
        "retrieved_memories": {
            "long_term": [
                {
                    "type": "anti_pattern",
                    "trigger": "bad path",
                    "content": "Do not search legacy_billing_wrong_path",
                    "tags": ["legacy_billing_wrong_path"],
                },
                {
                    "type": "semantic",
                    "trigger": "good path",
                    "content": "Payment callback logic is in order_service",
                    "tags": ["order_service"],
                },
            ]
        },
        "selected_skills": [],
        "skill_context": [],
        "code_context": {},
    }

    plan = planner.plan(state)

    assert "legacy_billing_wrong_path" not in plan.query
    assert "order_service" in plan.query or "order" in plan.query
