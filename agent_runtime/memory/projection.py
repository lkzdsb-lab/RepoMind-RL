"""One bounded, valid JSON view for all online consumers of session memory."""
from __future__ import annotations

import json
from typing import Any

from agent_runtime.memory.working import stable_id
from agent_runtime.memory.work_policy import should_project
from model.session_work import ProjectionAudience, SessionProjection, WorkItemPreview, WorkKind, parse_work_view


def project_session(state: dict[str, Any], audience: str, max_chars: int | None = None) -> dict[str, Any]:
    """ 统一投影，每个不同的 node 用不同的方式处理短期记忆"""
    memory = state.get("session_memory") or {}
    audience = ProjectionAudience(audience).value
    budget = max(1000, int(max_chars or state.get("session_memory_budget", 12000)))
    # 1. 取得完整的工作记录
    items = state.get("session_work_items", memory.get("work_items", []))
    # 2. 根据调用节点筛选记录
    items = [parse_work_view(item).to_flat_dict() for item in items]
    items = [item for item in items if should_project(item, audience, state.get("task_id", ""))]
    items.sort(key=lambda item: (item["kind"] != WorkKind.REQUIREMENT, item["item_id"]))

    # 如果使 completion node 则过滤掉从 memory 得到的旧结论，关注本轮新产生的
    if audience == ProjectionAudience.COMPLETION:
        reads = [call for call in state.get("tool_calls", []) if call.get("name") == "read_session_memory"]
        selected = {item["item_id"] for item in (reads[-1].get("output") or {}).get("items", [])} if reads else set()
        items.sort(key=lambda item: item["item_id"] not in selected)

    result = {"schema_version": 1, "audience": audience,
              "revision": stable_id("view_", [(i["item_id"], i["version"], i.get("evidence_validity")) for i in items]),
              "topic_id": state.get("topic_id") or (memory.get("topic") or {}).get("topic_id", ""),
              "objective": state.get("effective_objective") or (memory.get("topic") or {}).get("title", ""),
              "total_items": len(items), "items": [], "omitted_items": len(items),
              "detail_tool": "read_session_memory", "history": [], "history_omitted": 0, "topics": [],
              "preview_only": True, "unassigned_messages": [],
              "unassigned_count": len(memory.get("unassigned_messages", []))}
    def fits() -> bool:
        return len(json.dumps(result, ensure_ascii=False)) <= budget
    # Required metadata can itself exceed the budget: retain a bounded preview.
    result["objective"] = str(result["objective"])[:max(100, budget // 8)]

    # 对得到的 item 进行文字长度的限制，超出限制的 item 放进 omitted 字段
    for item in items:
        entry = {key: item.get(key) for key in ("item_id", "version", "kind", "status", "description",
                                                "source_task_id", "source_message_id", "historical", "evidence_validity")}
        entry["description"] = str(entry["description"] or "")[:360]
        entry["detail_omitted"] = True
        # 如果是 用户原话 则添加上引用范围
        if item["kind"] == WorkKind.REQUIREMENT:
            entry.update(quote=str(item.get("quote", ""))[:360], edit_scope=item.get("edit_scope", "none"))
        if item.get("candidate_id"):
            entry["candidate_id"] = item["candidate_id"]
        result["items"].append(WorkItemPreview.model_validate(entry).model_dump(mode="json", exclude_unset=True))
        result["omitted_items"] -= 1
        if not fits():
            result["items"].pop()
            result["omitted_items"] += 1
            break

    # 如果是 任务分析 node，这里都是历史的 mem，则需要关注未绑定的信息以及历史的 topic
    if audience == ProjectionAudience.ANALYZER:
        for message in memory.get("unassigned_messages", []):
            result["unassigned_messages"].append({**message, "content": message["content"][:500], "preview_only": True})
            if not fits():
                result["unassigned_messages"].pop()
                break
        for topic in memory.get("topics", []):
            result["topics"].append(topic)
            if not fits():
                result["topics"].pop()
                break
    turns = memory.get("recent_turns", [])
    for turn in reversed(turns):
        outcome = turn.get("assistant_result") or {}
        entry = {"turn_index": turn.get("turn_index"), "user_message": str(turn.get("user_message") or "")[:700],
                 "summary": str(outcome.get("summary") or "")[:400], "status": turn.get("status")}
        result["history"].append(entry)
        if not fits():
            result["history"].pop()
            break
    result["history_omitted"] = len(turns) - len(result["history"])
    # Reserve room for final metadata without cutting serialized JSON.
    while not fits() and result["history"]:
        result["history"].pop()
        result["history_omitted"] += 1
    while not fits() and result["items"]:
        result["items"].pop()
        result["omitted_items"] += 1
    while not fits() and result["topics"]:
        result["topics"].pop()
    while not fits() and result["unassigned_messages"]:
        result["unassigned_messages"].pop()
    return SessionProjection.model_validate(result).model_dump(mode="json", exclude_unset=True)


def session_context(state: dict[str, Any], audience: str, max_chars: int | None = None) -> str:
    return json.dumps(project_session(state, audience, max_chars), ensure_ascii=False)


def candidate_context(state: dict[str, Any], max_chars: int = 5000) -> str:
    """ 对 findings 做内存预算限制"""
    candidates = state.get("draft_findings", [])
    result = {"total": len(candidates), "items": [], "omitted": len(candidates), "detail_tool": "read_session_memory"}
    for item in candidates:
        entry = {key: item.get(key) for key in ("candidate_id", "claim", "locations", "related_tests")}
        result["items"].append(entry)
        result["omitted"] -= 1
        if len(json.dumps(result, ensure_ascii=False)) > max_chars:
            result["items"].pop()
            result["omitted"] += 1
            break
    return json.dumps(result, ensure_ascii=False)
