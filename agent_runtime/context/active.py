"""Stable incremental event identities and deterministic digest size control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from agent_runtime.context.cards import ContextDigest, ContextItem
from agent_runtime.context.token_counter import estimate_tokens


def active_events(events, distilled, consumed):
    """ 给事件计算稳定指纹"""
    unique = {}
    for event, item in zip(events, distilled):
        payload = event.payload
        if event.event_type == "file_event":
            identity = ["file", payload.get("file_path"), payload.get("file_revision"),
                        payload.get("start_line"), payload.get("end_line"), payload.get("content_excerpt")]
        elif event.event_type == "verification_event":
            identity = ["verification", payload.get("source_edit_revision"), payload.get("command"), payload.get("exit_code"),
                        payload.get("stdout_excerpt"), payload.get("stderr_excerpt")]
        else:
            identity = [item.event_type, item.source, item.facts, item.risks, item.open_questions]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key not in consumed:
            unique[key] = replace(item, event_id=key)
    selected = list(unique.values())
    items = [ContextItem(
        role="tool", item_type="distilled_event", item_id=item.event_id,
        content=json.dumps({"type": item.event_type, "source": item.source, "facts": item.facts,
                            "risks": item.risks, "open_questions": item.open_questions}, ensure_ascii=False),
    ) for item in selected]
    return selected, items


def incremental_fallback(items, state) -> ContextDigest:
    previous = state.get("context_digest") or {}
    digest = ContextDigest.from_dict(previous) if previous else ContextDigest(
        summary="", current_goal=str(state.get("title") or ""),
    )
    values = digest.to_dict()
    observations = list(digest.key_observations)
    for item in items:
        if item.content not in observations:
            observations.append(item.content)
    values.update(key_observations=observations, source_item_ids=[item.item_id for item in items],
                  compression_method="rule_based")
    return ContextDigest.from_dict(values)


def bound_digest(digest: ContextDigest, budget: int) -> ContextDigest:
    values = digest.to_dict()
    budget = max(0, budget)
    def size():
        return estimate_tokens(ContextDigest.from_dict(values).render_for_prompt())
    for field in ("completed_tasks", "memory_refs", "tool_results", "key_observations",
                  "code_changes", "decisions", "open_tasks", "constraints"):
        values[field] = list(values[field])
        while values[field] and size() > budget:
            values[field].pop(0)
    for field in ("summary", "current_goal"):
        while values[field] and size() > budget:
            values[field] = values[field][:len(values[field]) // 2]
    return ContextDigest.from_dict(values)
