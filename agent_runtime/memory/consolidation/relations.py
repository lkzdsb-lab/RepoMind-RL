"""LLM-backed semantic relation classification between durable memories."""

from __future__ import annotations

import json

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.memory.consolidation.models import (
    MatchedMemory,
    RelationAssessment,
    RelationResolution,
)
from agent_runtime.memory.domain.models import MemoryDocument
from config import LLMConfig
from model.llm import MemoryRelationResponse
from prompts.templates import load_prompt, render_prompt


class LLMMemoryRelationResolver:
    """ 负责让 LLM 判断：新提炼的记忆与已有相似记忆，究竟是什么关系。"""
    def __init__(self, llm_config: LLMConfig) -> None:
        self.node = LLMJsonNode(
            name="memory_relation",
            llm_config=llm_config,
            system_prompt=load_prompt("system/memory_relation.md"),
            build_prompt=_relation_prompt,
            fallback=None,
            response_model=MemoryRelationResponse,
            raise_on_error=True,
        )

    def resolve(
        self,
        proposed: MemoryDocument,
        matches: list[MatchedMemory],
    ) -> RelationResolution:
        state = {
            "task_id": proposed.source.task_id,
            "llm_calls": [],
            "llm_errors": [],
            "llm_token_usage": {},
        }
        allowed = {item.document.memory_id for item in matches}
        try:
            response = self.node.run(
                state,
                {"proposed": proposed, "matches": matches},
                publish_update=False,
            )
            assessments: list[RelationAssessment] = []
            seen: set[str] = set()
            for raw in response.get("assessments", []):
                if not isinstance(raw, dict):
                    continue
                target = str(raw.get("target_memory_id") or "").strip()
                if target not in allowed or target in seen:
                    continue
                seen.add(target)
                assessments.append(_assessment(raw))
            if seen != allowed:
                missing = ", ".join(sorted(allowed - seen))
                raise ValueError(
                    f"relation response did not assess every semantic candidate: {missing}"
                )
            return RelationResolution(
                assessments=tuple(assessments),
                source="llm",
                target_documents={item.document.memory_id: item.document for item in matches},
                metadata={
                    "llm_calls": state.get("llm_calls", []),
                    "llm_token_usage": state.get("llm_token_usage", {}),
                },
            )
        except Exception as exc:
            return RelationResolution(
                source="llm_error",
                error=str(exc)[:2000],
                target_documents={item.document.memory_id: item.document for item in matches},
                metadata={"llm_errors": state.get("llm_errors", [])},
            )


def _assessment(raw: dict) -> RelationAssessment:
    relation = str(raw.get("relation") or "unrelated").strip().lower()
    if relation not in {"duplicate", "refine", "conflict", "unrelated"}:
        relation = "unrelated"
    return RelationAssessment(
        target_memory_id=str(raw.get("target_memory_id") or "").strip(),
        relation=relation,
        confidence=max(0.0, min(1.0, float(raw.get("confidence") or 0.0))),
        reason=str(raw.get("reason") or "").strip()[:2000],
        supporting_evidence_ids=tuple(_strings(raw.get("supporting_evidence_ids"), 30, 200)),
        merged_title=" ".join(str(raw.get("merged_title") or "").split())[:300],
        merged_knowledge=str(raw.get("merged_knowledge") or "").strip()[:50000],
        merged_applicability=str(raw.get("merged_applicability") or "").strip()[:10000],
        merged_invalidation=str(raw.get("merged_invalidation") or "").strip()[:10000],
    )


def _relation_prompt(state: dict, context: dict) -> str:
    proposed = context.get("proposed")
    matches = context.get("matches")
    if not isinstance(proposed, MemoryDocument) or not isinstance(matches, list):
        raise TypeError("memory relation resolution requires proposed memory and matches")
    payload = {
        "proposed_memory": proposed.to_dict(),
        "existing_memories": [
            {
                "similarity": item.score,
                "memory": item.document.to_dict(),
            }
            for item in matches
            if isinstance(item, MatchedMemory)
        ],
    }
    return render_prompt(
        "user/memory_relation.md",
        relation_packet=json.dumps(payload, ensure_ascii=False, default=str)[:60000],
    )


def _strings(value: object, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            str(item or "").strip()[:item_limit]
            for item in value[:limit]
            if str(item or "").strip()
        )
    )
