"""Semantic extraction adapters for memory candidates."""

from __future__ import annotations

import json
from typing import Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.memory.consolidation.models import (
    ExtractedMemory,
    ExtractionBatch,
    MemoryCandidate,
    TaskEvidenceBundle,
)
from config import LLMConfig
from model.llm import MemoryExtractionResponse
from prompts.templates import load_prompt, render_prompt


class MemoryExtractor(Protocol):
    def extract(
        self,
        bundle: TaskEvidenceBundle,
        candidates: list[MemoryCandidate],
    ) -> ExtractionBatch: ...


class RuleBasedMemoryExtractor:
    def extract(
        self,
        bundle: TaskEvidenceBundle,
        candidates: list[MemoryCandidate],
    ) -> ExtractionBatch:
        memories = tuple(_rule_memory(bundle, candidate) for candidate in candidates)
        return ExtractionBatch(
            source="rule_based",
            memories=memories,
            raw_response={"memories": [item.to_dict() for item in memories]},
        )


class LLMMemoryExtractor:
    def __init__(
        self,
        llm_config: LLMConfig,
        *,
        fallback: MemoryExtractor | None = None,
    ) -> None:
        self.fallback = fallback or RuleBasedMemoryExtractor()
        self.node = LLMJsonNode(
            name="memory_extractor",
            llm_config=llm_config,
            system_prompt=load_prompt("system/memory_extractor.md"),
            build_prompt=_memory_extraction_prompt,
            fallback=None,
            response_model=MemoryExtractionResponse,
            raise_on_error=True,
        )

    def extract(
        self,
        bundle: TaskEvidenceBundle,
        candidates: list[MemoryCandidate],
    ) -> ExtractionBatch:
        state = {
            "task_id": bundle.task_id,
            "llm_calls": [],
            "llm_errors": [],
            "llm_token_usage": {},
        }
        context = {
            "bundle": bundle.to_dict(),
            "candidates": [item.to_dict() for item in candidates],
        }
        try:
            response = self.node.run(state, context, publish_update=False)
            memories = tuple(
                _memory_from_dict(item)
                for item in response.get("memories", [])
                if isinstance(item, dict)
            )
            return ExtractionBatch(
                source="llm",
                memories=memories,
                raw_response=response,
                metadata={
                    "llm_calls": state.get("llm_calls", []),
                    "llm_token_usage": state.get("llm_token_usage", {}),
                },
            )
        except Exception as exc:
            fallback = self.fallback.extract(bundle, candidates)
            return ExtractionBatch(
                source="rule_based_fallback",
                memories=fallback.memories,
                raw_response={"error": str(exc)[:2000]},
                metadata={"llm_errors": state.get("llm_errors", [])},
            )


def _rule_memory(
    bundle: TaskEvidenceBundle,
    candidate: MemoryCandidate,
) -> ExtractedMemory:
    memory_type = candidate.suggested_type
    title = _title(candidate.content, memory_type)
    applicability = {
        "episodic": "Use when investigating a task with the same objective or affected files.",
        "semantic": "Use when reasoning about the referenced repository scope.",
        "procedural": "Use when executing a similar repository task; revalidate each step.",
        "anti_pattern": "Use to avoid repeating this failed approach under similar conditions.",
        "preference": "Apply when working with the same user and the preference remains current.",
    }.get(memory_type, "Use only when the archived evidence and current scope match.")
    scope_level = "user" if memory_type == "preference" else "repo"
    return ExtractedMemory(
        candidate_id=candidate.candidate_id,
        should_store=bool(candidate.content and candidate.evidence_ids),
        memory_type=memory_type,
        title=title,
        knowledge=candidate.content,
        applicability=applicability,
        evidence_ids=candidate.evidence_ids,
        triggers=tuple(_trigger_terms(bundle.objective, title)),
        tags=(memory_type, "consolidated"),
        scope_level=scope_level,
        files=candidate.files if scope_level == "repo" else (),
        symbols=candidate.symbols,
        invalidation="Revalidate when linked evidence, files, or repository behavior changes.",
    )


def _memory_from_dict(data: dict) -> ExtractedMemory:
    return ExtractedMemory(
        candidate_id=str(data.get("candidate_id") or "").strip(),
        should_store=bool(data.get("should_store", True)),
        memory_type=str(data.get("memory_type") or "").strip(),
        title=" ".join(str(data.get("title") or "").split())[:300],
        knowledge=str(data.get("knowledge") or "").strip()[:50000],
        applicability=str(data.get("applicability") or "").strip()[:10000],
        evidence_ids=tuple(_strings(data.get("evidence_ids"), 30, 200)),
        triggers=tuple(_strings(data.get("triggers"), 30, 300)),
        tags=tuple(_strings(data.get("tags"), 30, 100)),
        scope_level=str(data.get("scope_level") or "repo").strip(),
        module=str(data.get("module") or "").strip()[:500],
        files=tuple(_strings(data.get("files"), 50, 500)),
        symbols=tuple(_strings(data.get("symbols"), 50, 500)),
        invalidation=str(data.get("invalidation") or "").strip()[:10000],
    )


def _memory_extraction_prompt(state: dict, context: dict) -> str:
    bundle = dict(context.get("bundle") or {})
    candidates = list(context.get("candidates") or [])
    bundle["session_context"] = _bounded_json(bundle.get("session_context"), 5000)
    return render_prompt(
        "user/memory_extractor.md",
        evidence_bundle=_bounded_json(bundle, 18000),
        candidates=_bounded_json(candidates, 24000),
    )


def _bounded_json(value: object, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _title(content: str, memory_type: str) -> str:
    first = " ".join(str(content or "").split())
    if len(first) > 140:
        first = first[:137] + "..."
    return first or f"{memory_type.replace('_', ' ').title()} memory"


def _trigger_terms(objective: str, title: str) -> list[str]:
    values = []
    for source in (objective, title):
        text = " ".join(str(source or "").split())[:240]
        if text and text not in values:
            values.append(text)
    return values


def _strings(value: object, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:max_chars])
        if len(result) >= limit:
            break
    return result
