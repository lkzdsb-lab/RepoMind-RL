"""Semantic extraction adapters for memory candidates."""

from __future__ import annotations

import json
from typing import Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.memory.consolidation.constraints import classify_explicit_constraint
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
            "bundle": bundle,
            "candidates": candidates,
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
    memory_type = {
        "task_outcome": "episodic",
        "code_observation": "semantic",
        "successful_workflow": "procedural",
        "verification_failure": "anti_pattern",
        "error_pattern": "anti_pattern",
    }.get(candidate.origin_kind, "")
    constraint_scope = ""
    constraint_durability = ""
    constraint_explicit = False
    semantic_confidence = 1.0 if memory_type else 0.0
    if candidate.origin_kind == "user_statement":
        decision = classify_explicit_constraint(candidate.content)
        memory_type = "preference"
        constraint_scope = decision.scope
        constraint_durability = decision.durability
        constraint_explicit = decision.explicit
        semantic_confidence = decision.confidence
    title = _title(candidate.content, memory_type)
    applicability = {
        "episodic": "Use when investigating a task with the same objective or affected files.",
        "semantic": "Use when reasoning about the referenced repository scope.",
        "procedural": "Use when executing a similar repository task; revalidate each step.",
        "anti_pattern": "Use to avoid repeating this failed approach under similar conditions.",
        "preference": "Apply when working with the same user and the preference remains current.",
    }.get(memory_type, "Use only when the archived evidence and current scope match.")
    scope_level = (
        "user"
        if constraint_scope == "user"
        else "repo"
    )
    should_store = bool(
        candidate.content
        and candidate.evidence_ids
        and memory_type
        and (
            memory_type != "preference"
            or constraint_durability == "durable"
            and constraint_scope in {"repository", "user"}
            and constraint_explicit
        )
    )
    rejection_reason = ""
    if not candidate.evidence_ids:
        rejection_reason = "insufficient_evidence"
    elif not memory_type:
        rejection_reason = "unsupported_claim"
    elif memory_type == "preference" and constraint_durability == "temporary":
        rejection_reason = "temporary_constraint"
    elif memory_type == "preference" and constraint_scope not in {"repository", "user"}:
        rejection_reason = "scope_ambiguous"
    return ExtractedMemory(
        candidate_id=candidate.candidate_id,
        should_store=should_store,
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
        constraint_scope=constraint_scope,
        constraint_durability=constraint_durability,
        constraint_explicit=constraint_explicit,
        semantic_confidence=semantic_confidence,
        decision_reason=(
            "Conservative rule classification accepted the provenance-backed candidate."
            if should_store
            else "Conservative rule classification did not find a durable supported memory."
        ),
        rejection_reason=rejection_reason,
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
        constraint_scope=str(data.get("constraint_scope") or "").strip(),
        constraint_durability=str(data.get("constraint_durability") or "").strip(),
        constraint_explicit=bool(data.get("constraint_explicit", False)),
        semantic_confidence=max(
            0.0,
            min(1.0, float(data.get("semantic_confidence") or 0.0)),
        ),
        decision_reason=str(data.get("decision_reason") or "").strip()[:1000],
        rejection_reason=str(data.get("rejection_reason") or "").strip()[:100],
    )


def _memory_extraction_prompt(state: dict, context: dict) -> str:
    bundle = context.get("bundle")
    candidates = context.get("candidates")
    if not isinstance(bundle, TaskEvidenceBundle):
        raise TypeError("memory extraction requires a TaskEvidenceBundle")
    if not isinstance(candidates, list):
        raise TypeError("memory extraction requires resolved candidates")
    task_context = {
        "task_id": bundle.task_id,
        "repo_id": bundle.repo_id,
        "repo_revision": bundle.repo_revision,
        "archive_status": bundle.status,
        "objective": bundle.objective,
        "outcome": bundle.outcome,
        "affected_files": list(bundle.files[:50]),
    }
    return render_prompt(
        "user/memory_extractor.md",
        task_context=_bounded_json(task_context, 6000),
        candidate_packets=_bounded_json(
            _candidate_packets(bundle, candidates),
            48000,
        ),
    )


def _candidate_packets(
    bundle: TaskEvidenceBundle,
    candidates: list[MemoryCandidate],
) -> list[dict]:
    evidence_by_id = {item.evidence_id: item for item in bundle.evidence}
    packet_budget = max(1400, 42000 // max(1, len(candidates)))
    packets: list[dict] = []
    for candidate in candidates:
        relation_by_id = {
            item.evidence_id: item.relation for item in candidate.evidence_links
        }
        allowed_evidence = []
        evidence_ids = candidate.evidence_ids[:8]
        summary_limit = max(180, min(1200, packet_budget // max(2, len(evidence_ids) + 1)))
        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            allowed_evidence.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "kind": evidence.kind,
                    "summary": evidence.summary[:summary_limit],
                    "reference": evidence.reference,
                    "relation": relation_by_id.get(evidence_id, "resolved"),
                    "strength": evidence.strength,
                    "files": list(evidence.files[:20]),
                    "symbols": list(evidence.symbols[:20]),
                    "command": evidence.command[:1000],
                    "metadata": _small_metadata(evidence.metadata),
                }
            )
        candidate_payload = candidate.to_dict()
        candidate_payload["content"] = candidate.content[: max(500, packet_budget // 3)]
        packets.append(
            {
                "candidate": candidate_payload,
                "allowed_evidence": allowed_evidence,
            }
        )
    return packets


def _small_metadata(value: dict) -> dict:
    result = {}
    for key, item in list(value.items())[:12]:
        if isinstance(item, str):
            result[str(key)] = item[:600]
        elif isinstance(item, (int, float, bool)) or item is None:
            result[str(key)] = item
        elif isinstance(item, (list, tuple)):
            result[str(key)] = [str(entry)[:300] for entry in item[:12]]
    return result


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
