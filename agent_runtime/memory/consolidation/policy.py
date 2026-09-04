"""Deterministic evidence binding, promotion, identity, and exact deduplication."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from agent_runtime.memory.consolidation.models import (
    EvidenceItem,
    ExtractedMemory,
    MemoryCandidate,
    TaskEvidenceBundle,
)
from agent_runtime.memory.consolidation.constraints import (
    has_repository_marker,
    has_temporary_marker,
)
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.models import (
    MemoryDocument,
    MemoryEvidence,
    MemoryScope,
    MemorySource,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from agent_runtime.memory.io import compact_json
from utils import utc_now


_MEMORY_TYPES = {item.value: item for item in MemoryType}
_SCOPE_LEVELS = {item.value: item for item in ScopeLevel}
_STATUS_RANK = {
    MemoryStatus.DRAFT: 0,
    MemoryStatus.NEEDS_REVIEW: 1,
    MemoryStatus.VERIFIED: 2,
}

_ORIGIN_MEMORY_TYPES = {
    "task_outcome": {MemoryType.EPISODIC},
    "user_statement": {MemoryType.PREFERENCE},
    "code_observation": {MemoryType.SEMANTIC},
    "successful_workflow": {MemoryType.PROCEDURAL},
    "verification_failure": {MemoryType.ANTI_PATTERN},
    "error_pattern": {MemoryType.ANTI_PATTERN},
    "observer_observation": {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
        MemoryType.ANTI_PATTERN,
    },
}


class MemoryDocumentBuilder:
    def build(
        self,
        bundle: TaskEvidenceBundle,
        candidates: list[MemoryCandidate],
        extracted: tuple[ExtractedMemory, ...],
        *,
        pipeline_version: str,
    ) -> tuple[list[MemoryDocument], int, list[str]]:
        candidate_map = {item.candidate_id: item for item in candidates}
        evidence_map = {item.evidence_id: item for item in bundle.evidence}
        documents: list[MemoryDocument] = []
        rejected = 0
        warnings: list[str] = []
        seen_candidates: set[str] = set()
        for item in extracted:
            candidate = candidate_map.get(item.candidate_id)
            if candidate is None or item.candidate_id in seen_candidates:
                rejected += 1
                warnings.append(f"ignored unknown or duplicate candidate {item.candidate_id!r}")
                continue
            seen_candidates.add(item.candidate_id)
            if not item.should_store:
                rejected += 1
                continue
            try:
                document = self._document(
                    bundle,
                    candidate,
                    item,
                    evidence_map,
                    pipeline_version,
                )
            except ValueError as exc:
                rejected += 1
                warnings.append(f"candidate {item.candidate_id}: {exc}")
                continue
            documents.append(document)
        rejected += len(candidate_map) - len(seen_candidates)
        return documents, rejected, warnings

    def _document(
        self,
        bundle: TaskEvidenceBundle,
        candidate: MemoryCandidate,
        item: ExtractedMemory,
        evidence_map: dict[str, EvidenceItem],
        pipeline_version: str,
    ) -> MemoryDocument:
        """ 将归档文件提取的证据、历史记忆、 llm 提取的关键信息沉淀为 md"""
        memory_type = _MEMORY_TYPES.get(item.memory_type)
        if memory_type is None:
            raise ValueError(f"unsupported memory_type {item.memory_type!r}")
        if candidate.unresolved_references:
            raise ValueError(
                "candidate has unresolved evidence references: "
                + ", ".join(candidate.unresolved_references)
            )
        allowed_types = _ORIGIN_MEMORY_TYPES.get(candidate.origin_kind, set())
        if memory_type not in allowed_types:
            raise ValueError(
                f"memory_type {memory_type.value!r} is not allowed for "
                f"origin_kind {candidate.origin_kind!r}"
            )
        allowed_evidence = set(candidate.evidence_ids)
        evidence = [
            evidence_map[evidence_id]
            for evidence_id in item.evidence_ids
            if evidence_id in evidence_map and evidence_id in allowed_evidence
        ]
        if not evidence:
            raise ValueError("no valid evidence IDs")
        if not item.title or not item.knowledge or not item.applicability:
            raise ValueError("title, knowledge, and applicability are required")
        if memory_type == MemoryType.PREFERENCE:
            _validate_preference_decision(candidate, item, evidence)
        scope = _validated_scope(bundle, candidate, item, memory_type)
        knowledge = (
            candidate.content
            if memory_type == MemoryType.PREFERENCE
            else item.knowledge
        )
        evidence_strength = round(sum(value.strength for value in evidence) / len(evidence), 4)
        status = _promotion_status(bundle, candidate, memory_type, evidence)
        confidence = _confidence(status, evidence_strength, len(evidence))
        memory_id = _memory_id(bundle.repo_id, memory_type, scope, knowledge)
        now = utc_now()
        return MemoryDocument(
            memory_id=memory_id,
            memory_type=memory_type,
            title=item.title,
            status=status,
            scope=scope,
            source=MemorySource(
                task_id=bundle.task_id,
                archive_path=bundle.archive_path,
                archive_hash=bundle.archive_hash,
                repo_revision=bundle.repo_revision,
                pipeline_version=pipeline_version,
            ),
            knowledge=knowledge,
            applicability=item.applicability,
            triggers=item.triggers,
            tags=tuple(sorted(set(item.tags + (memory_type.value,)))),
            evidence=tuple(
                _memory_evidence(
                    value,
                    _evidence_relation(candidate, value.evidence_id),
                )
                for value in evidence
            ),
            invalidation=item.invalidation,
            confidence=confidence,
            evidence_strength=evidence_strength,
            created_at=now,
            updated_at=now,
            extensions={
                "candidate_id": candidate.candidate_id,
                "candidate_origin": candidate.origin_kind,
                "source_tasks": [bundle.task_id],
            },
        )


def save_with_exact_dedup(
    store: MarkdownMemoryDocumentStore,
    document: MemoryDocument,
) -> MemoryDocument:
    existing = store.get(document.memory_id)
    if existing is None:
        store.save(document)
        return document
    if (
        existing.memory_type != document.memory_type
        or existing.scope != document.scope
        or existing.knowledge != document.knowledge
    ):
        raise ValueError(f"memory ID collision for {document.memory_id}")
    evidence = {item.evidence_id: item for item in existing.evidence}
    evidence.update({item.evidence_id: item for item in document.evidence})
    source_tasks = list(existing.extensions.get("source_tasks") or [])
    if document.source.task_id not in source_tasks:
        source_tasks.append(document.source.task_id)
    extensions = dict(existing.extensions)
    extensions["source_tasks"] = source_tasks
    status = existing.status if existing.status not in _STATUS_RANK else max(
        (existing.status, document.status), key=lambda value: _STATUS_RANK.get(value, -1)
    )
    merged = replace(
        existing,
        status=status,
        triggers=tuple(dict.fromkeys(existing.triggers + document.triggers)),
        tags=tuple(sorted(set(existing.tags + document.tags))),
        evidence=tuple(evidence.values()),
        confidence=max(existing.confidence, document.confidence),
        evidence_strength=max(existing.evidence_strength, document.evidence_strength),
        updated_at=utc_now(),
        extensions=extensions,
    )
    store.save(merged)
    return merged


def _validated_scope(
    bundle: TaskEvidenceBundle,
    candidate: MemoryCandidate,
    item: ExtractedMemory,
    memory_type: MemoryType,
) -> MemoryScope:
    level = _SCOPE_LEVELS.get(item.scope_level)
    if level is None:
        raise ValueError(f"unsupported scope level {item.scope_level!r}")
    if memory_type == MemoryType.PREFERENCE:
        level = (
            ScopeLevel.USER
            if item.constraint_scope == "user"
            else ScopeLevel.REPO
        )
    elif level in {ScopeLevel.GLOBAL, ScopeLevel.USER}:
        raise ValueError("non-preference memories cannot escape repository scope")
    allowed_files = set(candidate.files)
    requested_files = tuple(path for path in item.files if path in allowed_files)
    if level == ScopeLevel.FILE and not requested_files:
        raise ValueError("file scope has no evidence-backed files")
    allowed_symbols = set(candidate.symbols)
    requested_symbols = tuple(value for value in item.symbols if value in allowed_symbols)
    if level == ScopeLevel.SYMBOL and not requested_symbols:
        raise ValueError("symbol scope has no evidence-backed symbols")
    if level == ScopeLevel.MODULE and not item.module:
        raise ValueError("module scope requires module")
    return MemoryScope(
        level=level,
        repo_id="" if level == ScopeLevel.USER else bundle.repo_id,
        module=item.module,
        files=requested_files or tuple(candidate.files),
        symbols=requested_symbols,
    )


def _promotion_status(
    bundle: TaskEvidenceBundle,
    candidate: MemoryCandidate,
    memory_type: MemoryType,
    evidence: list[EvidenceItem],
) -> MemoryStatus:
    verification_passed = any(
        item.kind == "verification" and item.metadata.get("exit_code") == 0
        for item in evidence
    )
    verification_failed = any(
        item.kind == "verification"
        and item.metadata.get("exit_code") not in (None, 0)
        for item in evidence
    )
    has_source = any(item.kind in {"source", "diff"} for item in evidence)
    if memory_type == MemoryType.EPISODIC and bundle.status == "finished":
        return MemoryStatus.VERIFIED
    if memory_type == MemoryType.SEMANTIC and has_source and verification_passed:
        return MemoryStatus.VERIFIED
    if memory_type == MemoryType.ANTI_PATTERN and verification_failed:
        return MemoryStatus.VERIFIED
    if memory_type == MemoryType.PROCEDURAL and verification_passed and any(
        item.kind == "trajectory" for item in evidence
    ):
        return MemoryStatus.VERIFIED
    if memory_type == MemoryType.PREFERENCE:
        return MemoryStatus.VERIFIED
    return MemoryStatus.DRAFT


def _confidence(status: MemoryStatus, evidence_strength: float, count: int) -> float:
    status_bonus = 0.15 if status == MemoryStatus.VERIFIED else 0.0
    count_bonus = min(0.1, max(0, count - 1) * 0.02)
    return round(min(1.0, evidence_strength * 0.75 + status_bonus + count_bonus), 4)


def _memory_evidence(item: EvidenceItem, relation: str) -> MemoryEvidence:
    return MemoryEvidence(
        evidence_id=item.evidence_id,
        kind=item.kind,
        reference=item.reference,
        strength=item.strength,
        metadata={"summary": item.summary, "relation": relation, **item.metadata},
    )


def _validate_preference_decision(
    candidate: MemoryCandidate,
    item: ExtractedMemory,
    evidence: list[EvidenceItem],
) -> None:
    if has_temporary_marker(candidate.content):
        raise ValueError("temporary user constraint cannot become a preference")
    if (
        has_repository_marker(candidate.content)
        and item.constraint_scope != "repository"
    ):
        raise ValueError("repository preference cannot be widened to user scope")
    if item.constraint_durability != "durable":
        raise ValueError("preference must be explicitly durable")
    if item.constraint_scope not in {"repository", "user"}:
        raise ValueError("preference scope must be repository or user")
    if not item.constraint_explicit:
        raise ValueError("preference must be explicit")
    if float(item.semantic_confidence) < 0.75:
        raise ValueError("preference semantic confidence is below 0.75")
    if not any(value.kind == "user_statement" for value in evidence):
        raise ValueError("preference requires original user_statement evidence")


def _evidence_relation(candidate: MemoryCandidate, evidence_id: str) -> str:
    for link in candidate.evidence_links:
        if link.evidence_id == evidence_id:
            return link.relation
    return "resolved"


def _memory_id(
    repo_id: str,
    memory_type: MemoryType,
    scope: MemoryScope,
    knowledge: str,
) -> str:
    identity = compact_json(
        {
            "repo_id": repo_id if scope.level != ScopeLevel.USER else "user",
            "memory_type": memory_type.value,
            "scope": scope.to_dict(),
            "knowledge": " ".join(knowledge.lower().split()),
        }
    )
    return f"mem_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
