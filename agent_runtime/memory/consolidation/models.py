"""Application contracts for the offline consolidation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_runtime.memory.domain.models import MemoryDocument


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    summary: str
    reference: str
    strength: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source_event_ids: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskEvidenceBundle:
    task_id: str
    repo_id: str
    archive_path: str
    archive_hash: str
    repo_revision: str
    status: str
    objective: str
    outcome: str
    files: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    runtime_candidates: tuple[dict[str, Any], ...]
    session_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: str
    origin_kind: str
    content: str
    evidence_refs: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceLink:
    evidence_id: str
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceResolution:
    candidate_id: str
    links: tuple[EvidenceLink, ...]
    unresolved_references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "links": [item.to_dict() for item in self.links],
            "unresolved_references": list(self.unresolved_references),
        }


@dataclass(frozen=True)
class MemoryCandidate:
    candidate_id: str
    origin_kind: str
    content: str
    evidence_ids: tuple[str, ...]
    evidence_links: tuple[EvidenceLink, ...] = ()
    unresolved_references: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedMemory:
    candidate_id: str
    should_store: bool
    memory_type: str
    title: str
    knowledge: str
    applicability: str
    evidence_ids: tuple[str, ...]
    triggers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    scope_level: str = "repo"
    module: str = ""
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    invalidation: str = ""
    constraint_scope: str = ""
    constraint_durability: str = ""
    constraint_explicit: bool = False
    semantic_confidence: float = 0.0
    decision_reason: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionBatch:
    source: str
    memories: tuple[ExtractedMemory, ...]
    raw_response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "memories": [item.to_dict() for item in self.memories],
            "raw_response": dict(self.raw_response),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MatchedMemory:
    document: MemoryDocument
    score: float
    source: str
    model_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.document.memory_id,
            "score": self.score,
            "source": self.source,
            "knowledge_hash": self.document.knowledge_hash,
            "revision": self.document.revision,
            "model_fingerprint": self.model_fingerprint,
        }


@dataclass(frozen=True)
class RelationAssessment:
    target_memory_id: str
    relation: str
    confidence: float
    reason: str
    supporting_evidence_ids: tuple[str, ...] = ()
    merged_title: str = ""
    merged_knowledge: str = ""
    merged_applicability: str = ""
    merged_invalidation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationResolution:
    assessments: tuple[RelationAssessment, ...] = ()
    source: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    target_documents: dict[str, MemoryDocument] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "error": self.error,
            "assessments": [item.to_dict() for item in self.assessments],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryMutation:
    action: str
    source_candidate_id: str
    target_memory_id: str
    before_revision: int = 0
    after_revision: int = 0
    before_knowledge_hash: str = ""
    after_knowledge_hash: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionResult:
    document: MemoryDocument
    mutation: MemoryMutation
