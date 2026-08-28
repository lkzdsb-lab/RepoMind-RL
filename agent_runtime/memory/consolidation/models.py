"""Application contracts for the offline consolidation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    summary: str
    reference: str
    strength: float
    metadata: dict[str, Any] = field(default_factory=dict)

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
class MemoryCandidate:
    candidate_id: str
    suggested_type: str
    content: str
    evidence_ids: tuple[str, ...]
    source: str
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

