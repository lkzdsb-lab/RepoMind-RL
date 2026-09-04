"""Read-only long-term-memory retrieval application models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlannedMemoryQuery:
    label: str
    text: str
    memory_types: tuple[str, ...] = ()
    scope_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    memory_type: str
    status: str
    title: str
    scope: dict[str, Any]
    knowledge: str
    applicability: str
    invalidation: str
    triggers: tuple[str, ...]
    tags: tuple[str, ...]
    confidence: float
    evidence_strength: float
    score: float
    reasons: tuple[str, ...]
    content_hash: str
    source_task_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRetrievalBatch:
    phase: str
    revision: str
    queries: tuple[PlannedMemoryQuery, ...]
    hits: tuple[dict[str, Any], ...]
    memories: tuple[RetrievedMemory, ...]
    context: str
    sections: dict[str, str] = field(default_factory=dict)
    section_memory_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "revision": self.revision,
            "queries": [item.to_dict() for item in self.queries],
            "hits": [dict(item) for item in self.hits],
            "memories": [item.to_dict() for item in self.memories],
            "context": self.context,
            "sections": dict(self.sections),
            "section_memory_ids": {
                key: list(values) for key, values in self.section_memory_ids.items()
            },
            "warnings": list(self.warnings),
        }
