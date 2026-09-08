"""Value objects for the rebuildable knowledge embedding projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    dimensions: int
    model_fingerprint: str


@dataclass(frozen=True)
class SemanticMatch:
    memory_id: str
    score: float
    knowledge_hash: str
    model_fingerprint: str
    revision: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SemanticSyncResult:
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
