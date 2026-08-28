"""Versioned domain models for durable memory and replayable task archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from utils import utc_now


MEMORY_SCHEMA_VERSION = 1
ARCHIVE_SCHEMA_VERSION = 1


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MemoryType(_StringEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ANTI_PATTERN = "anti_pattern"
    PREFERENCE = "preference"


class MemoryStatus(_StringEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    STALE = "stale"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class ScopeLevel(_StringEnum):
    GLOBAL = "global"
    USER = "user"
    REPO = "repo"
    MODULE = "module"
    FILE = "file"
    SYMBOL = "symbol"


def _required_text(value: Any, field_name: str, *, limit: int = 4000) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    if len(text) > limit:
        raise ValueError(f"{field_name} exceeds {limit} characters")
    return text


def _score(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return parsed


def _enum_value(enum_type: type[_StringEnum], value: Any, field_name: str) -> Any:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _strings(value: Any, *, limit: int = 100, item_limit: int = 1000) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:item_limit])
        if len(result) >= limit:
            break
    return result


@dataclass(frozen=True)
class MemoryScope:
    level: ScopeLevel
    repo_id: str = ""
    module: str = ""
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", _enum_value(ScopeLevel, self.level, "scope.level"))
        object.__setattr__(self, "repo_id", str(self.repo_id or "").strip()[:500])
        object.__setattr__(self, "module", str(self.module or "").strip()[:500])
        object.__setattr__(self, "files", tuple(_strings(self.files, limit=100, item_limit=500)))
        object.__setattr__(self, "symbols", tuple(_strings(self.symbols, limit=100, item_limit=500)))
        if self.level not in {ScopeLevel.GLOBAL, ScopeLevel.USER} and not self.repo_id:
            raise ValueError("scope.repo_id is required for repository-scoped memory")
        if self.level == ScopeLevel.MODULE and not self.module:
            raise ValueError("scope.module is required for module-scoped memory")
        if self.level == ScopeLevel.FILE and not self.files:
            raise ValueError("scope.files is required for file-scoped memory")
        if self.level == ScopeLevel.SYMBOL and not self.symbols:
            raise ValueError("scope.symbols is required for symbol-scoped memory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "repo_id": self.repo_id,
            "module": self.module,
            "files": list(self.files),
            "symbols": list(self.symbols),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryScope":
        return cls(
            level=_enum_value(ScopeLevel, data.get("level"), "scope.level"),
            repo_id=str(data.get("repo_id") or ""),
            module=str(data.get("module") or ""),
            files=tuple(_strings(data.get("files"))),
            symbols=tuple(_strings(data.get("symbols"))),
        )


@dataclass(frozen=True)
class MemoryEvidence:
    evidence_id: str
    kind: str
    reference: str
    strength: float = 0.5
    content_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _required_text(self.evidence_id, "evidence_id", limit=200))
        object.__setattr__(self, "kind", _required_text(self.kind, "evidence.kind", limit=80))
        object.__setattr__(self, "reference", _required_text(self.reference, "evidence.reference"))
        object.__setattr__(self, "strength", _score(self.strength, "evidence.strength"))
        object.__setattr__(self, "content_hash", str(self.content_hash or "").strip()[:128])
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryEvidence":
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True)
class MemorySource:
    task_id: str
    archive_path: str
    archive_hash: str
    repo_revision: str = ""
    pipeline_version: str = "memory-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _required_text(self.task_id, "source.task_id", limit=200))
        object.__setattr__(self, "archive_path", _required_text(self.archive_path, "source.archive_path"))
        object.__setattr__(self, "archive_hash", _required_text(self.archive_hash, "source.archive_hash", limit=128))
        object.__setattr__(self, "repo_revision", str(self.repo_revision or "").strip()[:500])
        object.__setattr__(self, "pipeline_version", _required_text(self.pipeline_version, "source.pipeline_version", limit=100))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemorySource":
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True)
class MemoryDocument:
    memory_id: str
    memory_type: MemoryType
    title: str
    status: MemoryStatus
    scope: MemoryScope
    source: MemorySource
    knowledge: str
    applicability: str
    triggers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    evidence: tuple[MemoryEvidence, ...] = ()
    invalidation: str = ""
    confidence: float = 0.5
    evidence_strength: float = 0.0
    schema_version: int = MEMORY_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema version: {self.schema_version}")
        object.__setattr__(self, "memory_id", _required_text(self.memory_id, "memory_id", limit=200))
        object.__setattr__(self, "memory_type", _enum_value(MemoryType, self.memory_type, "memory_type"))
        object.__setattr__(self, "status", _enum_value(MemoryStatus, self.status, "status"))
        if not isinstance(self.scope, MemoryScope):
            raise ValueError("scope must be a MemoryScope")
        if not isinstance(self.source, MemorySource):
            raise ValueError("source must be a MemorySource")
        object.__setattr__(self, "title", _required_text(self.title, "title", limit=300))
        object.__setattr__(self, "knowledge", _required_text(self.knowledge, "knowledge", limit=50000))
        object.__setattr__(self, "applicability", _required_text(self.applicability, "applicability", limit=10000))
        object.__setattr__(self, "triggers", tuple(_strings(self.triggers, limit=50, item_limit=300)))
        object.__setattr__(self, "tags", tuple(sorted(_strings(self.tags, limit=50, item_limit=100))))
        evidence = tuple(self.evidence or ())
        if any(not isinstance(item, MemoryEvidence) for item in evidence):
            raise ValueError("evidence must contain only MemoryEvidence values")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        object.__setattr__(self, "evidence_strength", _score(self.evidence_strength, "evidence_strength"))
        object.__setattr__(self, "invalidation", str(self.invalidation or "").strip()[:10000])
        object.__setattr__(self, "extensions", dict(self.extensions or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "title": self.title,
            "status": self.status.value,
            "scope": self.scope.to_dict(),
            "source": self.source.to_dict(),
            "triggers": list(self.triggers),
            "tags": list(self.tags),
            "confidence": self.confidence,
            "evidence_strength": self.evidence_strength,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "knowledge": self.knowledge,
            "applicability": self.applicability,
            "evidence": [item.to_dict() for item in self.evidence],
            "invalidation": self.invalidation,
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryDocument":
        known = set(cls.__dataclass_fields__)
        extensions = dict(data.get("extensions") or {})
        extensions.update({str(key): value for key, value in data.items() if key not in known})
        return cls(
            schema_version=int(data.get("schema_version", MEMORY_SCHEMA_VERSION)),
            memory_id=str(data.get("memory_id") or ""),
            memory_type=_enum_value(MemoryType, data.get("memory_type"), "memory_type"),
            title=str(data.get("title") or ""),
            status=_enum_value(MemoryStatus, data.get("status"), "status"),
            scope=MemoryScope.from_dict(dict(data.get("scope") or {})),
            source=MemorySource.from_dict(dict(data.get("source") or {})),
            knowledge=str(data.get("knowledge") or ""),
            applicability=str(data.get("applicability") or ""),
            triggers=tuple(_strings(data.get("triggers"))),
            tags=tuple(_strings(data.get("tags"))),
            evidence=tuple(
                MemoryEvidence.from_dict(item)
                for item in data.get("evidence", [])
                if isinstance(item, Mapping)
            ),
            invalidation=str(data.get("invalidation") or ""),
            confidence=data.get("confidence", 0.5),
            evidence_strength=data.get("evidence_strength", 0.0),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            extensions=extensions,
        )


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    repo_id: str = ""
    memory_types: tuple[MemoryType, ...] = ()
    scope_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    limit: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required_text(self.text, "query.text", limit=10000))
        if not 1 <= int(self.limit) <= 100:
            raise ValueError("query.limit must be between 1 and 100")


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    score: float
    source: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryRelation:
    source_memory_id: str
    target_memory_id: str
    relation_type: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class ConsolidationResult:
    task_id: str
    pipeline_version: str = ""
    status: str = ""
    memory_ids: tuple[str, ...] = ()
    rejected_candidates: int = 0
    extractor_source: str = ""
    warnings: tuple[str, ...] = ()
    run_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConsolidationResult":
        return cls(
            task_id=str(data.get("task_id") or ""),
            pipeline_version=str(data.get("pipeline_version") or ""),
            status=str(data.get("status") or ""),
            memory_ids=tuple(str(item) for item in data.get("memory_ids", [])),
            rejected_candidates=int(data.get("rejected_candidates") or 0),
            extractor_source=str(data.get("extractor_source") or ""),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            run_path=str(data.get("run_path") or ""),
        )


@dataclass(frozen=True)
class SessionMemorySnapshot:
    """Bounded session context visible before the archived task began."""

    session_id: str
    exported_at: str
    topic: Mapping[str, Any] = field(default_factory=dict)
    playbook: Mapping[str, Any] = field(default_factory=dict)
    recent_turns: tuple[Mapping[str, Any], ...] = ()
    memory_layers: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _required_text(self.session_id, "session_id", limit=200))
        object.__setattr__(self, "exported_at", _required_text(self.exported_at, "exported_at", limit=100))
        object.__setattr__(self, "topic", dict(self.topic or {}))
        object.__setattr__(self, "playbook", dict(self.playbook or {}))
        object.__setattr__(
            self,
            "recent_turns",
            tuple(dict(item) for item in self.recent_turns if isinstance(item, Mapping)),
        )
        object.__setattr__(
            self,
            "memory_layers",
            {
                str(key): tuple(
                    dict(item) for item in values if isinstance(item, Mapping)
                )
                for key, values in dict(self.memory_layers or {}).items()
                if isinstance(values, (list, tuple))
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exported_at": self.exported_at,
            "topic": dict(self.topic),
            "playbook": dict(self.playbook),
            "recent_turns": [dict(item) for item in self.recent_turns],
            "memory_layers": {
                key: [dict(item) for item in values]
                for key, values in sorted(self.memory_layers.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionMemorySnapshot":
        return cls(
            session_id=str(data.get("session_id") or ""),
            exported_at=str(data.get("exported_at") or ""),
            topic=dict(data.get("topic") or {}),
            playbook=dict(data.get("playbook") or {}),
            recent_turns=tuple(data.get("recent_turns") or ()),
            memory_layers=dict(data.get("memory_layers") or {}),
        )


@dataclass(frozen=True)
class ArchiveFileRecord:
    path: str
    sha256: str
    size_bytes: int
    media_type: str = "application/json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchiveFileRecord":
        return cls(
            path=_required_text(data.get("path"), "archive file path", limit=500),
            sha256=_required_text(data.get("sha256"), "archive file sha256", limit=128),
            size_bytes=max(0, int(data.get("size_bytes") or 0)),
            media_type=str(data.get("media_type") or "application/octet-stream")[:100],
        )


@dataclass(frozen=True)
class TaskArchiveManifest:
    task_id: str
    repo_id: str
    status: str
    created_at: str
    files: Mapping[str, ArchiveFileRecord]
    session_id: str = ""
    repo_revision: str = ""
    workspace_fingerprint: str = ""
    pipeline_version: str = "archive-v1"
    schema_version: int = ARCHIVE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) != ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"unsupported archive schema version: {self.schema_version}")
        object.__setattr__(self, "task_id", _required_text(self.task_id, "task_id", limit=200))
        object.__setattr__(self, "repo_id", _required_text(self.repo_id, "repo_id", limit=200))
        object.__setattr__(self, "status", _required_text(self.status, "status", limit=80))
        object.__setattr__(self, "created_at", _required_text(self.created_at, "created_at", limit=100))
        object.__setattr__(self, "files", dict(self.files or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "workspace_fingerprint": self.workspace_fingerprint,
            "status": self.status,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "files": {key: value.to_dict() for key, value in sorted(self.files.items())},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskArchiveManifest":
        raw_files = data.get("files") or {}
        return cls(
            schema_version=int(data.get("schema_version", ARCHIVE_SCHEMA_VERSION)),
            task_id=str(data.get("task_id") or ""),
            session_id=str(data.get("session_id") or ""),
            repo_id=str(data.get("repo_id") or ""),
            repo_revision=str(data.get("repo_revision") or ""),
            workspace_fingerprint=str(data.get("workspace_fingerprint") or ""),
            status=str(data.get("status") or ""),
            pipeline_version=str(data.get("pipeline_version") or "archive-v1"),
            created_at=str(data.get("created_at") or ""),
            files={
                str(key): ArchiveFileRecord.from_dict(value)
                for key, value in raw_files.items()
                if isinstance(value, Mapping)
            },
            metadata=dict(data.get("metadata") or {}),
        )
