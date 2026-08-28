"""Filesystem implementation of the canonical Markdown memory store."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_runtime.memory.documents.codec import MarkdownMemoryCodec
from agent_runtime.memory.domain.models import (
    MemoryDocument,
    MemoryStatus,
    ScopeLevel,
)
from agent_runtime.memory.io import atomic_write_bytes, safe_identifier, sha256_file
from utils import utc_now


class MemoryDocumentConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryValidationIssue:
    path: str
    message: str


class MarkdownMemoryDocumentStore:
    def __init__(
        self,
        root: str | Path,
        *,
        repo_path: str | Path = ".",
        codec: MarkdownMemoryCodec | None = None,
    ) -> None:
        repo = Path(repo_path or ".").resolve()
        root_value = str(root or "").strip()
        if not root_value:
            raise ValueError("long-term memory document root is required")
        configured_root = Path(root_value)
        self.root = (
            configured_root.resolve()
            if configured_root.is_absolute()
            else (repo / configured_root).resolve()
        )
        self.codec = codec or MarkdownMemoryCodec()

    @classmethod
    def from_config(cls, config: Any) -> "MarkdownMemoryDocumentStore":
        return cls(
            getattr(config, "long_term_memory_path", ".repomind/memory"),
            repo_path=getattr(config, "repo_path", "."),
        )

    def get(self, memory_id: str) -> MemoryDocument | None:
        record = self._find(memory_id)
        return self._read(record) if record is not None else None

    def save(self, document: MemoryDocument) -> Path:
        target = self.document_path(document)
        existing = self._find(document.memory_id)
        if existing is not None and existing != target:
            raise MemoryDocumentConflictError(
                "memory type or top-level scope cannot change in place; "
                f"existing={existing} requested={target}"
            )
        content = self.codec.encode(document)
        if target.is_file():
            try:
                existing_document = self._read(target)
            except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
                raise MemoryDocumentConflictError(
                    f"refusing to overwrite invalid memory document {target}: {exc}"
                ) from exc
            if existing_document.memory_id != document.memory_id:
                raise MemoryDocumentConflictError(
                    f"path {target} belongs to memory {existing_document.memory_id!r}"
                )
            if target.read_bytes() == content:
                return target
        atomic_write_bytes(target, content)
        return target

    def list(self) -> list[MemoryDocument]:
        return [self._read(path) for path in self._document_paths()]

    def records(self) -> list[tuple[MemoryDocument, Path]]:
        return [(self._read(path), path) for path in self._document_paths()]

    def deprecate(self, memory_id: str, reason: str) -> None:
        document = self.get(memory_id)
        if document is None:
            raise KeyError(f"unknown memory document: {memory_id}")
        deprecation_reason = str(reason or "").strip()
        if not deprecation_reason:
            raise ValueError("deprecation reason is required")
        extensions = dict(document.extensions)
        extensions["deprecation_reason"] = deprecation_reason[:2000]
        archived = replace(
            document,
            status=MemoryStatus.DEPRECATED,
            updated_at=utc_now(),
            extensions=extensions,
        )
        self.save(archived)

    def document_path(self, document: MemoryDocument) -> Path:
        memory_id = safe_identifier(document.memory_id, field_name="memory_id")
        scope_bucket = _scope_bucket(document.scope.level)
        return self.root / scope_bucket / document.memory_type.value / f"{memory_id}.md"

    def content_hash(self, memory_id: str) -> str:
        path = self._find(memory_id)
        if path is None:
            raise KeyError(f"unknown memory document: {memory_id}")
        return sha256_file(path)

    def validate(self) -> list[MemoryValidationIssue]:
        issues: list[MemoryValidationIssue] = []
        seen: dict[str, Path] = {}
        for path in self._document_paths():
            try:
                raw = path.read_bytes()
                document = self.codec.decode(raw)
                expected = self.document_path(document)
                if expected != path:
                    issues.append(
                        MemoryValidationIssue(path.as_posix(), f"expected path is {expected}")
                    )
                previous = seen.get(document.memory_id)
                if previous is not None:
                    issues.append(
                        MemoryValidationIssue(
                            path.as_posix(), f"duplicate memory_id also found at {previous}"
                        )
                    )
                else:
                    seen[document.memory_id] = path
                if self.codec.encode(document) != raw:
                    issues.append(
                        MemoryValidationIssue(path.as_posix(), "document is not canonical")
                    )
            except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
                issues.append(MemoryValidationIssue(path.as_posix(), str(exc)))
        return issues

    def _find(self, memory_id: str) -> Path | None:
        safe_id = safe_identifier(memory_id, field_name="memory_id")
        matches = [path for path in self._document_paths() if path.stem == safe_id]
        if len(matches) > 1:
            raise MemoryDocumentConflictError(
                f"duplicate memory_id {safe_id!r}: {', '.join(str(path) for path in matches)}"
            )
        return matches[0] if matches else None

    def _document_paths(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            path
            for path in self.root.glob("*/*/*.md")
            if path.is_file() and not path.name.startswith(".")
        )

    def _read(self, path: Path) -> MemoryDocument:
        return self.codec.decode(path.read_bytes())


def _scope_bucket(level: ScopeLevel) -> str:
    if level == ScopeLevel.GLOBAL:
        return "global"
    if level == ScopeLevel.USER:
        return "user"
    return "repo"
