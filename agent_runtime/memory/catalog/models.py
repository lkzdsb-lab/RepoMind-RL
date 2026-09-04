"""Catalog operation results kept independent from the canonical memory model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogIssue:
    code: str
    message: str
    memory_id: str = ""


@dataclass(frozen=True)
class CatalogStatus:
    document_count: int
    catalog_count: int
    issues: tuple[CatalogIssue, ...] = ()

    @property
    def healthy(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class CatalogSyncResult:
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0

