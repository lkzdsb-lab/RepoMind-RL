"""Explicit schema migration registry for Markdown memory metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_runtime.memory.domain.models import MEMORY_SCHEMA_VERSION


Migration = Callable[[dict[str, Any]], dict[str, Any]]


class MemorySchemaMigrator:
    def __init__(self, migrations: Mapping[int, Migration] | None = None) -> None:
        self._migrations = dict(migrations or {})

    def migrate(
        self,
        metadata: Mapping[str, Any],
        *,
        target_version: int = MEMORY_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        current = int(metadata.get("schema_version") or 0)
        if current <= 0:
            raise ValueError("memory schema_version is required")
        if current > target_version:
            raise ValueError(
                f"memory schema version {current} is newer than supported {target_version}"
            )
        migrated = dict(metadata)
        while current < target_version:
            migration = self._migrations.get(current)
            if migration is None:
                raise ValueError(
                    f"no memory migration registered for version {current} -> {current + 1}"
                )
            migrated = migration(dict(migrated))
            next_version = int(migrated.get("schema_version") or 0)
            if next_version != current + 1:
                raise ValueError(
                    f"memory migration {current} must produce schema_version {current + 1}"
                )
            current = next_version
        return migrated
