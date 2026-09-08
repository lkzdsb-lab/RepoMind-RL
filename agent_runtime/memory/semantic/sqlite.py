"""SQLite projection of MemoryDocument knowledge embeddings."""

from __future__ import annotations

import math
import os
import sqlite3
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.memory.domain.models import (
    MemoryDocument,
    MemoryStatus,
    normalize_memory_knowledge,
)
from agent_runtime.memory.io import compact_json
from agent_runtime.memory.semantic.embeddings import EmbeddingClient
from agent_runtime.memory.semantic.models import SemanticMatch, SemanticSyncResult
from utils import utc_now


SEMANTIC_SCHEMA_VERSION = 1


class SemanticIndexCompatibilityError(RuntimeError):
    pass


class SQLiteMemorySemanticIndex:
    def __init__(
        self,
        path: str | Path,
        *,
        embedding_client: EmbeddingClient,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "WAL",
    ) -> None:
        self.path = Path(path).resolve()
        self.embedding_client = embedding_client
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.journal_mode = str(journal_mode or "WAL").upper()
        if self.journal_mode not in {"WAL", "DELETE"}:
            raise ValueError("semantic index journal_mode must be WAL or DELETE")
        self._cache: dict[str, tuple[tuple[float, ...], str]] = {}

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        embedding_client: EmbeddingClient,
    ) -> "SQLiteMemorySemanticIndex":
        repo = Path(getattr(config, "repo_path", ".") or ".").resolve()
        configured = Path(
            getattr(config, "memory_semantic_index_path", ".repomind/memory/semantic.sqlite3")
        )
        return cls(
            configured if configured.is_absolute() else repo / configured,
            embedding_client=embedding_client,
            busy_timeout_ms=getattr(config, "memory_catalog_busy_timeout_ms", 5000),
        )

    def synchronize(self, document: MemoryDocument) -> None:
        self.synchronize_many([document])

    def readiness_issue(self) -> str:
        """Bounded diagnostic; full coverage checks belong to offline maintenance."""
        if not self.path.is_file():
            return 'semantic index missing; run memory semantic rebuild to index existing Markdown'
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                'SELECT model_fingerprint FROM memory_embeddings LIMIT 1'
            ).fetchone()
            if row is not None and row[0] != self.embedding_client.model_fingerprint:
                return 'embedding model changed; run memory semantic rebuild before relying on semantic deduplication'
        finally:
            connection.close()
        return ''

    def synchronize_many(self, documents: Iterable[MemoryDocument]) -> SemanticSyncResult:
        records = list(documents)
        if not records:
            self._initialize()
            return SemanticSyncResult()
        # 从表中查询过滤可复用的索引
        reusable = self._stored_vectors(records)
        missing_by_hash = {
            document.knowledge_hash: document
            for document in records
            if document.memory_id not in reusable
        }
        generated: dict[str, tuple[float, ...]] = {}
        fingerprint = self.embedding_client.model_fingerprint
        # 重新生成修改后的 vector
        if missing_by_hash:
            hashes = list(missing_by_hash)
            batch = self.embedding_client.embed(
                [normalize_memory_knowledge(missing_by_hash[value].knowledge) for value in hashes]
            )
            if len(batch.vectors) != len(hashes):
                raise RuntimeError("embedding provider returned an unexpected vector count")
            if batch.model_fingerprint != self.embedding_client.model_fingerprint:
                raise RuntimeError("embedding provider fingerprint changed during synchronization")
            fingerprint = batch.model_fingerprint
            generated = dict(zip(hashes, batch.vectors))
        connection = self._connect()
        # 写入数据库
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for document in records:
                    stored = reusable.get(document.memory_id)
                    vector = stored[0] if stored is not None else generated[document.knowledge_hash]
                    self._upsert(connection, document, vector, fingerprint)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return SemanticSyncResult(
            indexed=len(records) - len(reusable),
            unchanged=len(reusable),
        )

    def semantic_search(
        self,
        document: MemoryDocument,
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[SemanticMatch]:
        vector, fingerprint = self._vector(document)
        scope_key = compact_json(document.scope.to_dict())
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT memory_id, knowledge_hash, model_fingerprint, revision,
                       dimensions, vector
                FROM memory_embeddings
                WHERE memory_type = ? AND repo_id = ? AND scope_key = ?
                  AND status IN (?, ?, ?) AND model_fingerprint = ?
                  AND memory_id != ?
                """,
                (
                    document.memory_type.value,
                    document.scope.repo_id,
                    scope_key,
                    MemoryStatus.DRAFT.value,
                    MemoryStatus.VERIFIED.value,
                    MemoryStatus.NEEDS_REVIEW.value,
                    fingerprint,
                    document.memory_id,
                ),
            ).fetchall()
        finally:
            connection.close()
        matches: list[SemanticMatch] = []
        for row in rows:
            stored = _decode_vector(bytes(row["vector"]), int(row["dimensions"]))
            score = _cosine(vector, stored)
            if score < float(min_score):
                continue
            matches.append(
                SemanticMatch(
                    memory_id=str(row["memory_id"]),
                    score=round(score, 6),
                    knowledge_hash=str(row["knowledge_hash"]),
                    model_fingerprint=str(row["model_fingerprint"]),
                    revision=int(row["revision"]),
                )
            )
        return sorted(matches, key=lambda item: (-item.score, item.memory_id))[: max(1, int(limit))]

    def rebuild(self, documents: Iterable[MemoryDocument]) -> SemanticSyncResult:
        records = list(documents)
        temporary = self.path.parent / f".{self.path.name}.{uuid4().hex}.tmp"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        replacement = SQLiteMemorySemanticIndex(
            temporary,
            embedding_client=self.embedding_client,
            busy_timeout_ms=self.busy_timeout_ms,
            journal_mode="DELETE",
        )
        try:
            replacement._initialize()
            if records:
                unique_documents = {
                    document.knowledge_hash: document
                    for document in records
                }
                unique_hashes = list(unique_documents)
                batch = self.embedding_client.embed(
                    [
                        normalize_memory_knowledge(unique_documents[value].knowledge)
                        for value in unique_hashes
                    ]
                )
                if len(batch.vectors) != len(unique_hashes):
                    raise RuntimeError("embedding provider returned an unexpected vector count")
                vectors = dict(zip(unique_hashes, batch.vectors))
                connection = replacement._connect()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    for document in records:
                        replacement._upsert(
                            connection,
                            document,
                            vectors[document.knowledge_hash],
                            batch.model_fingerprint,
                        )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                finally:
                    connection.close()
            check = replacement._connect(read_only=True)
            try:
                integrity = str(check.execute("PRAGMA integrity_check").fetchone()[0])
            finally:
                check.close()
            if integrity.lower() != "ok":
                raise RuntimeError(f"temporary semantic index integrity check failed: {integrity}")
            self._checkpoint_existing()
            os.replace(temporary, self.path)
        finally:
            for candidate in (
                temporary,
                Path(str(temporary) + "-wal"),
                Path(str(temporary) + "-shm"),
            ):
                if candidate.exists():
                    candidate.unlink()
        return SemanticSyncResult(indexed=len(records))

    def _vector(self, document: MemoryDocument) -> tuple[tuple[float, ...], str]:
        cached = self._cache.get(document.knowledge_hash)
        if cached is not None:
            return cached
        batch = self.embedding_client.embed([normalize_memory_knowledge(document.knowledge)])
        result = (batch.vectors[0], batch.model_fingerprint)
        self._cache[document.knowledge_hash] = result
        return result

    def _stored_vectors(
        self,
        documents: list[MemoryDocument],
    ) -> dict[str, tuple[tuple[float, ...], str]]:
        """ 查找可复用的旧索引"""
        connection = self._connect(read_only=True)
        try:
            result: dict[str, tuple[tuple[float, ...], str]] = {}
            for document in documents:
                row = connection.execute(
                    """
                    SELECT dimensions, vector, model_fingerprint
                    FROM memory_embeddings
                    WHERE memory_id = ? AND knowledge_hash = ? AND model_fingerprint = ?
                    """,
                    (
                        document.memory_id,
                        document.knowledge_hash,
                        self.embedding_client.model_fingerprint,
                    ),
                ).fetchone()
                if row is not None:
                    result[document.memory_id] = (
                        _decode_vector(bytes(row["vector"]), int(row["dimensions"])),
                        str(row["model_fingerprint"]),
                    )
            return result
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        connection.close()

    def _checkpoint_existing(self) -> None:
        if not self.path.is_file():
            return
        try:
            connection = self._connect()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
        except sqlite3.DatabaseError:
            pass
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    def _upsert(
        self,
        connection: sqlite3.Connection,
        document: MemoryDocument,
        vector: tuple[float, ...],
        fingerprint: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_embeddings(
                memory_id, knowledge_hash, model_fingerprint, provider, model,
                dimensions, vector,
                memory_type, repo_id, scope_key, status, revision, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                knowledge_hash=excluded.knowledge_hash,
                model_fingerprint=excluded.model_fingerprint,
                provider=excluded.provider,
                model=excluded.model,
                dimensions=excluded.dimensions,
                vector=excluded.vector,
                memory_type=excluded.memory_type,
                repo_id=excluded.repo_id,
                scope_key=excluded.scope_key,
                status=excluded.status,
                revision=excluded.revision,
                indexed_at=excluded.indexed_at
            """,
            (
                document.memory_id,
                document.knowledge_hash,
                fingerprint,
                str(getattr(self.embedding_client, "provider", "unknown")),
                str(getattr(self.embedding_client, "model", "unknown")),
                len(vector),
                _encode_vector(vector),
                document.memory_type.value,
                document.scope.repo_id,
                compact_json(document.scope.to_dict()),
                document.status.value,
                document.revision,
                utc_now(),
            ),
        )

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        if not read_only:
            connection.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            connection.execute("PRAGMA synchronous = FULL")
        self._migrate(connection)
        if read_only:
            connection.execute("PRAGMA query_only = ON")
        return connection

    def _migrate(self, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == SEMANTIC_SCHEMA_VERSION:
            return
        if version != 0:
            raise SemanticIndexCompatibilityError(
                f"unsupported semantic index schema {version}"
            )
        connection.executescript(
            f"""
            BEGIN IMMEDIATE;
            CREATE TABLE memory_embeddings(
                memory_id TEXT PRIMARY KEY,
                knowledge_hash TEXT NOT NULL,
                model_fingerprint TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                memory_type TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                status TEXT NOT NULL,
                revision INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE INDEX memory_embeddings_filter_idx
                ON memory_embeddings(memory_type, repo_id, scope_key, status, model_fingerprint);
            PRAGMA user_version = {SEMANTIC_SCHEMA_VERSION};
            COMMIT;
            """
        )


def _encode_vector(vector: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _decode_vector(value: bytes, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0 or len(value) != dimensions * 4:
        raise ValueError("stored embedding vector has invalid dimensions")
    return tuple(struct.unpack(f"<{dimensions}f", value))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
