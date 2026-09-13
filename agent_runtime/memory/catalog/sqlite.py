"""SQLite metadata and FTS5 projection of canonical Markdown memory."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.memory.catalog.models import (
    CatalogIssue,
    CatalogStatus,
    CatalogSyncResult,
)
from agent_runtime.memory.catalog.tokenizer import KeywordNormalizer
from agent_runtime.memory.domain.models import (
    MemoryDocument,
    MemoryHit,
    MemoryQuery,
    MemoryStatus,
)
from agent_runtime.memory.io import sha256_file, compact_json
from utils import utc_now


CATALOG_SCHEMA_VERSION = 3
RANKING_VERSION = "keyword-rank-v1"
_ACTIVE_STATUSES = (
    MemoryStatus.DRAFT.value,
    MemoryStatus.VERIFIED.value,
    MemoryStatus.NEEDS_REVIEW.value,
)


class CatalogCompatibilityError(RuntimeError):
    pass


class SQLiteMemoryCatalog:
    """Rebuildable projection; Markdown remains the only canonical document."""

    def __init__(
        self,
        path: str | Path,
        *,
        document_root: str | Path,
        busy_timeout_ms: int = 5000,
        candidate_multiplier: int = 8,
        normalizer: KeywordNormalizer | None = None,
        journal_mode: str = "WAL",
    ) -> None:
        path_value = str(path or "").strip()
        if not path_value:
            raise ValueError("memory catalog path is required")
        self.path = Path(path_value).resolve()
        self.document_root = Path(document_root).resolve()
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.candidate_multiplier = min(100, max(1, int(candidate_multiplier)))
        self.normalizer = normalizer or KeywordNormalizer()
        self.journal_mode = str(journal_mode or "WAL").upper()
        if self.journal_mode not in {"WAL", "DELETE"}:
            raise ValueError("catalog journal_mode must be WAL or DELETE")

    @classmethod
    def from_config(cls, config: Any) -> "SQLiteMemoryCatalog":
        repo = Path(getattr(config, "repo_path", ".") or ".").resolve()
        catalog_path = Path(
            getattr(config, "memory_catalog_path", ".repomind/memory/catalog.sqlite3")
        )
        document_root = Path(
            getattr(config, "long_term_memory_path", ".repomind/memory")
        )
        return cls(
            catalog_path if catalog_path.is_absolute() else repo / catalog_path,
            document_root=(
                document_root if document_root.is_absolute() else repo / document_root
            ),
            busy_timeout_ms=getattr(config, "memory_catalog_busy_timeout_ms", 5000),
            candidate_multiplier=getattr(
                config, "memory_keyword_candidate_multiplier", 8
            ),
        )

    def synchronize(self, document: MemoryDocument, document_path: Path) -> None:
        with self._connect() as connection:
            self._assert_compatible(connection)
            self._begin(connection)
            try:
                self._upsert(connection, document, document_path)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def find_exact(self, document: MemoryDocument) -> list[MemoryHit]:
        if not self.path.is_file() and any(
            (self.document_root / bucket).is_dir() for bucket in ('repo', 'user', 'global')
        ):
            raise RuntimeError('Catalog missing for existing Markdown; run memory catalog rebuild')
        with self._connect() as connection:
            self._assert_compatible(connection)
            rows = connection.execute(
                "SELECT memory_id, content_hash FROM memories "
                "WHERE memory_type=? AND repo_id=? AND scope_key=? "
                "AND knowledge_hash=? AND status IN ('draft','verified','needs_review') "
                "ORDER BY memory_id LIMIT 20",
                (document.memory_type.value, document.scope.repo_id,
                 compact_json(document.scope.to_dict()), document.knowledge_hash),
            ).fetchall()
        return [MemoryHit(memory_id=row['memory_id'], score=1.0,
                          source='exact', content_hash=row['content_hash']) for row in rows]

    def sync(
        self,
        documents: Iterable[tuple[MemoryDocument, Path]],
        *,
        remove_missing: bool = True,
    ) -> CatalogSyncResult:
        records = list(documents)
        _assert_unique_records(records)
        with self._connect() as connection:
            self._assert_compatible(connection)
            self._begin(connection)
            try:
                result = self._sync_records(
                    connection,
                    records,
                    remove_missing=remove_missing,
                )
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def rebuild(
        self,
        documents: Iterable[tuple[MemoryDocument, Path]],
    ) -> CatalogSyncResult:
        records = list(documents)
        _assert_unique_records(records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.parent / f".{self.path.name}.{uuid4().hex}.tmp"
        replacement = SQLiteMemoryCatalog(
            temporary,
            document_root=self.document_root,
            busy_timeout_ms=self.busy_timeout_ms,
            candidate_multiplier=self.candidate_multiplier,
            normalizer=self.normalizer,
            journal_mode="DELETE",
        )
        try:
            result = replacement.sync(records, remove_missing=True)
            with replacement._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO catalog_metadata(key, value) VALUES (?, ?)",
                    ("last_rebuild_at", utc_now()),
                )
                integrity = str(
                    connection.execute("PRAGMA integrity_check").fetchone()[0]
                )
                if integrity.lower() != "ok":
                    raise RuntimeError(f"temporary catalog integrity check failed: {integrity}")
            self._checkpoint_existing()
            os.replace(temporary, self.path)
            return result
        finally:
            for candidate in (
                temporary,
                Path(str(temporary) + "-wal"),
                Path(str(temporary) + "-shm"),
            ):
                if candidate.exists():
                    candidate.unlink()

    def status(
        self,
        documents: Iterable[tuple[MemoryDocument, Path]],
    ) -> CatalogStatus:
        records = list(documents)
        issues: list[CatalogIssue] = []
        try:
            _assert_unique_records(records)
        except ValueError as exc:
            issues.append(CatalogIssue("duplicate_document", str(exc)))
        if not self.path.is_file():
            issues.append(CatalogIssue("catalog_missing", "catalog database is missing"))
            return CatalogStatus(len(records), 0, tuple(issues))
        try:
            with self._connect(initialize=False, read_only=True) as connection:
                return self._status_with_connection(connection, records, issues)
        except (OSError, sqlite3.DatabaseError) as exc:
            issues.append(CatalogIssue("database_error", str(exc)))
            return CatalogStatus(len(records), 0, tuple(issues))

    def _status_with_connection(
        self,
        connection: sqlite3.Connection,
        records: list[tuple[MemoryDocument, Path]],
        issues: list[CatalogIssue],
    ) -> CatalogStatus:
        issues.extend(self._compatibility_issues(connection))
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            issues.append(CatalogIssue("integrity_error", integrity))
        try:
            rows = {
                str(row["memory_id"]): row
                for row in connection.execute(
                    "SELECT memory_id, document_path, content_hash, memory_type, "
                    "status, scope_level, repo_id, module, title, updated_at "
                    "FROM memories"
                )
            }
        except sqlite3.DatabaseError as exc:
            issues.append(CatalogIssue("schema_error", str(exc)))
            return CatalogStatus(len(records), 0, tuple(issues))
        document_ids: set[str] = set()
        for document, path in records:
            document_ids.add(document.memory_id)
            row = rows.get(document.memory_id)
            if row is None:
                issues.append(
                    CatalogIssue(
                        "missing_projection",
                        "Markdown document is not indexed",
                        document.memory_id,
                    )
                )
                continue
            relative_path = self._document_path(path)
            if str(row["document_path"]) != relative_path:
                issues.append(
                    CatalogIssue(
                        "path_mismatch",
                        f"expected {relative_path}, indexed {row['document_path']}",
                        document.memory_id,
                    )
                )
            if str(row["content_hash"]) != sha256_file(path):
                issues.append(
                    CatalogIssue(
                        "stale_projection",
                        "Markdown content hash differs from catalog",
                        document.memory_id,
                    )
                )
            indexed_metadata = (
                str(row["memory_type"]),
                str(row["status"]),
                str(row["scope_level"]),
                str(row["repo_id"]),
                str(row["module"]),
                str(row["title"]),
                str(row["updated_at"]),
            )
            expected_metadata = (
                document.memory_type.value,
                document.status.value,
                document.scope.level.value,
                document.scope.repo_id,
                document.scope.module,
                document.title,
                document.updated_at,
            )
            if indexed_metadata != expected_metadata:
                issues.append(
                    CatalogIssue(
                        "metadata_mismatch",
                        "structured Catalog fields differ from Markdown",
                        document.memory_id,
                    )
                )
        for memory_id in sorted(set(rows) - document_ids):
            issues.append(
                CatalogIssue(
                    "orphan_projection",
                    "Catalog row has no Markdown document",
                    memory_id,
                )
            )
        try:
            fts_counts = {
                str(row[0]): int(row[1])
                for row in connection.execute(
                    "SELECT memory_id, COUNT(*) FROM memory_fts GROUP BY memory_id"
                )
            }
            for memory_id in sorted(set(rows) - set(fts_counts)):
                issues.append(
                    CatalogIssue(
                        "missing_fts",
                        "Catalog row has no FTS projection",
                        memory_id,
                    )
                )
            for memory_id in sorted(set(fts_counts) - set(rows)):
                issues.append(
                    CatalogIssue(
                        "orphan_fts",
                        "FTS row has no Catalog metadata row",
                        memory_id,
                    )
                )
            for memory_id, count in sorted(fts_counts.items()):
                if count > 1:
                    issues.append(
                        CatalogIssue(
                            "duplicate_fts",
                            f"FTS contains {count} rows for this memory",
                            memory_id,
                        )
                    )
        except sqlite3.DatabaseError as exc:
            issues.append(CatalogIssue("fts_error", str(exc)))
        return CatalogStatus(len(records), len(rows), tuple(issues))

    def keyword_search(self, query: MemoryQuery) -> list[MemoryHit]:
        """
            通过关键词使用 FTS 搜索
        """
        if not self.path.is_file():
            raise CatalogCompatibilityError(
                "memory catalog is missing; rebuild the catalog before searching"
            )
        tokens = self.normalizer.tokens(query.text, limit=64)
        if not tokens:
            return []
        match_expression = " OR ".join(_fts_quote(token) for token in tokens)
        sql, parameters = self._search_sql(query, match_expression)
        with self._connect(initialize=False, read_only=True) as connection:
            self._assert_compatible(connection)
            rows = list(connection.execute(sql, parameters))
            facets = self._facets(connection, [str(row["memory_id"]) for row in rows])
        ranked: list[MemoryHit] = []
        query_text = " ".join(query.text.casefold().split())
        query_tokens = set(tokens)
        for rank, row in enumerate(rows, start=1):
            memory_id = str(row["memory_id"])
            memory_facets = facets.get(memory_id, {})
            title_tokens = set(self.normalizer.tokens(str(row["title"]), limit=128))
            matched_title = sorted(query_tokens & title_tokens)
            matched_facets = sorted(
                query_tokens
                & set(
                    self.normalizer.tokens(
                        tuple(
                            value
                            for values in memory_facets.values()
                            for value in values
                        ),
                        limit=256,
                    )
                )
            )
            reasons = [f"keyword rank {rank}"]
            score = 0.55 / rank
            normalized_title = " ".join(str(row["title"]).casefold().split())
            if query_text and query_text in normalized_title:
                score += 0.15
                reasons.append("exact title phrase")
            elif matched_title:
                score += 0.08
                reasons.append(f"title tokens: {', '.join(matched_title[:5])}")
            if matched_facets:
                score += 0.05
                reasons.append(f"scope/tag tokens: {', '.join(matched_facets[:5])}")
            if str(row["status"]) == MemoryStatus.VERIFIED.value:
                score += 0.08
                reasons.append("verified")
            score += float(row["confidence"]) * 0.09
            score += float(row["evidence_strength"]) * 0.08
            ranked.append(
                MemoryHit(
                    memory_id=memory_id,
                    score=round(min(1.0, score), 6),
                    source="sqlite_fts5",
                    reasons=tuple(reasons),
                    content_hash=str(row["content_hash"]),
                    document_path=str(row["document_path"]),
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.memory_id))
        return ranked[: query.limit]

    def _sync_records(
        self,
        connection: sqlite3.Connection,
        records: list[tuple[MemoryDocument, Path]],
        *,
        remove_missing: bool,
    ) -> CatalogSyncResult:
        indexed = 0
        unchanged = 0
        memory_ids: set[str] = set()
        for document, path in records:
            memory_ids.add(document.memory_id)
            content_hash = sha256_file(path)
            relative_path = self._document_path(path)
            row = connection.execute(
                "SELECT content_hash, document_path FROM memories WHERE memory_id = ?",
                (document.memory_id,),
            ).fetchone()
            if (
                row is not None
                and str(row["content_hash"]) == content_hash
                and str(row["document_path"]) == relative_path
            ):
                unchanged += 1
                continue
            self._upsert(
                connection,
                document,
                path,
                content_hash=content_hash,
                relative_path=relative_path,
            )
            indexed += 1
        removed = 0
        if remove_missing:
            existing = {
                str(row[0]) for row in connection.execute("SELECT memory_id FROM memories")
            }
            for memory_id in sorted(existing - memory_ids):
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
                connection.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                removed += 1
        return CatalogSyncResult(indexed=indexed, unchanged=unchanged, removed=removed)

    def _upsert(
        self,
        connection: sqlite3.Connection,
        document: MemoryDocument,
        document_path: Path,
        *,
        content_hash: str | None = None,
        relative_path: str | None = None,
    ) -> None:
        content_hash = content_hash or sha256_file(document_path)
        relative_path = relative_path or self._document_path(document_path)
        connection.execute(
            """
            INSERT INTO memories(
                memory_id, document_path, content_hash, memory_type, status,
                scope_level, repo_id, module, title, knowledge, applicability,
                invalidation, confidence, evidence_strength, source_task_id,
                revision, knowledge_hash, scope_key, created_at, updated_at, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                document_path=excluded.document_path,
                content_hash=excluded.content_hash,
                memory_type=excluded.memory_type,
                status=excluded.status,
                scope_level=excluded.scope_level,
                repo_id=excluded.repo_id,
                module=excluded.module,
                title=excluded.title,
                knowledge=excluded.knowledge,
                applicability=excluded.applicability,
                invalidation=excluded.invalidation,
                confidence=excluded.confidence,
                evidence_strength=excluded.evidence_strength,
                source_task_id=excluded.source_task_id,
                revision=excluded.revision,
                knowledge_hash=excluded.knowledge_hash,
                scope_key=excluded.scope_key,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                indexed_at=excluded.indexed_at
            """,
            (
                document.memory_id,
                relative_path,
                content_hash,
                document.memory_type.value,
                document.status.value,
                document.scope.level.value,
                document.scope.repo_id,
                document.scope.module,
                document.title,
                document.knowledge,
                document.applicability,
                document.invalidation,
                document.confidence,
                document.evidence_strength,
                document.source.task_id,
                document.revision,
                document.knowledge_hash,
                compact_json(document.scope.to_dict()),
                document.created_at,
                document.updated_at,
                utc_now(),
            ),
        )
        connection.execute("DELETE FROM memory_facets WHERE memory_id = ?", (document.memory_id,))
        facets = _document_facets(document)
        connection.executemany(
            "INSERT INTO memory_facets(memory_id, facet_type, facet_value) VALUES (?, ?, ?)",
            (
                (document.memory_id, facet_type, value)
                for facet_type, values in facets.items()
                for value in values
            ),
        )
        connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (document.memory_id,))
        connection.execute(
            """
            INSERT INTO memory_fts(
                memory_id, title, knowledge, applicability, triggers,
                tags, files, symbols, search_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.memory_id,
                document.title,
                document.knowledge,
                document.applicability,
                " ".join(document.triggers),
                " ".join(document.tags),
                " ".join(document.scope.files),
                " ".join(document.scope.symbols),
                " ".join(self.normalizer.document_tokens(document)),
            ),
        )

    def _search_sql(
        self,
        query: MemoryQuery,
        match_expression: str,
    ) -> tuple[str, list[Any]]:
        """ 多表联合过滤
            memory_fts：按查询词找候选、计算 BM25
                 ↓ JOIN memory_id
            memories：按仓库、状态、类型过滤，提供评分和一致性检查字段
                 ↓ 可选 EXISTS
            memory_facets：按标签或 scope_hints 进一步过滤
        """
        clauses = ["memory_fts MATCH ?"]
        parameters: list[Any] = [match_expression]
        statuses = tuple(item.value for item in query.statuses) or _ACTIVE_STATUSES
        clauses.append(f"m.status IN ({_placeholders(statuses)})")
        parameters.extend(statuses)
        # 仓库级过滤
        if query.repo_id:
            clauses.append("(m.scope_level IN ('global', 'user') OR m.repo_id = ?)")
            parameters.append(query.repo_id)
        if query.memory_types:
            values = tuple(item.value for item in query.memory_types)
            clauses.append(f"m.memory_type IN ({_placeholders(values)})")
            parameters.extend(values)
        for tag in query.tags:
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_facets mf "
                "WHERE mf.memory_id = m.memory_id AND mf.facet_type = 'tag' "
                "AND mf.facet_value = ?)"
            )
            parameters.append(tag)
        # 已移除这个字段
        if query.scope_hints:
            clauses.append(
                "(m.module IN ("
                + _placeholders(query.scope_hints)
                + ") OR EXISTS (SELECT 1 FROM memory_facets mf "
                "WHERE mf.memory_id = m.memory_id "
                "AND mf.facet_type IN ('file', 'symbol', 'trigger', 'tag') "
                "AND mf.facet_value IN ("
                + _placeholders(query.scope_hints)
                + ")))"
            )
            parameters.extend(query.scope_hints)
            parameters.extend(query.scope_hints)
        candidate_limit = min(800, max(query.limit, query.limit * self.candidate_multiplier))
        parameters.append(candidate_limit)
        sql = f"""
            SELECT
                m.*,
                bm25(memory_fts, 0.0, 8.0, 5.0, 3.0, 4.0, 4.0, 5.0, 5.0, 1.0)
                    AS keyword_rank
            FROM memory_fts
            JOIN memories m ON m.memory_id = memory_fts.memory_id
            WHERE {' AND '.join(clauses)}
            ORDER BY keyword_rank ASC, m.memory_id ASC
            LIMIT ?
        """
        return sql, parameters

    def _facets(
        self,
        connection: sqlite3.Connection,
        memory_ids: list[str],
    ) -> dict[str, dict[str, list[str]]]:
        if not memory_ids:
            return {}
        result: dict[str, dict[str, list[str]]] = {}
        sql = (
            "SELECT memory_id, facet_type, facet_value FROM memory_facets "
            f"WHERE memory_id IN ({_placeholders(memory_ids)})"
        )
        for row in connection.execute(sql, memory_ids):
            facets = result.setdefault(str(row["memory_id"]), {})
            facets.setdefault(str(row["facet_type"]), []).append(str(row["facet_value"]))
        return result

    def _document_path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.document_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"memory document is outside document root: {resolved}") from exc

    def _assert_compatible(self, connection: sqlite3.Connection) -> None:
        issues = self._compatibility_issues(connection)
        if issues:
            raise CatalogCompatibilityError(
                "; ".join(issue.message for issue in issues) + "; rebuild the catalog"
            )

    def _compatibility_issues(
        self,
        connection: sqlite3.Connection,
    ) -> list[CatalogIssue]:
        issues: list[CatalogIssue] = []
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != CATALOG_SCHEMA_VERSION:
            issues.append(
                CatalogIssue(
                    "schema_version",
                    f"catalog schema {version}, expected {CATALOG_SCHEMA_VERSION}",
                )
            )
            return issues
        try:
            metadata = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT key, value FROM catalog_metadata")
            }
        except sqlite3.DatabaseError as exc:
            return [CatalogIssue("metadata_error", str(exc))]
        if metadata.get("tokenizer_version") != self.normalizer.version:
            issues.append(
                CatalogIssue(
                    "tokenizer_version",
                    f"catalog tokenizer {metadata.get('tokenizer_version')!r}, "
                    f"expected {self.normalizer.version!r}",
                )
            )
        if metadata.get("ranking_version") != RANKING_VERSION:
            issues.append(
                CatalogIssue(
                    "ranking_version",
                    f"catalog ranking {metadata.get('ranking_version')!r}, "
                    f"expected {RANKING_VERSION!r}",
                )
            )
        return issues

    def _checkpoint_existing(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self._connect(initialize=False) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            # Rebuild must still recover a corrupt derived Catalog.
            pass
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    def _begin(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @contextmanager
    def _connect(
        self,
        *,
        initialize: bool = True,
        read_only: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        if read_only:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro",
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
                uri=True,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
            )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            if read_only:
                connection.execute("PRAGMA query_only = ON")
            else:
                connection.execute(f"PRAGMA journal_mode = {self.journal_mode}")
                connection.execute("PRAGMA synchronous = FULL")
            if initialize:
                self._migrate(connection)
            yield connection
        finally:
            connection.close()

    def _migrate(self, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CATALOG_SCHEMA_VERSION:
            raise CatalogCompatibilityError(
                f"catalog schema {version} is newer than supported {CATALOG_SCHEMA_VERSION}"
            )
        if version == CATALOG_SCHEMA_VERSION:
            return
        if version != 0:
            raise CatalogCompatibilityError(f"unsupported catalog schema {version}")
        try:
            connection.executescript(
                f"""
                BEGIN IMMEDIATE;
                CREATE TABLE catalog_metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE memories(
                    memory_id TEXT PRIMARY KEY,
                    document_path TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scope_level TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    module TEXT NOT NULL,
                    title TEXT NOT NULL,
                    knowledge TEXT NOT NULL,
                    applicability TEXT NOT NULL,
                    invalidation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_strength REAL NOT NULL,
                    source_task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    knowledge_hash TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE INDEX memories_filter_idx
                    ON memories(repo_id, status, memory_type, scope_level);
                CREATE INDEX memories_exact_idx
                    ON memories(memory_type, repo_id, scope_key, knowledge_hash, status, memory_id);
                CREATE TABLE memory_facets(
                    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    facet_type TEXT NOT NULL,
                    facet_value TEXT NOT NULL,
                    PRIMARY KEY(memory_id, facet_type, facet_value)
                );
                CREATE INDEX memory_facets_lookup_idx
                    ON memory_facets(facet_type, facet_value, memory_id);
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    memory_id UNINDEXED,
                    title,
                    knowledge,
                    applicability,
                    triggers,
                    tags,
                    files,
                    symbols,
                    search_tokens,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                INSERT INTO catalog_metadata(key, value)
                    VALUES ('tokenizer_version', '{self.normalizer.version}');
                INSERT INTO catalog_metadata(key, value)
                    VALUES ('ranking_version', '{RANKING_VERSION}');
                INSERT INTO catalog_metadata(key, value)
                    VALUES ('created_at', '{utc_now()}');
                PRAGMA user_version = {CATALOG_SCHEMA_VERSION};
                COMMIT;
                """
            )
        except sqlite3.DatabaseError as exc:
            if connection.in_transaction:
                connection.rollback()
            raise RuntimeError(
                "SQLite FTS5 catalog initialization failed; "
                "the Python SQLite build must include FTS5"
            ) from exc


def _document_facets(document: MemoryDocument) -> dict[str, tuple[str, ...]]:
    source_tasks = document.extensions.get("source_tasks") or ()
    if isinstance(source_tasks, str):
        source_tasks = (source_tasks,)
    elif not isinstance(source_tasks, (list, tuple, set)):
        source_tasks = ()
    raw = {
        "tag": document.tags,
        "trigger": document.triggers,
        "file": document.scope.files,
        "symbol": document.scope.symbols,
        "source_task": source_tasks,
    }
    return {
        facet_type: tuple(
            dict.fromkeys(
                str(value).strip()[:500]
                for value in values
                if str(value).strip()
            )
        )
        for facet_type, values in raw.items()
    }


def _assert_unique_records(records: list[tuple[MemoryDocument, Path]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for document, _ in records:
        if document.memory_id in seen:
            duplicates.add(document.memory_id)
        seen.add(document.memory_id)
    if duplicates:
        raise ValueError(f"duplicate Markdown memory IDs: {', '.join(sorted(duplicates))}")


def _fts_quote(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _placeholders(values: Iterable[object]) -> str:
    values = tuple(values)
    if not values:
        raise ValueError("SQL placeholder collection cannot be empty")
    return ", ".join("?" for _ in values)
