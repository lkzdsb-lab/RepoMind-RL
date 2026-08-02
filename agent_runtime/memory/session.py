"""Disk-backed, session-scoped memory for multi-turn agent conversations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from loguru import logger

from model.agent.graph import AgentState
from agent_runtime.memory.file_cache import normalize_snapshot, prune_cache
from utils import _clean_string_list


class SessionMemoryService:
    """Owns online conversation memory without long-term promotion side effects."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_turns: int = 6,
        max_chars: int = 12000,
        file_cache_max_files: int = 50,
        file_cache_max_spans: int = 200,
        file_cache_max_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.path = Path(path)
        self.max_turns = max(1, int(max_turns))
        self.max_chars = max(1000, int(max_chars))
        self.file_cache_max_files = max(1, int(file_cache_max_files))
        self.file_cache_max_spans = max(1, int(file_cache_max_spans))
        self.file_cache_max_bytes = max(1, int(file_cache_max_bytes))
        self._schema_lock = Lock()
        self._initialized = False

    @classmethod
    def from_config(cls, config: Any) -> "SessionMemoryService":
        return cls(
            getattr(config, "session_memory_path", ".repomind/session_memory.db"),
            max_turns=getattr(config, "session_memory_max_turns", 6),
            max_chars=getattr(config, "session_memory_max_chars", 12000),
            file_cache_max_files=getattr(config, "session_file_cache_max_files", 50),
            file_cache_max_spans=getattr(config, "session_file_cache_max_spans", 200),
            file_cache_max_bytes=getattr(
                config, "session_file_cache_max_bytes", 10 * 1024 * 1024
            ),
        )

    def open_session(self, repo_path: str) -> str:
        """Return the latest active session for a repository, creating one if needed."""
        repo_key = _repo_key(repo_path)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id FROM sessions
                WHERE repo_path = ? AND status = 'active'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (repo_key,),
            ).fetchone()
            if row:
                return str(row["session_id"])
            return self._insert_session(connection, repo_key)

    def new_session(self, repo_path: str) -> str:
        """Close active repository sessions and start a clean conversation."""
        repo_key = _repo_key(repo_path)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET status = 'closed', updated_at = ? "
                "WHERE repo_path = ? AND status = 'active'",
                (now, repo_key),
            )
            return self._insert_session(connection, repo_key)

    def prepare_turn(self, session_id: str, user_message: str) -> dict[str, Any]:
        """Build the bounded context supplied directly to task analysis."""
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not session:
                raise ValueError(f"Unknown memory session: {session_id}")

            topic = self._active_topic(connection, session)
            topic_id = str(topic["topic_id"]) if topic else ""
            playbook = self._latest_playbook(connection, topic_id)
            turns = self._recent_turns(connection, session_id, topic_id)
            memories = self._active_memories(connection, session_id, topic_id)
            file_cache, file_order = self._validated_file_cache(connection, session)

        pack: dict[str, Any] = {
            "session_id": session_id,
            "current_message": str(user_message or "").strip(),
            "topic": _topic_dict(topic),
            "playbook": _json_object(playbook["content"]) if playbook else {},
            "recent_turns": turns,
            **memories,
        }
        pack["_read_file_cache"] = file_cache
        pack["_read_file_order"] = file_order
        pack["rendered"] = _bounded_json(
            {key: value for key, value in pack.items() if not key.startswith("_")},
            self.max_chars,
        )
        return pack

    def commit_turn(
        self,
        session_id: str,
        user_message: str,
        state: AgentState,
    ) -> None:
        """Persist one completed turn and atomically advance its topic playbook."""
        now = _utc_now()
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not session:
                raise ValueError(f"Unknown memory session: {session_id}")
            topic = self._active_topic(connection, session)
            if not topic:
                topic = self._create_topic(connection, session_id, user_message, now)

            topic_id = str(topic["topic_id"])
            turn_index = int(topic["turn_count"] or 0) + 1
            result = _turn_result(state)
            cursor = connection.execute(
                """
                INSERT INTO turns(
                    session_id, topic_id, turn_index, task_id, user_message,
                    assistant_result, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    topic_id,
                    turn_index,
                    str(state.get("task_id") or ""),
                    str(user_message or "").strip(),
                    json.dumps(result, ensure_ascii=False, default=str),
                    str(state.get("status") or "unknown"),
                    now,
                ),
            )
            turn_id = int(cursor.lastrowid)
            self._insert_memory_items(
                connection,
                session_id,
                topic_id,
                turn_id,
                turn_index,
                state,
                result,
                now,
            )
            self._insert_evidence_refs(connection, session_id, topic_id, turn_id, state, now)
            self._persist_file_cache(connection, session_id, turn_id, state, now)
            self._insert_playbook(
                connection,
                session_id,
                topic_id,
                turn_id,
                turn_index,
                user_message,
                result,
                now,
            )
            connection.execute(
                """
                UPDATE topics SET summary = ?, turn_count = ?, updated_at = ?
                WHERE topic_id = ?
                """,
                (_topic_summary(user_message, result), turn_index, now, topic_id),
            )
            connection.execute(
                "UPDATE sessions SET active_topic_id = ?, updated_at = ? WHERE session_id = ?",
                (topic_id, now, session_id),
            )
            self._archive_expired_memories(connection, topic_id, turn_index)

        logger.bind(session_id=session_id, task_id=state.get("task_id")).info(
            "session memory turn committed topic_id={} turn_index={}",
            topic_id,
            turn_index,
        )

    def _validated_file_cache(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Load snapshots whose stored revision still matches the repository file."""
        session_id = str(session["session_id"])
        repo = Path(str(session["repo_path"])).resolve()
        rows = connection.execute(
            """
            SELECT file_path, file_revision, snapshot, cache_order
            FROM session_file_cache
            WHERE session_id = ?
            ORDER BY cache_order ASC, updated_at ASC
            LIMIT ?
            """,
            (session_id, self.file_cache_max_files),
        ).fetchall()
        cache: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        stale_paths: list[str] = []
        for row in rows:
            file_path = str(row["file_path"] or "").strip()
            target = (repo / file_path).resolve()
            if not file_path or (target != repo and repo not in target.parents) or not target.is_file():
                stale_paths.append(file_path)
                continue
            try:
                revision = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError:
                stale_paths.append(file_path)
                continue
            if revision != str(row["file_revision"] or ""):
                stale_paths.append(file_path)
                continue
            snapshot = normalize_snapshot(_json_object(row["snapshot"]), file_path=file_path)
            if not snapshot:
                stale_paths.append(file_path)
                continue
            snapshot["file_revision"] = revision
            snapshot["session_cache_reused"] = True
            cache[file_path] = snapshot
            order.append(file_path)
        if stale_paths:
            connection.executemany(
                "DELETE FROM session_file_cache WHERE session_id = ? AND file_path = ?",
                [(session_id, path) for path in stale_paths if path],
            )
        logger.bind(session_id=session_id).debug(
            "session file cache validated reused={} stale={}",
            len(cache),
            len(stale_paths),
        )
        return cache, order

    def _persist_file_cache(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        turn_id: int,
        state: AgentState,
        now: str,
    ) -> None:
        raw_cache = state.get("read_file_cache")
        if not isinstance(raw_cache, dict):
            return
        cache, order = prune_cache(
            raw_cache,
            state.get("read_file_order", []),
            max_files=self.file_cache_max_files,
            max_spans=self.file_cache_max_spans,
            max_bytes=self.file_cache_max_bytes,
        )
        for cache_order, file_path in enumerate(order):
            snapshot = cache.get(file_path)
            if not isinstance(snapshot, dict):
                continue
            revision = str(snapshot.get("file_revision") or "").strip()
            if not revision:
                continue
            persisted = dict(snapshot)
            persisted.pop("session_cache_reused", None)
            connection.execute(
                """
                INSERT INTO session_file_cache(
                    session_id, file_path, file_revision, snapshot,
                    cache_order, size_bytes, turn_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, file_path) DO UPDATE SET
                    file_revision = excluded.file_revision,
                    snapshot = excluded.snapshot,
                    cache_order = excluded.cache_order,
                    size_bytes = excluded.size_bytes,
                    turn_id = excluded.turn_id,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    file_path,
                    revision,
                    json.dumps(persisted, ensure_ascii=False, default=str),
                    cache_order,
                    int(persisted.get("size_bytes") or 0),
                    turn_id,
                    now,
                ),
            )
        if order:
            placeholders = ",".join("?" for _ in order)
            connection.execute(
                f"DELETE FROM session_file_cache WHERE session_id = ? "
                f"AND file_path NOT IN ({placeholders})",
                (session_id, *order),
            )
        else:
            connection.execute(
                "DELETE FROM session_file_cache WHERE session_id = ?",
                (session_id,),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            self._ensure_schema(connection)
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._initialized:
            return
        with self._schema_lock:
            if self._initialized:
                return
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(_SCHEMA)
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(memory_items)")
            }
            if "expires_turn" not in columns:
                connection.execute("ALTER TABLE memory_items ADD COLUMN expires_turn INTEGER")
            cache_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(session_file_cache)")
            }
            if "size_bytes" not in cache_columns:
                connection.execute(
                    "ALTER TABLE session_file_cache "
                    "ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0"
                )
            self._initialized = True

    def _insert_session(self, connection: sqlite3.Connection, repo_path: str) -> str:
        session_id = str(uuid4())
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO sessions(session_id, repo_path, status, created_at, updated_at)
            VALUES (?, ?, 'active', ?, ?)
            """,
            (session_id, repo_path, now, now),
        )
        return session_id

    def _active_topic(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
    ) -> sqlite3.Row | None:
        topic_id = str(session["active_topic_id"] or "")
        if not topic_id:
            return None
        return connection.execute(
            "SELECT * FROM topics WHERE topic_id = ? AND status = 'active'",
            (topic_id,),
        ).fetchone()

    def _create_topic(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        user_message: str,
        now: str,
    ) -> sqlite3.Row:
        topic_id = str(uuid4())
        title = str(user_message or "New topic").strip()[:160] or "New topic"
        connection.execute(
            """
            INSERT INTO topics(
                topic_id, session_id, title, status, summary, turn_count, created_at, updated_at
            ) VALUES (?, ?, ?, 'active', '', 0, ?, ?)
            """,
            (topic_id, session_id, title, now, now),
        )
        return connection.execute(
            "SELECT * FROM topics WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()

    def _latest_playbook(
        self,
        connection: sqlite3.Connection,
        topic_id: str,
    ) -> sqlite3.Row | None:
        if not topic_id:
            return None
        return connection.execute(
            """
            SELECT content FROM playbook_versions
            WHERE topic_id = ? ORDER BY version DESC LIMIT 1
            """,
            (topic_id,),
        ).fetchone()

    def _recent_turns(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        topic_id: str,
    ) -> list[dict[str, Any]]:
        if not topic_id:
            return []
        rows = connection.execute(
            """
            SELECT turn_index, user_message, assistant_result, status, created_at
            FROM turns WHERE session_id = ? AND topic_id = ?
            ORDER BY turn_index DESC LIMIT ?
            """,
            (session_id, topic_id, self.max_turns),
        ).fetchall()
        return [
            {
                "turn_index": int(row["turn_index"]),
                "user_message": str(row["user_message"]),
                "assistant_result": _json_object(row["assistant_result"]),
                "status": str(row["status"]),
                "created_at": str(row["created_at"]),
            }
            for row in reversed(rows)
        ]

    def _active_memories(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        topic_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        if not topic_id:
            return _empty_memory_layers()
        rows = connection.execute(
            """
            SELECT category, content, importance, created_at
            FROM memory_items
            WHERE session_id = ? AND topic_id = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT ?
            """,
            (session_id, topic_id, self.max_turns * 4),
        ).fetchall()
        layers = _empty_memory_layers()
        for row in reversed(rows):
            category = str(row["category"])
            key = f"{category}_memory" if category != "preference" else "preferences"
            if key not in layers:
                continue
            layers[key].append({
                "category": str(row["category"]),
                "content": _json_object(row["content"]),
                "importance": float(row["importance"]),
                "created_at": str(row["created_at"]),
            })
        return layers

    def _insert_memory_items(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        topic_id: str,
        turn_id: int,
        turn_index: int,
        state: AgentState,
        result: dict[str, Any],
        now: str,
    ) -> None:
        items = [
            ("conversation", result, 1.0, turn_index + self.max_turns),
            ("tool", _tool_memory(state), 0.8, turn_index + self.max_turns),
        ]
        if result.get("next_steps"):
            items.append(
                (
                    "temporary",
                    {"next_steps": result["next_steps"], "source_turn": turn_index},
                    0.7,
                    turn_index + 2,
                )
            )
        for category, content, importance, expires_turn in items:
            connection.execute(
                """
                INSERT INTO memory_items(
                    memory_id, session_id, topic_id, turn_id, category, content,
                    importance, status, expires_turn, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    str(uuid4()), session_id, topic_id, turn_id, category,
                    json.dumps(content, ensure_ascii=False), importance, expires_turn, now,
                ),
            )

    def _insert_evidence_refs(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        topic_id: str,
        turn_id: int,
        state: AgentState,
        now: str,
    ) -> None:
        refs: list[tuple[str, str]] = []
        refs.extend(("candidate_file", value) for value in _clean_string_list(state.get("candidate_files"), 20, 400))
        refs.extend(("edited_file", value) for value in _clean_string_list(state.get("edited_files"), 20, 400))
        for item in state.get("verification_commands", [])[-10:]:
            if isinstance(item, dict) and item.get("command"):
                refs.append(("verification_command", str(item["command"])[:1000]))
        for kind, reference in refs:
            connection.execute(
                """
                INSERT INTO evidence_refs(
                    evidence_id, session_id, topic_id, turn_id, kind, reference, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), session_id, topic_id, turn_id, kind, reference, now),
            )

    def _insert_playbook(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        topic_id: str,
        turn_id: int,
        version: int,
        user_message: str,
        result: dict[str, Any],
        now: str,
    ) -> None:
        content = {
            "current_objective": str(user_message or "").strip(),
            "latest_outcome": result.get("summary", ""),
            "work_done": result.get("work_done", []),
            "files": result.get("files", []),
            "verification": result.get("verification", []),
            "next_steps": result.get("next_steps", []),
            "updated_turn": version,
        }
        connection.execute(
            """
            INSERT INTO playbook_versions(
                playbook_id, session_id, topic_id, version, turn_id, content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()), session_id, topic_id, version, turn_id,
                json.dumps(content, ensure_ascii=False), now,
            ),
        )

    def _archive_expired_memories(
        self,
        connection: sqlite3.Connection,
        topic_id: str,
        current_turn: int,
    ) -> None:
        connection.execute(
            """
            UPDATE memory_items SET status = 'archived'
            WHERE topic_id = ? AND status = 'active' AND expires_turn IS NOT NULL
              AND expires_turn <= ?
            """,
            (topic_id, current_turn),
        )


def _turn_result(state: AgentState) -> dict[str, Any]:
    report = state.get("final_report") if isinstance(state.get("final_report"), dict) else {}
    return {
        "summary": str(report.get("summary") or state.get("error") or "").strip()[:2000],
        "findings": _clean_string_list(report.get("findings"), 20, 600),
        "work_done": _clean_string_list(report.get("work_done"), 12, 500),
        "files": _clean_string_list(
            state.get("edited_files") or state.get("candidate_files"), 20, 400
        ),
        "verification": _clean_string_list(report.get("test_results"), 10, 600),
        "next_steps": _clean_string_list(report.get("next_steps"), 10, 500),
        "status": str(state.get("status") or "unknown"),
    }


def _tool_memory(state: AgentState) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for call in state.get("tool_calls", [])[-20:]:
        if not isinstance(call, dict):
            continue
        output = call.get("output") if isinstance(call.get("output"), dict) else {}
        actions.append(
            {
                "name": str(call.get("name") or ""),
                "status": "error" if call.get("error") or output.get("error") else "ok",
                "file": str(output.get("file_path") or "")[:400],
                "command": str(output.get("command") or "")[:600],
                "exit_code": output.get("exit_code"),
            }
        )
    return {"actions": actions}


def _empty_memory_layers() -> dict[str, list[dict[str, Any]]]:
    return {
        "conversation_memory": [],
        "tool_memory": [],
        "temporary_memory": [],
        "preferences": [],
    }


def _topic_dict(topic: sqlite3.Row | None) -> dict[str, Any]:
    if not topic:
        return {}
    return {
        "topic_id": str(topic["topic_id"]),
        "title": str(topic["title"]),
        "summary": str(topic["summary"]),
        "turn_count": int(topic["turn_count"]),
    }


def _topic_summary(user_message: str, result: dict[str, Any]) -> str:
    objective = str(user_message or "").strip()
    outcome = str(result.get("summary") or "").strip()
    return f"Objective: {objective}\nLatest outcome: {outcome}"[:3000]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _bounded_json(value: dict[str, Any], max_chars: int) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= max_chars:
        return rendered
    compact = dict(value)
    compact["conversation_memory"] = []
    rendered = json.dumps(compact, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= max_chars:
        return rendered
    excerpt_size = max(200, max_chars // 3)
    return json.dumps(
        {
            "truncated": True,
            "latest_context_excerpt": rendered[-excerpt_size:],
        },
        ensure_ascii=False,
        indent=2,
    )


def _repo_key(repo_path: str) -> str:
    return str(Path(repo_path or ".").resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    repo_path TEXT NOT NULL,
    status TEXT NOT NULL,
    active_topic_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_repo_status
    ON sessions(repo_path, status, updated_at);

CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topics_session_status ON topics(session_id, status);

CREATE TABLE IF NOT EXISTS turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    topic_id TEXT NOT NULL REFERENCES topics(topic_id),
    turn_index INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_result TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(topic_id, turn_index)
);

CREATE TABLE IF NOT EXISTS memory_items (
    memory_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    topic_id TEXT NOT NULL REFERENCES topics(topic_id),
    turn_id INTEGER REFERENCES turns(turn_id),
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL,
    expires_turn INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_active
    ON memory_items(session_id, topic_id, status, created_at);

CREATE TABLE IF NOT EXISTS playbook_versions (
    playbook_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    topic_id TEXT NOT NULL REFERENCES topics(topic_id),
    version INTEGER NOT NULL,
    turn_id INTEGER NOT NULL REFERENCES turns(turn_id),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(topic_id, version)
);

CREATE TABLE IF NOT EXISTS evidence_refs (
    evidence_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    topic_id TEXT NOT NULL REFERENCES topics(topic_id),
    turn_id INTEGER NOT NULL REFERENCES turns(turn_id),
    kind TEXT NOT NULL,
    reference TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_file_cache (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    file_path TEXT NOT NULL,
    file_revision TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    cache_order INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    turn_id INTEGER REFERENCES turns(turn_id),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(session_id, file_path)
);
CREATE INDEX IF NOT EXISTS idx_session_file_cache_order
    ON session_file_cache(session_id, cache_order);

CREATE TABLE IF NOT EXISTS memory_jobs (
    job_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_items(memory_id),
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    lease_until TEXT,
    pipeline_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_jobs_ready
    ON memory_jobs(status, available_at);
"""
