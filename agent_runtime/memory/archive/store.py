"""Filesystem-backed immutable Task Archive store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model.agent.graph import AgentState
from utils import utc_now

from agent_runtime.memory.io import (
    atomic_directory,
    canonical_json_bytes,
    json_lines_bytes,
    read_json,
    safe_identifier,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    write_bytes,
)
from agent_runtime.memory.archive.repository import (
    capture_source_snapshots,
    repository_id,
    repository_revision,
    workspace_fingerprint,
)
from agent_runtime.memory.archive.sanitizer import sanitize
from agent_runtime.memory.domain.models import ArchiveFileRecord, TaskArchiveManifest


class TaskArchiveIntegrityError(RuntimeError):
    pass


class TaskArchiveStoreImpl:
    """Write-once task archives whose artifacts are covered by a manifest."""

    def __init__(
        self,
        root: str | Path,
        *,
        repo_path: str | Path,
        max_text_chars: int = 200_000,
        max_source_file_bytes: int = 200_000,
        max_source_total_bytes: int = 2_000_000,
    ) -> None:
        self.repo_path = Path(repo_path or ".").resolve()
        root_value = str(root or "").strip()
        if not root_value:
            raise ValueError("task archive root is required")
        configured_root = Path(root_value)
        self.root = (
            configured_root.resolve()
            if configured_root.is_absolute()
            else (self.repo_path / configured_root).resolve()
        )
        self.max_text_chars = max(1000, int(max_text_chars))
        self.max_source_file_bytes = max(0, int(max_source_file_bytes))
        self.max_source_total_bytes = max(0, int(max_source_total_bytes))

    @classmethod
    def from_config(cls, config: Any) -> "TaskArchiveStoreImpl":
        return cls(
            getattr(config, "task_archive_path", ".repomind/archive/tasks"),
            repo_path=getattr(config, "repo_path", "."),
            max_text_chars=getattr(config, "task_archive_max_text_chars", 200_000),
            max_source_file_bytes=getattr(config, "task_archive_max_source_file_bytes", 200_000),
            max_source_total_bytes=getattr(config, "task_archive_max_source_total_bytes", 2_000_000),
        )

    def archive(
        self,
        state: AgentState,
        trace_path: str | Path = "",
        *,
        session_snapshot: dict[str, Any] | None = None,
    ) -> TaskArchiveManifest:
        task_id = safe_identifier(str(state.get("task_id") or ""), field_name="task_id")
        destination = self.task_path(task_id)
        if destination.exists():
            errors = self.verify(task_id)
            if errors:
                raise TaskArchiveIntegrityError(
                    f"existing task archive is invalid: {'; '.join(errors)}"
                )
            return self.load_manifest(task_id)

        with atomic_directory(destination) as temporary:
            manifest = self._write_archive(
                temporary,
                state,
                trace_path,
                session_snapshot=session_snapshot,
            )
            return manifest

    def load_manifest(self, task_id: str) -> TaskArchiveManifest:
        directory = self.task_path(task_id)
        return TaskArchiveManifest.from_dict(read_json(directory / "manifest.json"))

    def verify(self, task_id: str) -> list[str]:
        directory = self.task_path(task_id)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return ["manifest.json is missing"]
        try:
            manifest = TaskArchiveManifest.from_dict(read_json(manifest_path))
        except (OSError, ValueError, TypeError) as exc:
            return [f"manifest.json is invalid: {exc}"]
        errors: list[str] = []
        if manifest.task_id != task_id:
            errors.append("manifest task_id does not match directory")
        for logical_name, record in manifest.files.items():
            try:
                path = safe_relative_path(directory, record.path)
            except ValueError as exc:
                errors.append(f"{logical_name}: {exc}")
                continue
            if not path.is_file():
                errors.append(f"{logical_name}: file is missing")
                continue
            actual_size = path.stat().st_size
            if actual_size != record.size_bytes:
                errors.append(
                    f"{logical_name}: size mismatch expected={record.size_bytes} actual={actual_size}"
                )
            actual_hash = sha256_file(path)
            if actual_hash != record.sha256:
                errors.append(f"{logical_name}: sha256 mismatch")
        return errors

    def task_path(self, task_id: str) -> Path:
        return self.root / safe_identifier(task_id, field_name="task_id")

    def artifact_path(self, task_id: str, logical_name: str) -> Path:
        manifest = self.load_manifest(task_id)
        record = manifest.files.get(str(logical_name))
        if record is None:
            raise KeyError(f"archive artifact is missing: {logical_name}")
        return safe_relative_path(self.task_path(task_id), record.path)

    def read_json_artifact(self, task_id: str, logical_name: str) -> object:
        return json.loads(
            self.artifact_path(task_id, logical_name).read_text(encoding="utf-8")
        )

    def read_text_artifact(self, task_id: str, logical_name: str) -> str:
        return self.artifact_path(task_id, logical_name).read_text(encoding="utf-8")

    def manifest_hash(self, task_id: str) -> str:
        return sha256_file(self.task_path(task_id) / "manifest.json")

    def _write_archive(
        self,
        directory: Path,
        state: AgentState,
        trace_path: str | Path,
        *,
        session_snapshot: dict[str, Any] | None,
    ) -> TaskArchiveManifest:
        clean_state = sanitize(state, max_text_chars=self.max_text_chars)
        clean_state["task_archive_status"] = "archived"
        clean_state["task_archive_path"] = "."
        clean_state["task_archive_error"] = ""
        source_paths = _source_paths(state)
        snapshots = capture_source_snapshots(
            self.repo_path,
            source_paths,
            max_file_bytes=self.max_source_file_bytes,
            max_total_bytes=self.max_source_total_bytes,
        )
        clean_snapshots = sanitize(snapshots, max_text_chars=self.max_text_chars)
        clean_session = sanitize(
            session_snapshot or {}, max_text_chars=self.max_text_chars
        )
        artifacts = _artifact_payloads(clean_state, clean_snapshots, clean_session)
        records: dict[str, ArchiveFileRecord] = {}
        for logical_name, (relative_path, content, media_type) in artifacts.items():
            target = safe_relative_path(directory, relative_path)
            write_bytes(target, content)
            records[logical_name] = ArchiveFileRecord(
                path=relative_path,
                sha256=sha256_bytes(content),
                size_bytes=len(content),
                media_type=media_type,
            )

        manifest = TaskArchiveManifest(
            task_id=str(state.get("task_id") or ""),
            session_id=str(state.get("session_id") or ""),
            repo_id=repository_id(self.repo_path),
            repo_revision=repository_revision(self.repo_path),
            workspace_fingerprint=workspace_fingerprint(snapshots),
            status=str(state.get("status") or "unknown"),
            pipeline_version="archive-v1",
            created_at=utc_now(),
            files=records,
            metadata={
                "trace_source": Path(trace_path).name if trace_path else "",
                "source_file_count": len(snapshots),
            },
        )
        write_bytes(directory / "manifest.json", canonical_json_bytes(manifest.to_dict()))
        return manifest


def _artifact_payloads(
    state: dict[str, Any],
    source_snapshots: Any,
    session_snapshot: dict[str, Any],
) -> dict[str, tuple[str, bytes, str]]:
    trajectory = state.get("trajectory") if isinstance(state.get("trajectory"), list) else []
    final_report = state.get("final_report") if isinstance(state.get("final_report"), dict) else {}
    verification = {
        "test_results": state.get("test_results", []),
        "verification_commands": state.get("verification_commands", []),
        "command_results": state.get("command_results", []),
        "verification_stale": bool(state.get("verification_stale", False)),
    }
    result = {
        "events": ("events.jsonl", json_lines_bytes(trajectory), "application/x-ndjson"),
        "final_state": ("final_state.json", canonical_json_bytes(state), "application/json"),
        "final_report": ("final_report.json", canonical_json_bytes(final_report), "application/json"),
        "verification": ("verification.json", canonical_json_bytes(verification), "application/json"),
        "memory_candidates": (
            "memory_candidates.json",
            canonical_json_bytes({"candidates": state.get("memory_candidates", [])}),
            "application/json",
        ),
        "source_snapshots": (
            "source_snapshots.json",
            canonical_json_bytes({"files": source_snapshots}),
            "application/json",
        ),
        "session_snapshot": (
            "session_snapshot.json",
            canonical_json_bytes(session_snapshot),
            "application/json",
        ),
    }
    patch = str(state.get("patch") or "")
    if patch:
        result["diff"] = ("diff.patch", patch.encode("utf-8"), "text/x-diff")
    return result


def _source_paths(state: AgentState) -> list[str]:
    values: list[str] = []
    for field_name in ("edited_files", "candidate_files", "read_file_order"):
        raw = state.get(field_name)
        if not isinstance(raw, list):
            continue
        for item in raw:
            path = str(item or "").strip().replace("\\", "/")
            if path and path not in values:
                values.append(path)
    return values
