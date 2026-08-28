"""Repository identity and source-evidence capture without shell dependencies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent_runtime.memory.archive.security import is_sensitive_path


def repository_id(repo_path: str | Path) -> str:
    root = Path(repo_path or ".").resolve()
    origin = _git_config_value(root / ".git" / "config", "url")
    identity = origin or root.as_posix()
    return f"sha256:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def repository_revision(repo_path: str | Path) -> str:
    git_dir = _git_dir(Path(repo_path or ".").resolve())
    if git_dir is None:
        return ""
    head = git_dir / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not value.startswith("ref:"):
        return value[:200]
    ref_name = value.partition(":")[2].strip()
    ref_path = git_dir.joinpath(*ref_name.split("/"))
    try:
        return ref_path.read_text(encoding="utf-8").strip()[:200]
    except OSError:
        return _packed_ref(git_dir / "packed-refs", ref_name)


def capture_source_snapshots(
    repo_path: str | Path,
    file_paths: list[str],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[dict[str, Any]]:
    root = Path(repo_path or ".").resolve()
    remaining = max(0, int(max_total_bytes))
    snapshots: list[dict[str, Any]] = []
    for relative in _unique_paths(file_paths):
        record: dict[str, Any] = {"path": relative}
        if is_sensitive_path(relative):
            record["error"] = "sensitive_file_omitted"
            snapshots.append(record)
            continue
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            record["error"] = "path_outside_repository"
            snapshots.append(record)
            continue
        if not target.is_file():
            record["error"] = "file_not_found"
            snapshots.append(record)
            continue
        try:
            content = target.read_bytes()
        except OSError as exc:
            record["error"] = f"read_failed:{type(exc).__name__}"
            snapshots.append(record)
            continue
        record.update(
            {
                "source_sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
        allowed = min(max(0, int(max_file_bytes)), remaining)
        if len(content) <= allowed:
            record["content"] = content.decode("utf-8", errors="replace")
            remaining -= len(content)
        else:
            record["content_omitted"] = True
            record["reason"] = "source_snapshot_budget_exceeded"
        snapshots.append(record)
    return snapshots


def workspace_fingerprint(snapshots: list[dict[str, Any]]) -> str:
    pairs = sorted(
        (str(item.get("path") or ""), str(item.get("source_sha256") or ""))
        for item in snapshots
        if item.get("path") and item.get("source_sha256")
    )
    if not pairs:
        return ""
    digest = hashlib.sha256()
    for path, content_hash in pairs:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _unique_paths(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip().replace("\\", "/")
        if text and text not in result:
            result.append(text)
    return result


def _git_dir(root: Path) -> Path | None:
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.lower().startswith("gitdir:"):
        return None
    candidate = Path(text.partition(":")[2].strip())
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _packed_ref(path: Path, ref_name: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        if line.startswith("#") or line.startswith("^"):
            continue
        revision, _, name = line.partition(" ")
        if name.strip() == ref_name:
            return revision.strip()[:200]
    return ""


def _git_config_value(path: Path, key: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    for line in lines:
        stripped = line.strip()
        name, separator, value = stripped.partition("=")
        if separator and name.strip().lower() == key.lower():
            return value.strip()
    return ""
