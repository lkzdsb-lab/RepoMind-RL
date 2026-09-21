"""Bounded UTF-8 artifacts and a reproducible, secret-excluding source snapshot."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import FileConfig


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows readers may briefly hold a sharing lock on an old progress file.
    for attempt in range(3):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.02 * (attempt + 1))


def read_json(path: Path, *, max_bytes: int = 2_000_000):
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Artifact exceeds size limit")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Artifact must contain a JSON object")
    return value


def source_files(root: Path, deadline: float | None = None):
    # Secrets, agent memory and evaluator artifacts must never enter the sandbox.
    excluded = set(FileConfig.LIST_IGNORED_DIRS) | {"evaluation", "config.json"}
    pending = [root]
    count = 0
    visited = 0
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            visited += 1
            if visited > 50000:
                raise ValueError("Source snapshot exceeds 50000 directory entries")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Source snapshot deadline exceeded")
            if path.is_symlink() or path.name.startswith(".") or path.name in excluded:
                continue
            if getattr(path, "is_junction", lambda: False)():
                continue
            if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
                continue
            if not path.resolve().is_relative_to(root):
                continue
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                count += 1
                if count > 10000:
                    raise ValueError("Source snapshot exceeds 10000 files")
                yield path


def fingerprint(root: Path, destination: Path | None = None, *, deadline: float | None = None) -> str:
    """ 计算执行前源码指纹"""
    digest = hashlib.sha256()
    total = 0
    for path in source_files(root, deadline):
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"Source file exceeds 5 MB: {path.relative_to(root)}")
        data = path.read_bytes()
        total += len(data)
        if total > 100_000_000:
            raise ValueError("Source snapshot exceeds 100 MB")
        relative = path.relative_to(root)
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).digest())
        mode = path.stat().st_mode & 0o777
        digest.update(str(mode & 0o111).encode("ascii") + b"\0")
        if destination is not None:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(mode)
    return digest.hexdigest()


class EvidenceStore:
    def __init__(self, directory: Path, source_fingerprint: str = ""):
        self.directory = directory
        self.source_fingerprint = source_fingerprint
        self.records: dict[str, dict] = {}

    def append(self, operation: str, arguments: dict, result: dict) -> dict:
        evidence_id = f"ev-{len(self.records) + 1}"
        record = {"evidence_id": evidence_id, "attempt_id": self.directory.name,
                  "source_fingerprint": self.source_fingerprint,
                  "created_at": datetime.now(timezone.utc).isoformat(), "operation": operation,
                  "arguments": arguments, "result": result}
        self.records[evidence_id] = record
        with (self.directory / "evidence.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {**result, "evidence_ids": [evidence_id]}

    def load(self, *, tolerate_partial: bool = False) -> None:
        path = self.directory / "evidence.jsonl"
        if not path.exists():
            return
        if path.stat().st_size > 32_000_000:
            raise ValueError("Evidence log exceeds size limit")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if tolerate_partial and not line.endswith("\n"):
                    break
                record = json.loads(line)
                if (not isinstance(record, dict) or not isinstance(record.get("result"), dict)
                        or not isinstance(record.get("arguments"), dict)
                        or not isinstance(record.get("evidence_id"), str)
                        or not isinstance(record.get("operation"), str)
                        or not isinstance(record.get("created_at"), str)):
                    raise ValueError("Malformed evidence record")
                if record.get("attempt_id") != self.directory.name:
                    raise ValueError("Evidence attempt mismatch")
                if self.source_fingerprint and record.get("source_fingerprint") != self.source_fingerprint:
                    raise ValueError("Evidence source mismatch")
                if record["evidence_id"] in self.records:
                    raise ValueError("Duplicate evidence ID")
                self.records[record["evidence_id"]] = record
