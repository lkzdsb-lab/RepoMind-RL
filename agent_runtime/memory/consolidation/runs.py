"""Immutable, versioned records of consolidation inputs and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_runtime.memory.domain.models import ConsolidationResult
from agent_runtime.memory.io import (
    atomic_directory,
    canonical_json_bytes,
    safe_identifier,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    write_bytes,
)
from utils import utc_now


class ConsolidationRunStore:
    """ 沉淀 pipeline 记录"""
    def __init__(self, root: str | Path, *, repo_path: str | Path = ".") -> None:
        repo = Path(repo_path or ".").resolve()
        root_value = str(root or "").strip()
        if not root_value:
            raise ValueError("consolidation run root is required")
        configured = Path(root_value)
        self.root = configured.resolve() if configured.is_absolute() else (repo / configured).resolve()

    def run_path(self, task_id: str, pipeline_version: str) -> Path:
        return self.root / safe_identifier(task_id, field_name="task_id") / safe_identifier(
            pipeline_version, field_name="pipeline_version"
        )

    def load(self, task_id: str, pipeline_version: str) -> ConsolidationResult | None:
        path = self.run_path(task_id, pipeline_version)
        if not path.is_dir():
            return None
        errors = self.verify(task_id, pipeline_version)
        if errors:
            raise ValueError(f"invalid consolidation run: {'; '.join(errors)}")
        data = json.loads((path / "result.json").read_text(encoding="utf-8"))
        return ConsolidationResult.from_dict(data)

    def save(
        self,
        outcome: ConsolidationResult,
        artifacts: dict[str, Any],
    ) -> Path:
        """  save 一次离线处理的流水记录"""
        destination = self.run_path(outcome.task_id, outcome.pipeline_version)
        if destination.exists():
            raise FileExistsError(f"consolidation run already exists: {destination}")
        payloads = {
            f"{name}.json": canonical_json_bytes(value)
            for name, value in artifacts.items()
        }
        payloads["result.json"] = canonical_json_bytes(outcome.to_dict())
        with atomic_directory(destination) as temporary:
            records: dict[str, Any] = {}
            for name, content in sorted(payloads.items()):
                write_bytes(temporary / name, content)
                records[name] = {
                    "sha256": sha256_bytes(content),
                    "size_bytes": len(content),
                }
            write_bytes(
                temporary / "manifest.json",
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "task_id": outcome.task_id,
                        "pipeline_version": outcome.pipeline_version,
                        "created_at": utc_now(),
                        "files": records,
                    }
                ),
            )
        return destination

    def read_artifact(
        self,
        task_id: str,
        pipeline_version: str,
        name: str,
    ) -> object:
        safe_name = safe_identifier(name, field_name="artifact name")
        path = self.run_path(task_id, pipeline_version) / f"{safe_name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def verify(self, task_id: str, pipeline_version: str) -> list[str]:
        path = self.run_path(task_id, pipeline_version)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            return ["manifest.json is missing"]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [f"manifest.json is invalid: {exc}"]
        errors: list[str] = []
        if str(manifest.get("task_id") or "") != task_id:
            errors.append("manifest task_id does not match directory")
        if str(manifest.get("pipeline_version") or "") != pipeline_version:
            errors.append("manifest pipeline_version does not match directory")
        files = dict(manifest.get("files") or {})
        if "result.json" not in files:
            errors.append("result.json is not declared")
        for name, record in files.items():
            if not isinstance(record, dict):
                errors.append(f"{name} has an invalid record")
                continue
            try:
                artifact = safe_relative_path(path, str(name))
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
                continue
            if not artifact.is_file():
                errors.append(f"{name} is missing")
                continue
            if artifact.stat().st_size != int(record.get("size_bytes") or -1):
                errors.append(f"{name} size mismatch")
            if sha256_file(artifact) != str(record.get("sha256") or ""):
                errors.append(f"{name} sha256 mismatch")
        return errors
