"""Persistence for codebase context indexes."""

from __future__ import annotations

import json
from pathlib import Path

from agent_runtime.codebase_context.models import CodebaseContextIndex
from config import CodeBaseConig

config = CodeBaseConig()

class CodebaseContextStore:
    def __init__(self, repo_path: str | Path, index_path: str = config.DEFAULT_INDEX_PATH) -> None:
        self.repo_path = Path(repo_path)
        self.index_path = self.repo_path / index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.index_path.exists()

    def load(self) -> CodebaseContextIndex:
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return CodebaseContextIndex.from_dict(data)

    def save(self, index: CodebaseContextIndex) -> Path:
        self.index_path.write_text(
            json.dumps(index.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.index_path

    def is_stale(self) -> bool:
        if not self.index_path.exists():
            return True
        index_mtime = self.index_path.stat().st_mtime
        for path in self.repo_path.rglob("*"):
            if not path.is_file() or _is_ignored(path, self.repo_path):
                continue
            if path.stat().st_mtime > index_mtime:
                return True
        return False


def _is_ignored(path: Path, repo_path: Path) -> bool:
    ignored_parts = {
        ".git",
        ".repomind",
        ".venv",
        "__pycache__",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "target",
        "coverage",
    }
    rel = path.relative_to(repo_path)
    return bool(set(rel.parts).intersection(ignored_parts))
