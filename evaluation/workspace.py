from __future__ import annotations

import difflib
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


IGNORED_PARTS = {
    ".git",
    ".idea",
    ".repomind",
    ".venv",
    "__pycache__",
    "node_modules",
    "vendor",
}


@dataclass(frozen=True)
class FileSnapshot:
    digest: str
    size: int
    text: str | None


def copy_fixture(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise NotADirectoryError(f"Evaluation fixture does not exist: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=_copy_ignore)


def snapshot_workspace(root: Path) -> dict[str, FileSnapshot]:
    result: dict[str, FileSnapshot] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _ignored(path.relative_to(root)):
            continue
        data = path.read_bytes()
        result[path.relative_to(root).as_posix()] = FileSnapshot(
            digest=hashlib.sha256(data).hexdigest(),
            size=len(data),
            text=_decode_text(data),
        )
    return result


def snapshot_digest(snapshot: dict[str, FileSnapshot]) -> str:
    digest = hashlib.sha256()
    for path, item in sorted(snapshot.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def changed_files(
    before: dict[str, FileSnapshot],
    after: dict[str, FileSnapshot],
) -> list[str]:
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


def unified_workspace_diff(
    before: dict[str, FileSnapshot],
    after: dict[str, FileSnapshot],
) -> str:
    chunks: list[str] = []
    for path in changed_files(before, after):
        old = before.get(path)
        new = after.get(path)
        if (old and old.text is None) or (new and new.text is None):
            chunks.append(f"Binary file changed: {path}\n")
            continue
        old_lines = (old.text if old else "").splitlines(keepends=True)
        new_lines = (new.text if new else "").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}" if old else "/dev/null",
                tofile=f"b/{path}" if new else "/dev/null",
            )
        )
    return "".join(chunks)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_PARTS}


def _ignored(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts)


def _decode_text(data: bytes) -> str | None:
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None
