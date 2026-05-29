"""Primitive text search tool backed by ripgrep or a Python fallback."""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict
from utils import _clean_string_list


IGNORED_DIRS = {
    ".git",
    ".repomind",
    ".venv",
    "__pycache__",
    "node_modules",
    "vendor",
}
SEARCH_MAX_CHAR = 200


def search_text(repo_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = str(args.get("pattern") or args.get("query") or "").strip()
    if not pattern:
        return {"error": "search_text requires pattern.", "matches": []}

    regex = bool(args.get("regex", True))
    context_lines = max(0, min(int(_safe_int(args.get("context_lines"), 0)), 5))
    max_results = max(1, min(int(_safe_int(args.get("max_results"), 50)), 200))
    globs = _clean_string_list(args.get("globs"), limit=12, max_chars=SEARCH_MAX_CHAR)

    repo = Path(repo_path).resolve()
    if not repo.exists() or not repo.is_dir():
        return {"error": f"Repository path is not a directory: {repo_path}", "matches": []}

    if not shutil.which("rg"):
        return _python_search(
            repo,
            pattern,
            regex=regex,
            globs=globs,
            context_lines=context_lines,
            max_results=max_results,
        )

    try:
        command = ["rg", "-n", "--hidden", "--glob", "!.git"]
        if not regex:
            command.append("--fixed-strings")
        if context_lines:
            command.extend(["-C", str(context_lines)])
        for item in globs:
            command.extend(["--glob", item])
        command.extend([pattern, "."])

        result = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=max(5, min(int(_safe_int(args.get("timeout"), 20)), 120)),
        )
    except subprocess.TimeoutExpired:
        return {
            "pattern": pattern,
            "regex": regex,
            "matches": [],
            "error": "search timeout",
        }

    lines = result.stdout.splitlines()[:max_results]
    output = {
        "pattern": pattern,
        "regex": regex,
        "globs": globs,
        "context_lines": context_lines,
        "matches": [_parse_match(line) for line in lines],
        "raw_matches": lines,
        "truncated": len(result.stdout.splitlines()) > max_results,
        "exit_code": result.returncode,
        "command": shlex.join(command),
        "engine": "rg",
        "status": "success",
    }
    if result.returncode not in (0, 1):
        output["status"] = "failed"
        output["error"] = result.stderr.strip() or f"rg exited with {result.returncode}"
    return output


def _python_search(
    repo: Path,
    pattern: str,
    *,
    regex: bool,
    globs: list[str],
    context_lines: int,
    max_results: int,
) -> Dict[str, Any]:
    try:
        compiled = re.compile(pattern) if regex else None
    except re.error as exc:
        return {
            "pattern": pattern,
            "regex": regex,
            "matches": [],
            "error": f"Invalid regex: {exc}",
            "engine": "python",
        }

    matches: list[dict[str, Any]] = []
    raw_matches: list[str] = []
    scanned_files = 0
    skipped_files = 0
    for path in _iter_search_files(repo):
        if len(matches) >= max_results:
            break
        rel = path.relative_to(repo).as_posix()
        if globs and not _matches_globs(rel, globs):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            skipped_files += 1
            continue
        scanned_files += 1
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            if _line_matches(line, pattern, regex=regex, compiled=compiled):
                item = {
                    "file_path": rel,
                    "line": index,
                    "text": line,
                }
                if context_lines:
                    before_start = max(0, index - 1 - context_lines)
                    after_end = min(len(lines), index + context_lines)
                    item["before"] = lines[before_start:index - 1]
                    item["after"] = lines[index:after_end]
                matches.append(item)
                raw_matches.append(f"{rel}:{index}:{line}")
                if len(matches) >= max_results:
                    break

    return {
        "pattern": pattern,
        "regex": regex,
        "globs": globs,
        "context_lines": context_lines,
        "matches": matches,
        "raw_matches": raw_matches,
        "truncated": len(matches) >= max_results,
        "exit_code": 0 if matches else 1,
        "command": "",
        "engine": "python",
        "status": "success",
        "warning": "ripgrep is not installed; used Python search fallback.",
        "scanned_files": scanned_files,
        "skipped_files": skipped_files,
    }


def _parse_match(line: str) -> dict[str, Any]:
    parts = line.split(":", 2)
    if len(parts) >= 3 and parts[1].isdigit():
        path = parts[0][2:] if parts[0].startswith("./") else parts[0]
        return {"file_path": path, "line": int(parts[1]), "text": parts[2]}
    return {"text": line}


def _line_matches(
    line: str,
    pattern: str,
    *,
    regex: bool,
    compiled: re.Pattern[str] | None,
) -> bool:
    if regex:
        return bool(compiled and compiled.search(line))
    return pattern in line


def _matches_globs(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, item) or fnmatch.fnmatch(Path(path).name, item) for item in globs)


def _iter_search_files(repo: Path):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [item for item in dirs if item not in IGNORED_DIRS]
        for filename in files:
            yield Path(root) / filename


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
