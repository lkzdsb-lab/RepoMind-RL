"""Guarded repository editing tools."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, Union
from loguru import logger
from utils import _safe_float, _clean_string_list, _clamp_float


DENIED_PATH_PARTS = {
    ".git",
    ".repomind",
    "__pycache__",
    "node_modules",
    "vendor",
}


def apply_code_patch(repo_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply minimal exact-replacement edits.

    This tool deliberately does not accept arbitrary shell commands or raw patch
    files. The executor injects `_guard`; LLM-provided permission fields are not
    trusted.
    """
    # 获取注入的 安全约束
    guard = args.get("_guard")
    if not isinstance(guard, dict):
        guard = {}
        logger.info(
            "apply_code_patch requires a non-empty guard dictionary.",
        )
    if not bool(guard.get("editing_enabled", False)):
        return {
            "error": "Editing is disabled. Enable editing before using apply_code_patch.",
            "applied": False,
        }

    confidence_threshold = _safe_float(guard.get("confidence_threshold"), 0.75)
    confidence = _clamp_float(args.get("confidence"), 0.5, "apply_code_patch invalid confidence")
    questions = _clean_questions(args.get("uncertainty_questions"))
    if questions:
        return _needs_user_input(
            "The proposed edit contains unresolved uncertainty.",
            questions,
        )
    if confidence < confidence_threshold:
        return _needs_user_input(
            f"Edit confidence {confidence:.2f} is below threshold {confidence_threshold:.2f}.",
            [
                "请确认这次修改的目标行为、约束或验收标准，以便安全改代码。",
            ],
        )

    changes = args.get("changes")
    if not isinstance(changes, list) or not changes:
        return {"error": "apply_code_patch requires a non-empty changes list.", "applied": False}

    max_files = max(1, int(_safe_float(guard.get("max_files"), 5)))
    max_changed_lines = max(1, int(_safe_float(guard.get("max_changed_lines"), 300)))
    max_file_bytes = max(1, int(_safe_float(guard.get("max_file_bytes"), 200000)))
    require_read = bool(guard.get("require_read_before_write", True))
    allow_create = bool(guard.get("allow_create", False))
    allowed_files = {
        str(item).strip()
        for item in guard.get("allowed_files", [])
        if str(item).strip()
    }
    read_contents = guard.get("read_contents")
    if not isinstance(read_contents, dict):
        read_contents = {}

    unique_files = _unique_file_paths(changes)
    if len(unique_files) > max_files:
        return {
            "error": f"apply_code_patch can edit at most {max_files} files per call.",
            "applied": False,
            "changed_files": unique_files,
        }

    repo = Path(repo_path).resolve()
    planned_contents: dict[str, str] = {}
    original_contents: dict[str, str] = {}
    changed_files: list[str] = []

    for index, raw_change in enumerate(changes):
        if not isinstance(raw_change, dict):
            return {"error": f"Change #{index + 1} must be an object.", "applied": False}
        error = _validate_change_shape(raw_change, index)
        if error:
            return {"error": error, "applied": False}

        file_path = str(raw_change.get("file_path") or "").strip()
        operation = str(raw_change.get("operation") or "replace").strip().lower()
        if Path(file_path).is_absolute():
            return {"error": f"Absolute paths are not allowed: {file_path}", "applied": False}
        if _is_denied_path(file_path):
            return {"error": f"Editing protected path is not allowed: {file_path}", "applied": False}
        if require_read and operation != "create" and file_path not in allowed_files:
            return {
                "error": f"File must be read in this run before editing: {file_path}",
                "applied": False,
                "needs_more_context": True,
                "suggested_next_action": "read_file",
            }

        try:
            target = _safe_path(repo, file_path)
        except ValueError as exc:
            return {"error": str(exc), "applied": False}

        if operation == "create":
            if not allow_create:
                return {"error": "Creating files is disabled.", "applied": False}
            if target.exists():
                return {"error": f"Cannot create existing file: {file_path}", "applied": False}
            original = ""
            current = planned_contents.get(file_path, "")
            new_text = str(raw_change.get("new_text") or "")
            planned_contents[file_path] = current + new_text
            original_contents.setdefault(file_path, original)
        elif operation in {"replace", "append", "insert_after", "insert_before"}:
            original = original_contents.get(file_path)
            if original is None:
                loaded = _load_text_file(target, max_file_bytes, file_path)
                if loaded.get("error"):
                    return {"error": loaded["error"], "applied": False}
                original = str(loaded["content"])
                original_contents[file_path] = original
            current = planned_contents.get(file_path, original)
            old_text = str(raw_change.get("old_text") or "")
            new_text = str(raw_change.get("new_text") or "")
            read_content = str(read_contents.get(file_path) or "")
            if operation == "append":
                if require_read and file_path not in allowed_files:
                    return {
                        "error": f"File must be read in this run before editing: {file_path}",
                        "applied": False,
                        "needs_more_context": True,
                        "suggested_next_action": "read_file",
                    }
                planned_contents[file_path] = current + new_text
                if file_path not in changed_files:
                    changed_files.append(file_path)
                continue
            if operation == "replace" and old_text == "":
                if current != "":
                    return {
                        "error": (
                            "Empty old_text is only allowed when replacing the full content "
                            f"of an empty file: {file_path}"
                        ),
                        "applied": False,
                        "needs_more_context": True,
                        "suggested_next_action": "read_file",
                    }
                if require_read and file_path not in allowed_files:
                    return {
                        "error": f"File must be read in this run before editing: {file_path}",
                        "applied": False,
                        "needs_more_context": True,
                        "suggested_next_action": "read_file",
                    }
                planned_contents[file_path] = new_text
                if file_path not in changed_files:
                    changed_files.append(file_path)
                continue
            if operation in {"insert_after", "insert_before"}:
                if require_read and old_text not in read_content:
                    logger.warning(
                        f"old_text not in read_content \n"
                        f"old_text: {old_text}\n"
                        f"read_content: {read_content}"
                    )
                    return {
                        "error": (
                            "Anchor old_text must come from content read during this run: "
                            f"{file_path}"
                        ),
                        "applied": False,
                        "needs_more_context": True,
                        "suggested_next_action": "read_file",
                        "conflict_context": _conflict_context(read_content or current, old_text),
                    }
                expected_or_err = _validate_occurrences(raw_change, current, old_text, file_path)
                if isinstance(expected_or_err, dict):
                    return expected_or_err
                expected = expected_or_err
                replacement = (
                    old_text + new_text if operation == "insert_after" else new_text + old_text
                )
                planned_contents[file_path] = current.replace(old_text, replacement, expected)
                if file_path not in changed_files:
                    changed_files.append(file_path)
                continue
            if require_read and old_text not in read_content:
                return {
                    "error": (
                        "old_text must come from content read during this run: "
                        f"{file_path}"
                    ),
                    "applied": False,
                    "needs_more_context": True,
                    "suggested_next_action": "read_file",
                    "conflict_context": _conflict_context(read_content or current, old_text),
                }
            expected_or_err = _validate_occurrences(raw_change, current, old_text, file_path)
            if isinstance(expected_or_err, dict):
                return expected_or_err
            expected = expected_or_err
            planned_contents[file_path] = current.replace(old_text, new_text, expected)
        else:
            return {"error": f"Unsupported edit operation: {operation}", "applied": False}

        if file_path not in changed_files:
            changed_files.append(file_path)

    diff = _combined_diff(original_contents, planned_contents)
    if not diff:
        return {"error": "Patch produced no changes.", "applied": False}
    changed_line_count = _changed_line_count(diff)
    if changed_line_count > max_changed_lines:
        return {
            "error": (
                f"Edit changes {changed_line_count} lines, exceeding limit "
                f"{max_changed_lines}."
            ),
            "applied": False,
            "changed_files": changed_files,
            "diff": diff[-12000:],
        }

    dry_run = bool(args.get("dry_run", False))
    if not dry_run:
        for file_path, content in planned_contents.items():
            target = _safe_path(repo, file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    return {
        "applied": not dry_run,
        "dry_run": dry_run,
        "preview": dry_run,
        "would_apply": True,
        "changed_files": changed_files,
        "updated_contents": {
            file_path: planned_contents[file_path]
            for file_path in changed_files
            if file_path in planned_contents
        },
        "change_count": len(changes),
        "changed_line_count": changed_line_count,
        "reason": str(args.get("reason") or "").strip(),
        "assumptions": _clean_string_list(args.get("assumptions"), -1, 500),
        "diff": diff[-12000:],
        "summary": (
            f"{'Prepared' if dry_run else 'Applied'} {len(changes)} change(s) "
            f"across {len(changed_files)} file(s)."
        ),
    }


def _validate_change_shape(change: dict[str, Any], index: int) -> str:
    file_path = str(change.get("file_path") or "").strip()
    if not file_path:
        return f"Change #{index + 1} is missing file_path."
    operation = str(change.get("operation") or "replace").strip().lower()
    if operation == "replace":
        if "new_text" not in change:
            return f"Change #{index + 1} is missing new_text."
    elif operation == "append":
        if "new_text" not in change:
            return f"Change #{index + 1} is missing new_text."
    elif operation in {"insert_after", "insert_before"}:
        if not str(change.get("old_text") or ""):
            return f"Change #{index + 1} is missing old_text anchor."
        if "new_text" not in change:
            return f"Change #{index + 1} is missing new_text."
    elif operation == "create":
        if "new_text" not in change:
            return f"Change #{index + 1} is missing new_text."
    else:
        return f"Unsupported edit operation: {operation}"
    return ""


def _safe_path(repo: Path, file_path: str) -> Path:
    target = (repo / file_path).resolve()
    if target != repo and repo not in target.parents:
        raise ValueError(f"Path escapes repository: {file_path}")
    return target


def _is_denied_path(file_path: str) -> bool:
    return any(part in DENIED_PATH_PARTS for part in Path(file_path).parts)


def _load_text_file(target: Path, max_file_bytes: int, file_path: str) -> dict[str, Any]:
    if not target.exists():
        return {"error": f"File not found: {file_path}"}
    if not target.is_file():
        return {"error": f"Not a file: {file_path}"}
    raw = target.read_bytes()
    if len(raw) > max_file_bytes:
        return {
            "error": f"File is too large to edit safely: {file_path} ({len(raw)} bytes)"
        }
    if b"\0" in raw:
        return {"error": f"Binary files are not supported: {file_path}"}
    try:
        return {"content": raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {"error": f"File is not valid UTF-8: {file_path}"}


def _combined_diff(original: dict[str, str], planned: dict[str, str]) -> str:
    chunks: list[str] = []
    for file_path in sorted(planned):
        before = original.get(file_path, "")
        after = planned[file_path]
        if before == after:
            continue
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
    return "".join(chunks)


def _changed_line_count(diff: str) -> int:
    return sum(
        1
        for line in diff.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def _conflict_context(content: str, old_text: str, *, radius: int = 3) -> dict[str, Any]:
    if not content or not old_text:
        return {"reason": "empty content or old_text", "candidates": []}
    lines = content.splitlines()
    old_lines = [line for line in old_text.splitlines() if line.strip()]
    anchors = old_lines[:3] or [old_text.strip()]
    candidates: list[dict[str, Any]] = []
    for anchor in anchors:
        for index, line in enumerate(lines):
            if anchor and anchor in line:
                start = max(0, index - radius)
                end = min(len(lines), index + radius + 1)
                candidates.append(
                    {
                        "line": index + 1,
                        "anchor": anchor[:200],
                        "snippet": "\n".join(lines[start:end])[:2000],
                    }
                )
                break
        if candidates:
            break
    if not candidates:
        close = difflib.get_close_matches(anchors[0], lines, n=3, cutoff=0.45)
        for line in close:
            index = lines.index(line)
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            candidates.append(
                {
                    "line": index + 1,
                    "anchor": anchors[0][:200],
                    "snippet": "\n".join(lines[start:end])[:2000],
                }
            )
    return {
        "reason": "old_text did not match the current file content exactly.",
        "candidates": candidates[:3],
    }


def _unique_file_paths(changes: list[Any]) -> list[str]:
    paths: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        path = str(change.get("file_path") or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _clean_questions(value: Any) -> list[str]:
    questions = _clean_string_list(value, -1, 500)
    return [item for item in questions if item][:3]


def _needs_user_input(reason: str, questions: list[str]) -> dict[str, Any]:
    return {
        "needs_user_input": True,
        "applied": False,
        "reason": reason,
        "questions": questions[:3],
    }


def _validate_occurrences(raw_change: Dict[str, Any], current: str, old_text: str, file_path: str) -> \
Union[int, Dict[str, Any]]:
    """ 校验文本匹配次数"""
    expected = int(_safe_float(raw_change.get("expected_occurrences"), 1))
    if expected <= 0:
        logger.error(f"Expected occurrences should be a positive integer, found {expected}")
        return {
            "error": "expected_occurrences must be greater than 0.",
            "applied": False,
        }
    occurrences = current.count(old_text)
    if occurrences != expected:
        logger.error(f"文本匹配次数不同")
        return {
            "error": (
                f"Expected anchor old_text to occur {expected} time(s) in {file_path}, "
                f"found {occurrences}."
            ),
            "applied": False,
            "needs_more_context": True,
            "suggested_next_action": "read_file",
            "conflict_context": _conflict_context(current, old_text),
        }
    return expected
