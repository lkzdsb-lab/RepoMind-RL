import os
from pathlib import Path
from typing import Any, Dict, List
from config import FileConfig

def _safe_path(repo_path: str, file_path: str) -> Path:
    repo = Path(repo_path).resolve()
    target = (repo / file_path).resolve()
    if target != repo and repo not in target.parents:
        raise ValueError(f"Path escapes repository: {file_path}")
    return target

# 查看文件列表
def list_files(repo_path: str, max_files: int = FileConfig.MAX_LIST_FILE_LIMIT) -> Dict[str, Any]:
    repo = Path(repo_path).resolve()
    files: List[str] = []
    ignored = set(FileConfig.LIST_IGNORED_DIRS)

    for root, dirs, filenames in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in ignored]

        for filename in filenames:
            path = Path(root) / filename
            rel = path.relative_to(repo).as_posix()
            files.append(rel)

            if len(files) >= max_files:
                return {"files": files, "truncated": True, "ignored_dirs": sorted(ignored)}

    return {"files": files, "truncated": False, "ignored_dirs": sorted(ignored)}

# 读取文件
def read_file(
    repo_path: str,
    file_path: str,
    max_chars: int = FileConfig.MAX_READ_FILE_LIMIT,
    start_line: int | None = None,
    end_line: int | None = None,
) -> Dict[str, Any]:
    try:
        target = _safe_path(repo_path, file_path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not target.exists():
        return {"error": f"File not found: {file_path}"}
    if not target.is_file():
        return {"error": f"Not a file: {file_path}"}

    if start_line is not None and start_line < 1:
        return {"error": "start_line must be greater than 0."}
    if end_line is not None and end_line < 1:
        return {"error": "end_line must be greater than 0."}
    if start_line is not None and end_line is not None and end_line < start_line:
        return {"error": "end_line must be greater than or equal to start_line."}

    content = target.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    selected_start = start_line or 1
    selected_end = end_line or total_lines or 1
    if start_line is not None or end_line is not None:
        start_index = max(0, selected_start - 1)
        end_index = min(total_lines, selected_end)
        selected = "".join(lines[start_index:end_index])
    else:
        selected = content
        selected_end = total_lines

    truncated = len(selected) > max_chars
    selected = selected[:max_chars]
    return {
        "file_path": file_path,
        "content": selected,
        "truncated": truncated,
        "start_line": selected_start,
        "end_line": selected_end,
        "total_lines": total_lines,
        "line_range_requested": start_line is not None or end_line is not None,
    }
