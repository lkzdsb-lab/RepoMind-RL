import os
from pathlib import Path
from typing import Dict, Any, List

def _safe_path(repo_path: str, file_path: str) -> Path:
    repo = Path(repo_path).resolve()
    target = (repo / file_path).resolve()
    if target != repo and repo not in target.parents:
        raise ValueError(f"Path escapes repository: {file_path}")
    return target

# 查看文件列表
def list_files(repo_path: str, max_files: int = 200) -> Dict[str, Any]:
    files: List[str] = []
    ignored = {".git", "vendor", "node_modules", "__pycache__"}

    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored]

        for filename in filenames:
            path = Path(root) / filename
            rel = path.relative_to(repo_path).as_posix()
            files.append(rel)

            if len(files) >= max_files:
                return {"files": files, "truncated": True}

    return {"files": files, "truncated": False}

# 读取文件
def read_file(repo_path: str, file_path: str, max_chars: int = 8000) -> Dict[str, Any]:
    try:
        target = _safe_path(repo_path, file_path)
    except ValueError as exc:
        return {"error": str(exc)}

    if not target.exists():
        return {"error": f"File not found: {file_path}"}
    if not target.is_file():
        return {"error": f"Not a file: {file_path}"}

    content = target.read_text(encoding="utf-8", errors="ignore")
    return {
        "file_path": file_path,
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }
