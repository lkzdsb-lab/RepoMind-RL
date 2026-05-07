import os
from pathlib import Path
from typing import Dict, Any, List

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
    target = Path(repo_path) / file_path

    if not target.exists():
        return {"error": f"File not found: {file_path}"}

    content = target.read_text(encoding="utf-8", errors="ignore")
    return {
        "file_path": file_path,
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }
