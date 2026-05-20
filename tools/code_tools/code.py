import shutil
import subprocess
from typing import Dict, Any

def search_code(repo_path: str, query: str, max_results: int = 30) -> Dict[str, Any]:
    if not query:
        return {"query": query, "matches": [], "exit_code": 0}

    try:
        if shutil.which("rg"):
            command = ["rg", "-n", "--hidden", "--glob", "!.git", query, "."]
        else:
            command = ["grep", "-RIn", query, "."]

        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )

        lines = result.stdout.splitlines()[:max_results]
        return {
            "query": query,
            "matches": lines,
            "exit_code": result.returncode,
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {"query": query, "error": "search timeout"}
