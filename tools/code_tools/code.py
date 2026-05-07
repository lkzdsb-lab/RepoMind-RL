import subprocess
from typing import Dict, Any

def search_code(repo_path: str, query: str, max_results: int = 30) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["grep", "-RIn", query, "."],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=20,
        )

        lines = result.stdout.splitlines()[:max_results]
        return {
            "query": query,
            "matches": lines,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"query": query, "error": "search timeout"}