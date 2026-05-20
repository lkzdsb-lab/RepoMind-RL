import subprocess
from typing import Dict, Any


def git_diff(repo_path: str) -> Dict[str, Any]:
    result = subprocess.run(
        ["git", "diff"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )

    return {
        "exit_code": result.returncode,
        "diff": result.stdout[-12000:],
        "stderr": result.stderr,
    }
