import subprocess
from typing import Dict, Any


def git_diff(repo_path: str) -> Dict[str, Any]:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
        return {
            "exit_code": 0,
            "diff": "",
            "stderr": probe.stderr,
            "skipped": True,
            "unsupported": True,
            "reason": "not_git_repo",
            "message": "Target directory is not a Git repository; git diff was skipped.",
        }

    result = subprocess.run(
        ["git", "diff"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=20,
    )

    return {
        "exit_code": result.returncode,
        "diff": result.stdout[-12000:],
        "stderr": result.stderr,
    }
