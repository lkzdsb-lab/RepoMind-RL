import subprocess
from typing import Dict, Any


def run_go_test(repo_path: str, command: str = "go test ./...") -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command.split(),
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
        )

        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout[-6000:],
            "stderr": result.stderr[-6000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": "go test timeout",
        }