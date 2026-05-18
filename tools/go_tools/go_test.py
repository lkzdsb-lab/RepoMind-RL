import subprocess
import shlex
from typing import Dict, Any


def run_command(repo_path: str, command: str = "pytest", timeout: int = 120) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
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
            "stderr": "command timeout",
        }
    except ValueError as exc:
        return {
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
        }
