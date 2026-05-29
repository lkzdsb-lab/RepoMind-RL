from typing import Dict, Any

from tools.shell_tools.command import run_shell_command


def run_command(repo_path: str, command: str = "pytest", timeout: int = 120) -> Dict[str, Any]:
    return run_shell_command(
        repo_path,
        {
            "command": command,
            "purpose": "verification",
            "timeout": timeout,
            "reason": "configured verification command",
            "allow_shell": False,
        },
    )
