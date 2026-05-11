"""
Action contracts used by the agent runtime.

The policy layer emits Action objects; the executor layer only knows how to
dispatch them through ToolRegistry. This keeps future RL/LLM policies isolated
from tool implementation details.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


ActionName = Literal[
    "list_files",
    "search_code",
    "read_file",
    "run_tests",
    "git_diff",
    "write_memory",
    "finish",
]


@dataclass(frozen=True)
class Action:
    name: ActionName
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""

