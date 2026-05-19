"""
Action contracts used by the agent runtime.

The policy layer emits Action objects; the executor layer only knows how to
dispatch them through ToolRegistry. This keeps future RL/LLM policies isolated
from tool implementation details.
"""
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class Action:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

# 每个动作的语义
@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
