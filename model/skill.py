from dataclasses import dataclass, field
from typing import Any

@dataclass
class SkillSpec:
    name: str
    description: str
    version: str = "0.1.0"
    triggers: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    entrypoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)