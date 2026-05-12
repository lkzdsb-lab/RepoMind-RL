from dataclasses import dataclass, field
from typing import Any

@dataclass
class PromptSpec:
    name: str
    template: str
    version: str = "0.1.0"
    variables: list[str] = field(default_factory=list)
    model_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)