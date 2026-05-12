from dataclasses import dataclass, field
from typing import Any, Callable

NodeHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class NodeSpec:
    name: str
    description: str
    handler: NodeHandler
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)