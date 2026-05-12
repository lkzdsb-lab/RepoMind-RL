from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

ToolFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]
ToolReducer = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

# 工具类 model
@dataclass
class ToolSpec:
    name: str
    description: str
    runner: ToolFn
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reducer: ToolReducer | None = None
