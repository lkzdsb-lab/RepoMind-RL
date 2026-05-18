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
    """ 定义工具接收参数 """
    output_schema: Dict[str, Any] = field(default_factory=dict)
    """ 定义工具返回参数 """
    permissions: list[str] = field(default_factory=list)
    """ 权限 """
    metadata: Dict[str, Any] = field(default_factory=dict)
    reducer: ToolReducer | None = None
    """ 定义了工具的运行结果如何合并回全局状态（AgentState） """
