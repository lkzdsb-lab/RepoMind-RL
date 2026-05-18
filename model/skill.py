from dataclasses import dataclass, field
from typing import Any

@dataclass
class SkillSpec:
    name: str
    description: str
    version: str = "0.1.0"
    """ 版本控制 """
    triggers: list[str] = field(default_factory=list)
    """ 触发关键字 """
    resources: list[str] = field(default_factory=list)
    """ 声明访问该 skill 内部所需要的资源 """
    entrypoints: dict[str, str] = field(default_factory=dict)
    """ 指定逻辑入口，寻找怎么调用这个skill的方式 """
    metadata: dict[str, Any] = field(default_factory=dict)
    """ 扩展字段 """