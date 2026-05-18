from __future__ import annotations

from typing import Any, Mapping
from types import MappingProxyType

# 注册工具基类
class BaseRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, spec: Any) -> None:
        if not getattr(spec, "name", ""):
            raise ValueError("Registry spec must have a non-empty name.")
        self._items[spec.name] = spec

    def get(self, name: str) -> Any | None:
        return self._items.get(name)

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._items))