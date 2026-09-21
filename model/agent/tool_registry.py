"""Tool registration and dispatch without application-specific dependencies."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from loguru import logger

from model.agent.tools import ToolSpec, normalize_tool_result, run_tool_spec


class ToolRegistryBase:
    """Shared registry for the main agent and restricted tool collections."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            logger.warning("overriding registered tool name={}", spec.name)
        else:
            logger.debug("registering tool name={}", spec.name)
        self._tools[spec.name] = spec

    def run(
        self,
        name: str,
        repo_path: str,
        args: dict[str, Any] | None = None,
        *,
        allowed_permissions: list[str] | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name not in self._tools:
            logger.warning("unknown tool requested name={}", name)
            return normalize_tool_result({"error": f"Unknown tool: {name}"}, tool_name=name)
        logger.debug("tool registry dispatch name={} repo_path={} args={}", name, repo_path, args or {})
        return run_tool_spec(
            self._tools[name], repo_path, args or {},
            allowed_permissions=allowed_permissions, runtime_context=runtime_context,
        )

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def items(self) -> Mapping[str, ToolSpec]:
        return MappingProxyType(dict(self._tools))
