"""
Runtime registries for tools, nodes, prompts, and skills.

The manager keeps mutable registries while each agent run consumes an immutable
snapshot. That lets the process accept updates without changing behavior in the
middle of a run.
"""

from __future__ import annotations

import importlib
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from graph.register import NodeRegistry
from model.agent.tools import ToolSpec
from model.agent.node import NodeSpec
from model.skill import SkillSpec
from model.prompt import PromptSpec
from agent_runtime.tool_registry import ToolRegistry
from prompts.register import PromptRegistry
from skills.register import SkillRegistry

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


# 注册一个 graph 执行时的快照，frozen 用于保护执行过程中不被更改
@dataclass(frozen=True)
class RegistrySnapshot:
    tools: Mapping[str, ToolSpec]
    nodes: Mapping[str, NodeSpec]
    prompts: Mapping[str, PromptSpec]
    skills: Mapping[str, SkillSpec]

    def get_tool(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def get_node(self, name: str) -> NodeSpec | None:
        return self.nodes.get(name)

    def get_prompt(self, name: str) -> PromptSpec | None:
        return self.prompts.get(name)

    def get_skill(self, name: str) -> SkillSpec | None:
        return self.skills.get(name)

    def run_tool(
        self,
        name: str,
        repo_path: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = self.get_tool(name)
        if spec is None:
            return {"error": f"Unknown tool: {name}"}
        return spec.runner(repo_path, args or {})

    # 用 name 反射获取已经注册的元素集合
    def names(self, kind: str) -> list[str]:
        registry = getattr(self, kind, None)
        if isinstance(registry, Mapping):
            registry = MappingProxyType(registry)
        if registry is None:
            raise ValueError(f"Unknown registry kind: {kind}")
        return sorted(registry)


# 注册管理类
class RegistryManager:
    def __init__(
        self,
        tools: Any | None = None,
        nodes: NodeRegistry | None = None,
        prompts: PromptRegistry | None = None,
        skills: SkillRegistry | None = None,
        manifest_dir: str | Path | None = None,
    ) -> None:
        if tools is None:
            tools = ToolRegistry()

        self.tools = tools
        self.nodes = nodes or NodeRegistry()
        self.prompts = prompts or PromptRegistry()
        self.skills = skills or SkillRegistry()

        # 数据加载
        if manifest_dir is not None:
            self.load_manifests(manifest_dir)

    def load_manifests(self, manifest_dir: str | Path) -> None:
        ManifestLoader(self).load_dir(manifest_dir)

    def snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(
            tools=MappingProxyType(dict(self.tools.items())),
            nodes=MappingProxyType(dict(self.nodes.items())),
            prompts=MappingProxyType(dict(self.prompts.items())),
            skills=MappingProxyType(dict(self.skills.items())),
        )


class ManifestLoader:
    SUPPORTED_SUFFIXES = {".json", ".toml"}

    def __init__(self, manager: RegistryManager) -> None:
        self.manager = manager

    def load_dir(self, manifest_dir: str | Path) -> None:
        root = Path(manifest_dir)
        if not root.exists():
            raise FileNotFoundError(f"Manifest directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Manifest path is not a directory: {root}")

        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in self.SUPPORTED_SUFFIXES:
                self.load_file(path)

    # 从配置文件中加载 tools、skills 等数据
    def load_file(self, path: str | Path) -> None:
        manifest_path = Path(path)
        data = self._read_manifest(manifest_path)
        kind = str(data.get("kind", "")).lower()

        if kind == "tool":
            self.manager.tools.register(self._tool_spec(data))
        elif kind == "node":
            self.manager.nodes.register(self._node_spec(data))
        elif kind == "prompt":
            self.manager.prompts.register(self._prompt_spec(data, manifest_path.parent))
        elif kind == "skill":
            self.manager.skills.register(self._skill_spec(data))
        else:
            raise ValueError(f"Unsupported manifest kind `{kind}` in {manifest_path}")

    def _read_manifest(self, path: Path) -> dict[str, Any]:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".toml":
            text = path.read_text(encoding="utf-8")
            if tomllib is not None:
                return tomllib.loads(text)
            return _loads_simple_toml(text)
        raise ValueError(f"Unsupported manifest suffix: {path.suffix}")

    def _tool_spec(self, data: dict[str, Any]) -> ToolSpec:
        return ToolSpec(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            runner=_import_callable(str(data["runner"])),
            input_schema=dict(data.get("input_schema", {})),
            output_schema=dict(data.get("output_schema", {})),
            permissions=list(data.get("permissions", [])),
            metadata=dict(data.get("metadata", {})),
            reducer=_optional_callable(data.get("reducer")),
        )

    def _node_spec(self, data: dict[str, Any]) -> NodeSpec:
        return NodeSpec(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            handler=_import_callable(str(data["handler"])),
            input_schema=dict(data.get("input_schema", {})),
            output_schema=dict(data.get("output_schema", {})),
            dependencies=list(data.get("dependencies", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def _prompt_spec(self, data: dict[str, Any], base_dir: Path) -> PromptSpec:
        template = data.get("template")
        template_file = data.get("template_file")
        if template is None and template_file:
            template = (base_dir / str(template_file)).read_text(encoding="utf-8")
        if template is None:
            raise ValueError(f"Prompt `{data.get('name')}` must define template or template_file.")

        return PromptSpec(
            name=str(data["name"]),
            version=str(data.get("version", "0.1.0")),
            template=str(template),
            variables=list(data.get("variables", [])),
            model_policy=dict(data.get("model_policy", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def _skill_spec(self, data: dict[str, Any]) -> SkillSpec:
        return SkillSpec(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            version=str(data.get("version", "0.1.0")),
            triggers=list(data.get("triggers", [])),
            resources=list(data.get("resources", [])),
            entrypoints=dict(data.get("entrypoints", {})),
            metadata=dict(data.get("metadata", {})),
        )


def _optional_callable(value: Any) -> Callable[..., Any] | None:
    if not value:
        return None
    return _import_callable(str(value))


def _import_callable(path: str) -> Callable[..., Any]:
    module_name, separator, attr_name = path.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"Callable path must use `module:function` format: {path}")

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    if not callable(value):
        raise TypeError(f"Imported value is not callable: {path}")
    return value


def _loads_simple_toml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: dict[str, Any] = data

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[[") and line.endswith("]]"):
            path = line[2:-2].strip().split(".")
            parent = _ensure_toml_table(data, path[:-1])
            items = parent.setdefault(path[-1], [])
            if not isinstance(items, list):
                raise ValueError(f"TOML array table conflicts with existing key: {line}")
            current = {}
            items.append(current)
            continue

        if line.startswith("[") and line.endswith("]"):
            current = _ensure_toml_table(data, line[1:-1].strip().split("."))
            continue

        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Unsupported TOML line: {line}")
        current[key.strip()] = _parse_toml_value(value.strip())

    return data


def _ensure_toml_table(data: dict[str, Any], path: list[str]) -> dict[str, Any]:
    current = data
    for part in path:
        value = current.setdefault(part, {})
        if not isinstance(value, dict):
            raise ValueError(f"TOML table conflicts with existing key: {part}")
        current = value
    return current


def _parse_toml_value(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith(('"', "'", "[")):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
