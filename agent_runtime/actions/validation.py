"""Validate LLM action arguments against the selected tool contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import isclass
from typing import Any

from model.agent.tools import ToolSpec
from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ActionArgsResult:
    valid: bool
    args: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    ignored_fields: list[str] = field(default_factory=list)
    schema_applied: bool = False


class ActionArgumentValidator:
    def __init__(self, tool_specs: Mapping[str, ToolSpec] | None = None) -> None:
        self._tool_specs: Mapping[str, ToolSpec] = tool_specs or {}

    def set_tool_specs(self, tool_specs: Mapping[str, ToolSpec]) -> None:
        self._tool_specs = tool_specs

    def validate(self, action_name: str, raw_args: dict[str, Any]) -> ActionArgsResult:
        if action_name == "finish":
            return ActionArgsResult(
                valid=True,
                ignored_fields=sorted(str(key) for key in raw_args),
                schema_applied=True,
            )

        spec = self._tool_specs.get(action_name)
        schema = spec.input_schema if spec is not None else None
        if not isclass(schema) or not issubclass(schema, BaseModel):
            return ActionArgsResult(valid=True, args=dict(raw_args))

        allowed = set(schema.model_fields)
        projected = {
            str(key): value
            for key, value in raw_args.items()
            if str(key) in allowed
        }
        ignored = sorted(str(key) for key in raw_args if str(key) not in allowed)
        try:
            model = schema.model_validate(projected)
        except ValidationError as exc:
            return ActionArgsResult(
                valid=False,
                errors=[
                    {
                        "field": ".".join(str(part) for part in error.get("loc", ())),
                        "reason": str(error.get("msg") or "invalid value"),
                        "type": str(error.get("type") or "validation_error"),
                    }
                    for error in exc.errors(include_context=False)
                ],
                ignored_fields=ignored,
                schema_applied=True,
            )
        return ActionArgsResult(
            valid=True,
            args=model.model_dump(exclude_none=True),
            ignored_fields=ignored,
            schema_applied=True,
        )
