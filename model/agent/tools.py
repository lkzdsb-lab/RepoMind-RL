from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isclass
from typing import Any, Callable, Dict, Iterable

from pydantic import BaseModel, ValidationError

ToolFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]
ToolReducer = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

# 工具类 model
@dataclass
class ToolSpec:
    name: str
    description: str
    runner: ToolFn
    input_schema: Any = field(default_factory=dict)
    """ 定义工具接收参数 """
    output_schema: Dict[str, Any] = field(default_factory=dict)
    """ 定义工具返回参数 """
    permissions: list[str] = field(default_factory=list)
    """ 权限 """
    metadata: Dict[str, Any] = field(default_factory=dict)
    reducer: ToolReducer | None = None
    """ 定义了工具的运行结果如何合并回全局状态（AgentState） """


STANDARD_RESULT_KEYS = {
    "ok",
    "status",
    "error",
    "message",
    "data",
    "artifacts",
}


def run_tool_spec(
    spec: ToolSpec,
    repo_path: str,
    args: Dict[str, Any] | None = None,
    *,
    allowed_permissions: Iterable[str] | None = None,
    runtime_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Validate user-facing args, then execute with trusted runtime context."""
    raw_args = args or {}
    permission_error = _permission_error(spec, allowed_permissions)
    if permission_error:
        return normalize_tool_result(permission_error, tool_name=spec.name)

    validated_args, validation_error = validate_tool_args(spec, raw_args)
    if validation_error:
        return normalize_tool_result(validation_error, tool_name=spec.name)

    execution_args = dict(validated_args)
    if runtime_context:
        execution_args["_runtime_context"] = dict(runtime_context)
    output = spec.runner(repo_path, execution_args)
    if not isinstance(output, dict):
        output = {"result": output}
    return normalize_tool_result(output, tool_name=spec.name)


def validate_tool_args(
    spec: ToolSpec,
    args: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    schema = spec.input_schema
    if not _is_pydantic_model(schema):
        return dict(args), None
    try:
        model = schema.model_validate(args)
    except ValidationError as exc:
        try:
            validation_errors = exc.errors(include_context=False)
        except TypeError:
            validation_errors = exc.errors()
        return {}, {
            "error": f"Invalid input for tool {spec.name}.",
            "validation_errors": validation_errors,
            "needs_more_context": True,
        }
    data = model.model_dump(exclude_none=True)
    extra = getattr(model, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        data.update(extra)
    return data, None


def normalize_tool_result(
    output: Dict[str, Any],
    *,
    tool_name: str = "",
) -> Dict[str, Any]:
    """
        统一 tool 的返回，避免 llm 出现幻觉
    """
    result = dict(output)
    status = str(result.get("status") or "").strip().lower()
    if not status:
        status = _infer_status(result)
    ok = bool(result.get("ok")) if "ok" in result else status in {"success", "skipped"}
    message = str(
        result.get("message")
        or result.get("summary")
        or result.get("error")
        or status
    ).strip()
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    if result.get("diff") and not any(item.get("type") == "diff" for item in artifacts if isinstance(item, dict)):
        artifacts.append(
            {
                "type": "diff",
                "name": f"{tool_name or 'tool'} diff",
                "content": result.get("diff"),
            }
        )
    data = result.get("data")
    if not isinstance(data, dict):
        data = {
            key: value
            for key, value in result.items()
            if key not in STANDARD_RESULT_KEYS
        }
    result.update(
        {
            "ok": ok,
            "status": status,
            "message": message,
            "data": data,
            "artifacts": artifacts,
        }
    )
    return result


def tool_spec_prompt_dict(spec: ToolSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": _schema_for_prompt(spec.input_schema),
        "permissions": list(spec.permissions),
    }


def _permission_error(
    spec: ToolSpec,
    allowed_permissions: Iterable[str] | None,
) -> Dict[str, Any]:
    if allowed_permissions is None:
        return {}
    required = set(spec.permissions)
    allowed = set(allowed_permissions)
    missing = sorted(required - allowed)
    if not missing:
        return {}
    return {
        "error": f"Permission denied for tool {spec.name}: missing {', '.join(missing)}.",
        "missing_permissions": missing,
        "required_permissions": sorted(required),
        "allowed_permissions": sorted(allowed),
        "fatal": True,
    }


def _infer_status(output: Dict[str, Any]) -> str:
    if output.get("needs_user_input"):
        return "needs_user_input"
    if output.get("needs_more_context"):
        return "needs_more_context"
    if output.get("skipped") or output.get("unsupported"):
        return "skipped"
    if output.get("error"):
        return "failed"
    exit_code = output.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return "failed"
    return "success"


def _is_pydantic_model(value: Any) -> bool:
    return bool(isclass(value) and issubclass(value, BaseModel))


def _schema_for_prompt(schema: Any) -> Dict[str, Any]:
    if _is_pydantic_model(schema):
        raw = schema.model_json_schema()
        prompt_schema = {
            "type": raw.get("type", "object"),
            "properties": raw.get("properties", {}),
            "required": raw.get("required", []),
        }
        if "$defs" in raw:
            prompt_schema["$defs"] = raw["$defs"]
        return prompt_schema
    if isinstance(schema, dict):
        return schema
    return {}
