"""Typed runtime events for context handling.

The event layer keeps prompt context work away from raw trajectory/tool-call
JSON. It is intentionally derived from AgentState so existing trace/state
formats remain compatible.
"""

from __future__ import annotations

from typing import Any

from model.agent.graph import AgentState
from model.agent.compress import ContextEvent, EventType, Importance, Retention
from utils import _truncate


def collect_context_events(state: AgentState) -> list[ContextEvent]:
    """
        从一个 context 提取相关的 event 类型信息
    """
    events: list[ContextEvent] = []
    title = str(state.get("title") or "").strip()
    description = str(state.get("description") or "").strip()
    loop_count = int(state.get("loop_count", 0))
    if title or description:
        events.append(
            ContextEvent(
                event_id="task:current",
                event_type="task_event",
                source="task",
                summary=_join_non_empty([title, description], " "),
                payload={
                    "title": title,
                    "description": description,
                    "verification_required": bool(state.get("verification_required", True)),
                    "verification_reason": state.get("verification_reason", ""),
                },
                importance="critical",
                retention="working",
                raw_ref={"state": "title/description"},
                loop_count=loop_count,
            )
        )

    if state.get("task_analysis"):
        events.append(
            ContextEvent(
                event_id="task:analysis",
                event_type="task_event",
                source="task_analysis",
                summary=_task_analysis_summary(state.get("task_analysis", {})),
                payload=_small_dict(state.get("task_analysis", {}), 20),
                importance="high",
                retention="working",
                raw_ref={"state": "task_analysis"},
                loop_count=loop_count,
            )
        )

    for index, item in enumerate(state.get("user_inputs", []) or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("answer") or item.get("content") or item.get("input") or "").strip()
        if not text:
            continue
        events.append(
            ContextEvent(
                event_id=f"user_input:{index}",
                event_type="user_event",
                source="user_input",
                summary=f"User added context: {_truncate(text, 260)}",
                payload={"text": text},
                importance="critical",
                retention="working",
                raw_ref={"state": "user_inputs", "index": index},
                loop_count=int(item.get("loop_count") or loop_count),
            )
        )

    for index, item in enumerate(state.get("plan_mode_events", []) or []):
        if not isinstance(item, dict):
            continue
        events.append(
            ContextEvent(
                event_id=f"plan_event:{index}",
                event_type="plan_event",
                source=str(item.get("source") or item.get("type") or "plan_mode"),
                summary=_plan_event_summary(item),
                payload=_small_dict(item, 16),
                importance="high",
                retention="working",
                raw_ref={"state": "plan_mode_events", "index": index},
                loop_count=int(item.get("loop_count") or loop_count),
            )
        )

    if state.get("debug_technical_plan"):
        events.append(
            ContextEvent(
                event_id="plan:technical",
                event_type="plan_event",
                source="debug_technical_plan",
                summary=_truncate(str(state.get("debug_technical_plan") or ""), 500),
                payload={
                    "approved": bool(state.get("plan_mode_approved", False)),
                    "evaluation": _truncate(str(state.get("plan_mode_evaluation") or ""), 600),
                },
                importance="high",
                retention="working",
                raw_ref={"state": "debug_technical_plan"},
                loop_count=loop_count,
            )
        )

    for index, call in enumerate(state.get("tool_calls", []) or []):
        if not isinstance(call, dict):
            continue
        events.append(_event_from_tool_call(call, index, loop_count))

    for index, item in enumerate(state.get("llm_errors", []) or []):
        if not isinstance(item, dict):
            continue
        events.append(
            ContextEvent(
                event_id=f"llm_error:{index}:{item.get('node', 'unknown')}",
                event_type="llm_event",
                source="llm_error",
                summary=_llm_error_summary(item),
                payload=_small_dict(item, 12),
                importance="high",
                retention="session",
                raw_ref={"state": "llm_errors", "index": index},
                loop_count=loop_count,
            )
        )

    for index, observation in enumerate(state.get("llm_observations", []) or []):
        if not isinstance(observation, dict):
            continue
        summary = str(observation.get("summary") or "").strip()
        if not summary:
            continue
        events.append(
            ContextEvent(
                event_id=f"llm_observation:{index}:{observation.get('latest_tool', 'unknown')}",
                event_type="progress_event",
                source="llm_observation",
                summary=_truncate(summary, 500),
                payload=_small_dict(observation, 16),
                importance=_importance_for_observation(observation),
                retention="session",
                raw_ref={"state": "llm_observations", "index": index},
                loop_count=loop_count,
            )
        )
    return events


def latest_tool_event(state: AgentState) -> ContextEvent | None:
    calls = state.get("tool_calls") or []
    if not calls:
        return None
    latest = calls[-1]
    if not isinstance(latest, dict):
        return None
    return _event_from_tool_call(latest, len(calls) - 1, int(state.get("loop_count", 0)))


def should_llm_observe_event(event: ContextEvent | None) -> bool:
    if event is None:
        return False
    if event.importance == "critical":
        return True
    if event.event_type == "file_event":
        return False
    if event.event_type in {"edit_event", "error_event"}:
        return event.importance in {"high", "critical"}
    if event.event_type == "verification_event":
        return bool(event.payload.get("exit_code") not in (None, 0))
    if event.event_type == "search_event":
        return bool(event.payload.get("candidate_count", 0) == 0)
    return False


def _event_from_tool_call(call: dict[str, Any], index: int, default_loop_count: int) -> ContextEvent:
    """
        从 tool_call 调用获取事件信息
    """
    name = str(call.get("name") or "unknown")
    output = call.get("output")
    if not isinstance(output, dict):
        output = {}
    event_type = _event_type_for_tool(name, output)
    importance = _importance_for_tool(name, output, event_type)
    retention = _retention_for_event(event_type, importance)
    payload = _payload_for_tool(name, call.get("input"), output)
    return ContextEvent(
        event_id=f"tool_call:{index}:{name}",
        event_type=event_type,
        source=name,
        summary=_summary_for_tool(name, output),
        payload=payload,
        importance=importance,
        retention=retention,
        raw_ref={"state": "tool_calls", "index": index, "name": name},
        loop_count=int(output.get("loop_count") or default_loop_count),
    )


def _event_type_for_tool(name: str, output: dict[str, Any]) -> EventType:
    if output.get("error") or output.get("fatal"):
        return "error_event"
    if name in {"search_code", "search_text", "search_code_context"}:
        return "search_event"
    if name == "read_file":
        return "file_event"
    if name in {"run_tests", "run_shell_command"} and output.get("purpose") == "verification":
        return "verification_event"
    if name == "run_tests":
        return "verification_event"
    if name == "apply_code_patch":
        return "edit_event"
    if name in {"EnterPlanMode", "ExitPlanMode"}:
        return "plan_event"
    if name == "request_user_input":
        return "user_event"
    return "tool_event"


def _importance_for_tool(name: str, output: dict[str, Any], event_type: EventType) -> Importance:
    """
        评价重要性
    """
    if output.get("fatal"):
        return "critical"
    if output.get("error") and not output.get("skipped"):
        return "high"
    if output.get("needs_user_input"):
        return "critical"
    if event_type in {"edit_event", "verification_event", "plan_event"}:
        return "high"
    if event_type == "file_event":
        return "high"
    if event_type == "search_event":
        return "medium" if _candidate_count(output) else "high"
    if name in {"git_diff", "list_files"}:
        return "low"
    return "medium"


def _retention_for_event(event_type: EventType, importance: Importance) -> Retention:
    """
        分级存储
    """
    if importance == "critical":
        return "working"
    if event_type in {"task_event", "user_event", "plan_event", "edit_event", "verification_event"}:
        return "working"
    if importance == "low":
        return "archive"
    return "session"


def _payload_for_tool(name: str, raw_input: Any, output: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": name,
        "input": _trim_value(raw_input, 1200),
    }
    for key in (
        "file_path",
        "pattern",
        "query",
        "command",
        "purpose",
        "exit_code",
        "error",
        "reason",
        "status",
        "skipped",
        "unsupported",
        "applied",
        "changed_files",
        "changed_line_count",
    ):
        if key in output:
            payload[key] = _trim_value(output.get(key), 1200)
    if name in {"search_text", "search_code"}:
        payload["candidate_count"] = len(output.get("matches") or [])
        payload["files"] = _files_from_matches(output.get("matches") or [])[:20]
    elif name == "search_code_context":
        payload["candidate_count"] = _candidate_count(output)
        payload["files"] = _files_from_code_context(output)[:20]
    elif name == "read_file":
        payload["content_excerpt"] = _truncate(str(output.get("content") or ""), 1600)
    elif name in {"run_tests", "run_shell_command"}:
        payload["stdout_excerpt"] = _truncate(str(output.get("stdout") or ""), 1000)
        payload["stderr_excerpt"] = _truncate(str(output.get("stderr") or ""), 1000)
    return payload


def _summary_for_tool(name: str, output: dict[str, Any]) -> str:
    if output.get("error"):
        return f"{name} failed: {_truncate(str(output.get('error')), 260)}"
    if output.get("skipped"):
        return f"{name} skipped: {_truncate(str(output.get('reason') or ''), 220)}"
    if name == "read_file":
        return f"Read file {output.get('file_path', 'unknown')}."
    if name in {"search_text", "search_code"}:
        return f"{name} returned {len(output.get('matches') or [])} matches."
    if name == "search_code_context":
        return f"search_code_context returned {_candidate_count(output)} candidates."
    if name in {"run_tests", "run_shell_command"}:
        return f"{name} command exit_code={output.get('exit_code')} purpose={output.get('purpose', '')}."
    if name == "apply_code_patch":
        return f"apply_code_patch applied={output.get('applied')} files={output.get('changed_files', [])}."
    if name in {"EnterPlanMode", "ExitPlanMode"}:
        return f"{name} approved={output.get('approved')} exited={output.get('exited')}."
    return f"{name} completed with keys: {', '.join(sorted(str(key) for key in output.keys()))}."


def _task_analysis_summary(data: dict[str, Any]) -> str:
    task_type = str(data.get("task_type") or "").strip()
    category = str(data.get("task_category") or "").strip()
    reason = str(data.get("verification_reason") or "").strip()
    return _join_non_empty(
        [
            f"task_type={task_type}" if task_type else "",
            f"category={category}" if category else "",
            f"verification_required={bool(data.get('verification_required', True))}",
            reason,
        ],
        "; ",
    )


def _plan_event_summary(item: dict[str, Any]) -> str:
    for key in ("summary", "technical_plan", "evaluation", "message", "reason"):
        value = str(item.get(key) or "").strip()
        if value:
            return _truncate(value, 500)
    return f"Plan event keys: {', '.join(sorted(str(key) for key in item.keys()))}."


def _llm_error_summary(item: dict[str, Any]) -> str:
    node = str(item.get("node") or "unknown")
    category = str(item.get("category") or "unknown")
    message = _truncate(str(item.get("message") or ""), 260)
    return f"LLM error node={node} category={category}: {message}"


def _importance_for_observation(item: dict[str, Any]) -> Importance:
    status = str(item.get("status") or "").lower()
    if status == "error":
        return "high"
    if item.get("missing_context"):
        return "high"
    return "medium"


def _candidate_count(output: dict[str, Any]) -> int:
    total = 0
    for key in ("files", "functions", "symbols", "api_routes", "db_models", "matches"):
        value = output.get(key)
        if isinstance(value, list):
            total += len(value)
    selected = output.get("selected_code_context")
    if isinstance(selected, dict):
        total += _candidate_count(selected)
    return total


def _files_from_code_context(output: dict[str, Any]) -> list[str]:
    files: list[str] = []
    selected = output.get("selected_code_context")
    if isinstance(selected, dict):
        output = selected
    for key in ("files", "functions", "symbols", "api_routes", "db_models", "matches"):
        for item in output.get(key, []) or []:
            path = ""
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("file_path") or "").strip()
            if path and path not in files:
                files.append(path)
    return files


def _files_from_matches(matches: list[Any]) -> list[str]:
    files: list[str] = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        path = str(item.get("file_path") or item.get("path") or "").strip()
        if path and path not in files:
            files.append(path)
    return files


def _small_dict(value: Any, limit: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _trim_value(item, 1200) for key, item in list(value.items())[:limit]}


def _trim_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, list):
        return [_trim_value(item, max_chars) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _trim_value(item, max_chars) for key, item in list(value.items())[:20]}
    return value


def _join_non_empty(values: list[str], sep: str) -> str:
    return sep.join(str(value).strip() for value in values if str(value).strip())
