"""Runtime sink for live user-facing progress updates."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable


UserUpdateSink = Callable[[dict[str, Any]], None]
ChangeEventSink = Callable[[dict[str, Any]], None]

_USER_UPDATE_SINK: ContextVar[UserUpdateSink | None] = ContextVar(
    "lee_agent_user_update_sink",
    default=None,
)
_CHANGE_EVENT_SINK: ContextVar[ChangeEventSink | None] = ContextVar(
    "lee_agent_change_event_sink",
    default=None,
)


def set_user_update_sink(sink: UserUpdateSink | None) -> None:
    _USER_UPDATE_SINK.set(sink)


def set_change_event_sink(sink: ChangeEventSink | None) -> None:
    _CHANGE_EVENT_SINK.set(sink)


def emit_user_update(update: dict[str, Any]) -> bool:
    """
        回调函数用于实时展示 llm 的操作
    """
    sink = _USER_UPDATE_SINK.get()
    if sink is None:
        return False
    try:
        sink(dict(update))
    except Exception:
        return False
    return True


def emit_change_event(event: dict[str, Any]) -> bool:
    sink = _CHANGE_EVENT_SINK.get()
    if sink is None:
        return False
    try:
        sink(dict(event))
    except Exception:
        return False
    return True
