"""Runtime sink for live user-facing progress updates."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable


UserUpdateSink = Callable[[dict[str, Any]], None]

_USER_UPDATE_SINK: ContextVar[UserUpdateSink | None] = ContextVar(
    "lee_agent_user_update_sink",
    default=None,
)


def set_user_update_sink(sink: UserUpdateSink | None) -> None:
    _USER_UPDATE_SINK.set(sink)


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

