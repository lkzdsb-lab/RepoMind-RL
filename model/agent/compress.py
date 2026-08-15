from dataclasses import dataclass, field, asdict
from typing import Any, Literal
from utils import utc_now

EventType = Literal[
    "task_event",
    "user_event",
    "plan_event",
    "tool_event",
    "search_event",
    "file_event",
    "edit_event",
    "verification_event",
    "error_event",
    "llm_event",
    "progress_event",
]
Importance = Literal["low", "medium", "high", "critical"]
Retention = Literal["working", "session", "archive"]

@dataclass
class ContextEvent:
    """
        event 模型
    """
    event_id: str
    event_type: EventType
    source: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    importance: Importance = "medium"
    retention: Retention = "session"
    raw_ref: dict[str, Any] = field(default_factory=dict)
    loop_count: int = 0
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DistillationLevel = Literal["pinned", "semantic", "archive"]


@dataclass
class DistilledEvent:
    """
        蒸馏 model
    """
    event_id: str
    event_type: str
    source: str
    level: DistillationLevel
    importance: str
    retention: str
    summary: str
    facts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    raw_ref: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)