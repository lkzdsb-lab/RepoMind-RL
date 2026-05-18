from __future__ import annotations

from dataclasses import dataclass, field
from utils import utc_now
from typing import Dict, Any


@dataclass
class TraceEvent:
    step_id: int
    node: str
    thought: str
    action: str | None = None
    action_input: Dict[str, Any] | None = None
    observation: Dict[str, Any] | None = None
    created_at: str = field(default_factory=utc_now)