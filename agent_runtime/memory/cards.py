"""
Memory card schema for Reward-Gated Causal Memory v1.

The first version stores JSONL locally. The schema is deliberately close to the
project overview so it can later move to SQLite/vector search without changing
executor code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


MemoryType = Literal["episodic", "semantic", "procedural", "anti_pattern"]
MemoryStatus = Literal["draft", "verified", "deprecated"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryCard:
    type: MemoryType
    scope: str
    trigger: str
    content: str
    evidence: list[str] = field(default_factory=list)
    reward_credit: float = 0.0
    reuse_success: int = 0
    reuse_failure: int = 0
    conflict_score: float = 0.0
    status: MemoryStatus = "draft"
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    last_used_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

