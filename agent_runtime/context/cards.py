"""Structured context compression schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from utils import utc_now
from uuid import uuid4


@dataclass
class ContextItem:
    role: str
    content: str
    item_type: str = "message"
    pinned: bool = False
    metadata: dict = field(default_factory=dict)
    item_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContextDigest:
    summary: str
    current_goal: str
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    key_observations: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    code_changes: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    source_item_ids: list[str] = field(default_factory=list)
    compression_method: str = "rule_based"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ContextDigest":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    def render_for_prompt(self) -> str:
        sections = [
            ("Summary", [self.summary] if self.summary else []),
            ("Current Goal", [self.current_goal] if self.current_goal else []),
            ("Constraints", self.constraints),
            ("Decisions", self.decisions),
            ("Completed Tasks", self.completed_tasks),
            ("Open Tasks", self.open_tasks),
            ("Key Observations", self.key_observations),
            ("Code Changes", self.code_changes),
            ("Memory References", self.memory_refs),
        ]
        rendered = ["# Compressed Prior Context"]
        for title, values in sections:
            if not values:
                continue
            rendered.append(f"\n## {title}")
            for value in values:
                rendered.append(f"- {value}")
        if self.tool_results:
            rendered.append("\n## Tool Results")
            for result in self.tool_results:
                name = result.get("name", "unknown")
                status = result.get("status", "unknown")
                summary = result.get("summary", "")
                rendered.append(f"- {name} [{status}]: {summary}")
        return "\n".join(rendered)

    def with_error(self, error: str) -> "ContextDigest":
        data = self.to_dict()
        constraints = list(data.get("constraints") or [])
        constraints.append(f"context_compression_fallback={error}")
        data["constraints"] = constraints
        data["compression_method"] = f"{self.compression_method}_fallback"
        return ContextDigest.from_dict(data)
