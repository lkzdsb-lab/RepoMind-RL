"""Memory schemas for layered, reward-gated agent memory."""

from __future__ import annotations
from utils import utc_now
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


MemoryType = Literal["episodic", "semantic", "procedural", "anti_pattern"]
MemoryStatus = Literal["draft", "verified", "deprecated"]
MemoryTier = Literal["short_term", "mid_term", "long_term", "skill"]





@dataclass
class MemoryCard:
    type: MemoryType
    scope: str
    trigger: str
    content: str
    tier: MemoryTier = "mid_term"
    evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    reward_credit: float = 0.0
    """ 得分，影响后续能否晋升? """
    reuse_success: int = 0
    reuse_failure: int = 0
    conflict_score: float = 0.0
    status: MemoryStatus = "draft"
    source_task_id: str | None = None
    promoted_from: str | None = None
    skill_name: str | None = None
    metadata: dict = field(default_factory=dict)
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_used_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryCard":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    def with_updates(self, **updates: object) -> "MemoryCard":
        """
        更新卡片信息
        """
        return replace(self, updated_at=utc_now(), **updates)

    def mark_retrieved(self, used_at: str | None = None) -> "MemoryCard":
        metadata = dict(self.metadata)
        try:
            retrieval_count = int(metadata.get("retrieval_count", 0))
        except (TypeError, ValueError):
            retrieval_count = 0
        metadata["retrieval_count"] = retrieval_count + 1
        return self.with_updates(last_used_at=used_at or utc_now(), metadata=metadata)

    def promotion_score(self) -> float:
        """
        计算当前卡片�? 推荐�?
        """
        reuse_score = 0.15 * self.reuse_success - 0.25 * self.reuse_failure
        age_penalty = min(max(self.conflict_score, 0.0), 1.0)
        return self.reward_credit + reuse_score - age_penalty


@dataclass
class MemorySearchResult:
    card: MemoryCard
    score: float
    source: MemoryTier

    def to_dict(self) -> dict:
        data = self.card.to_dict()
        data["score"] = self.score
        data["source"] = self.source
        return data


@dataclass
class MemoryContextPack:
    short_term: list[MemorySearchResult] = field(default_factory=list)
    mid_term: list[MemorySearchResult] = field(default_factory=list)
    long_term: list[MemorySearchResult] = field(default_factory=list)
    skill: list[MemorySearchResult] = field(default_factory=list)

    def all_results(self) -> list[MemorySearchResult]:
        return self.short_term + self.mid_term + self.long_term + self.skill

    def to_dict(self) -> dict:
        return {
            "short_term": [item.to_dict() for item in self.short_term],
            "mid_term": [item.to_dict() for item in self.mid_term],
            "long_term": [item.to_dict() for item in self.long_term],
            "skill": [item.to_dict() for item in self.skill],
        }

    def render_for_prompt(self, max_items: int = 12) -> str:
        sections = [
            ("Short-term Context", self.short_term),
            ("Episodic Memory", [r for r in self.mid_term if r.card.type == "episodic"]),
            ("Semantic Memory", [r for r in self.long_term if r.card.type == "semantic"]),
            ("Procedural Memory", [r for r in self.long_term if r.card.type == "procedural"]),
            ("Anti-pattern Memory", [r for r in self.long_term if r.card.type == "anti_pattern"]),
            ("Skill Memory", self.skill),
        ]
        rendered: list[str] = []
        used = 0
        for title, results in sections:
            if not results or used >= max_items:
                continue
            lines = []
            for result in results[: max_items - used]:
                card = result.card
                lines.append(
                    f"- [{card.type}/{card.status}/score={result.score:.2f}] "
                    f"{card.trigger}: {card.content}"
                )
                used += 1
            rendered.append(f"## {title}\n" + "\n".join(lines))
        return "\n\n".join(rendered)
