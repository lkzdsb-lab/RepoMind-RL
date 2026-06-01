"""Storage adapters for layered memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Protocol

from agent_runtime.memory.cards import MemoryCard, MemorySearchResult, MemoryTier


class MemoryStore(Protocol):
    tier: MemoryTier

    def append_card(self, card: MemoryCard) -> MemoryCard:
        ...

    def get_card(self, memory_id: str) -> MemoryCard | None:
        ...

    def upsert_card(self, card: MemoryCard) -> MemoryCard:
        ...

    def update_card(self, memory_id: str, **updates: Any) -> MemoryCard | None:
        ...

    def list_cards(self) -> list[MemoryCard]:
        ...

    def search_cards(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        ...

    def touch_card(self, memory_id: str, used_at: str | None = None) -> MemoryCard | None:
        ...

    def deprecate_card(self, memory_id: str, reason: str = "") -> MemoryCard | None:
        ...

    def record_reuse_feedback(self, memory_id: str, success: bool) -> MemoryCard | None:
        ...


class JsonlMemoryStore:
    tier: MemoryTier = "mid_term"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, card: MemoryCard) -> dict:
        self.append_card(card)
        return {"memory_id": card.memory_id, "path": self.path.as_posix()}

    def append_card(self, card: MemoryCard) -> MemoryCard:
        card = card.with_updates(tier=self.tier)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        return card

    def get_card(self, memory_id: str) -> MemoryCard | None:
        for card in self.list_cards():
            if card.memory_id == memory_id:
                return card
        return None

    def upsert_card(self, card: MemoryCard) -> MemoryCard:
        card = card.with_updates(tier=self.tier)
        cards = self.list_cards()
        for index, existing in enumerate(cards):
            if existing.memory_id == card.memory_id:
                cards[index] = card
                self._write_cards(cards)
                return card
        cards.append(card)
        self._write_cards(cards)
        return card

    def update_card(self, memory_id: str, **updates: Any) -> MemoryCard | None:
        cards = self.list_cards()
        for index, card in enumerate(cards):
            if card.memory_id != memory_id:
                continue
            updated = card.with_updates(**updates)
            cards[index] = updated
            self._write_cards(cards)
            return updated
        return None

    def touch_card(self, memory_id: str, used_at: str | None = None) -> MemoryCard | None:
        cards = self.list_cards()
        touched: MemoryCard | None = None

        for index, card in enumerate(cards):
            if card.memory_id != memory_id:
                continue
            updated = card.mark_retrieved(used_at)
            cards[index] = updated
            touched = touched or updated

        if touched is None:
            return None

        self._write_cards(cards)
        return touched

    def list(self) -> list[dict]:
        return [card.to_dict() for card in self.list_cards()]

    def list_cards(self) -> list[MemoryCard]:
        if not self.path.exists():
            return []

        cards = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    cards.append(MemoryCard.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        return cards

    def search(self, query: str, limit: int = 5) -> list[dict]:
        return [result.to_dict() for result in self.search_cards(query, limit)]

    def search_cards(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        query_terms = set(self._tokens(query))
        scored: list[MemorySearchResult] = []
        for card in self.list_cards():
            if card.status == "deprecated":
                continue
            score = self._lexical_score(query_terms, card)
            if score > 0:
                scored.append(MemorySearchResult(card=card, score=score, source=card.tier))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _lexical_score(self, query_terms: set[str], card: MemoryCard) -> float:
        text = " ".join(
            [
                card.trigger,
                card.content,
                card.scope,
                card.type,
                card.status,
                " ".join(card.tags),
            ]
        )
        memory_terms = set(self._tokens(text))
        if not query_terms or not memory_terms:
            return 0.0
        overlap = len(query_terms.intersection(memory_terms))
        return overlap / max(len(query_terms), 1) + min(card.promotion_score(), 2.0) * 0.1

    def _tokens(self, text: str) -> Iterable[str]:
        token = []
        for char in text.lower():
            if char.isalnum() or char in {"_", "-"}:
                token.append(char)
            elif token:
                yield "".join(token)
                token = []
        if token:
            yield "".join(token)

    def _write_cards(self, cards: list[MemoryCard]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            for card in cards:
                fp.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        tmp_path.replace(self.path)

    def deprecate_card(self, memory_id: str, reason: str = "") -> MemoryCard | None:
        card = self.get_card(memory_id)
        if card is None:
            return None
        metadata = dict(card.metadata)
        if reason:
            metadata["deprecated_reason"] = reason
        return self.update_card(memory_id, status="deprecated", metadata=metadata)

    def record_reuse_feedback(self, memory_id: str, success: bool) -> MemoryCard | None:
        card = self.get_card(memory_id)
        if card is None:
            return None
        updates: dict[str, Any]
        if success:
            updates = {"reuse_success": card.reuse_success + 1}
        else:
            updates = {
                "reuse_failure": card.reuse_failure + 1,
                "conflict_score": min(card.conflict_score + 0.1, 1.0),
            }
        return self.update_card(memory_id, **updates)


class LocalVectorMemoryStore(JsonlMemoryStore):
    """A local vector-store stand-in using token vectors.

    It preserves the long-term vector DB boundary without requiring an external
    service in local development. A real vector database can implement the same
    MemoryStore protocol.
    """

    tier: MemoryTier = "long_term"

    def append_card(self, card: MemoryCard) -> MemoryCard:
        return super().append_card(card.with_updates(tier=self.tier))

    def search_cards(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        query_terms = set(self._tokens(query))
        scored: list[MemorySearchResult] = []
        for card in self.list_cards():
            if card.status == "deprecated":
                continue
            score = self._cosine_token_score(query_terms, card)
            if score > 0:
                scored.append(MemorySearchResult(card=card, score=score, source=self.tier))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _cosine_token_score(self, query_terms: set[str], card: MemoryCard) -> float:
        text = " ".join([card.trigger, card.content, " ".join(card.tags), card.type])
        memory_terms = set(self._tokens(text))
        if not query_terms or not memory_terms:
            return 0.0
        overlap = len(query_terms.intersection(memory_terms))
        cosine = overlap / ((len(query_terms) * len(memory_terms)) ** 0.5)
        return cosine + min(card.promotion_score(), 2.0) * 0.15


class RedisMemoryStore:
    """Optional Redis-backed mid-term memory adapter.

    The implementation is intentionally lazy: importing this module does not
    require redis-py. Configure it only in environments that provide Redis.
    """

    tier: MemoryTier = "mid_term"

    def __init__(self, url: str, namespace: str = "repomind:memory") -> None:
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise RuntimeError("RedisMemoryStore requires the `redis` package.") from exc

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.namespace = namespace

    def append_card(self, card: MemoryCard) -> MemoryCard:
        card = card.with_updates(tier=self.tier)
        self.client.hset(
            self.namespace,
            card.memory_id,
            json.dumps(card.to_dict(), ensure_ascii=False),
        )
        return card

    def get_card(self, memory_id: str) -> MemoryCard | None:
        value = self.client.hget(self.namespace, memory_id)
        if value is None:
            return None
        return MemoryCard.from_dict(json.loads(value))

    def upsert_card(self, card: MemoryCard) -> MemoryCard:
        return self.append_card(card)

    def update_card(self, memory_id: str, **updates: Any) -> MemoryCard | None:
        card = self.get_card(memory_id)
        if card is None:
            return None
        updated = card.with_updates(**updates)
        self.client.hset(
            self.namespace,
            updated.memory_id,
            json.dumps(updated.to_dict(), ensure_ascii=False),
        )
        return updated

    def list_cards(self) -> list[MemoryCard]:
        values = self.client.hvals(self.namespace)
        return [MemoryCard.from_dict(json.loads(value)) for value in values]

    def touch_card(self, memory_id: str, used_at: str | None = None) -> MemoryCard | None:
        value = self.client.hget(self.namespace, memory_id)
        if value is None:
            return None
        card = MemoryCard.from_dict(json.loads(value))
        updated = card.mark_retrieved(used_at)
        self.client.hset(
            self.namespace,
            updated.memory_id,
            json.dumps(updated.to_dict(), ensure_ascii=False),
        )
        return updated

    def search_cards(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        fallback = JsonlMemoryStore(Path(".repomind/unused-memory-search.jsonl"))
        scored = []
        query_terms = set(fallback._tokens(query))
        for card in self.list_cards():
            if card.status == "deprecated":
                continue
            score = fallback._lexical_score(query_terms, card)
            if score > 0:
                scored.append(MemorySearchResult(card=card, score=score, source=self.tier))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def deprecate_card(self, memory_id: str, reason: str = "") -> MemoryCard | None:
        card = self.get_card(memory_id)
        if card is None:
            return None
        metadata = dict(card.metadata)
        if reason:
            metadata["deprecated_reason"] = reason
        return self.update_card(memory_id, status="deprecated", metadata=metadata)

    def record_reuse_feedback(self, memory_id: str, success: bool) -> MemoryCard | None:
        card = self.get_card(memory_id)
        if card is None:
            return None
        if success:
            return self.update_card(memory_id, reuse_success=card.reuse_success + 1)
        return self.update_card(
            memory_id,
            reuse_failure=card.reuse_failure + 1,
            conflict_score=min(card.conflict_score + 0.1, 1.0),
        )
