"""
Local JSONL memory store.

This is not the final RGCM implementation. It is a stable baseline that lets the
agent write and retrieve memory cards while keeping storage swappable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from agent_runtime.memory.cards import MemoryCard


class JsonlMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, card: MemoryCard) -> dict:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        return {"memory_id": card.memory_id, "path": self.path.as_posix()}

    def list(self) -> list[dict]:
        if not self.path.exists():
            return []

        cards = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                cards.append(json.loads(line))
        return cards

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query_terms = set(self._tokens(query))
        scored: list[tuple[int, dict]] = []
        for card in self.list():
            text = " ".join(
                str(card.get(key, ""))
                for key in ("trigger", "content", "scope", "type", "status")
            )
            score = len(query_terms.intersection(self._tokens(text)))
            if score > 0:
                scored.append((score, card))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [card for _, card in scored[:limit]]

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

