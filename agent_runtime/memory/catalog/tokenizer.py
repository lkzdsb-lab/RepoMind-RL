"""Versioned deterministic tokenization for code and Chinese/English text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from agent_runtime.memory.domain.models import MemoryDocument


TOKENIZER_VERSION = "keyword-v1"

_SEGMENTS = re.compile(r"[A-Za-z0-9_./:\\-]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_CAMEL_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
_CODE_SPLIT = re.compile(r"[_./:\\-]+")


class KeywordNormalizer:
    """Produce portable tokens without relying on optional SQLite extensions."""

    version = TOKENIZER_VERSION

    def tokens(self, *values: object, limit: int = 512) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in _flatten(values):
            normalized = unicodedata.normalize("NFKC", str(value or ""))
            for segment in _SEGMENTS.findall(normalized):
                generated = (
                    _cjk_tokens(segment)
                    if _is_cjk(segment[0])
                    else _code_tokens(segment)
                )
                for token in generated:
                    clean = token.casefold().strip()
                    if clean and clean not in seen:
                        seen.add(clean)
                        result.append(clean)
                        if len(result) >= limit:
                            return tuple(result)
        return tuple(result)

    def document_tokens(self, document: MemoryDocument) -> tuple[str, ...]:
        return self.tokens(
            document.title,
            document.knowledge,
            document.applicability,
            document.triggers,
            document.tags,
            document.scope.module,
            document.scope.files,
            document.scope.symbols,
            limit=2000,
        )


def _flatten(values: Iterable[object]) -> Iterable[object]:
    for value in values:
        if isinstance(value, (list, tuple, set)):
            yield from _flatten(value)
        else:
            yield value


def _is_cjk(value: str) -> bool:
    return "\u3400" <= value <= "\u4dbf" or "\u4e00" <= value <= "\u9fff"


def _cjk_tokens(value: str) -> list[str]:
    characters = list(value)
    bigrams = [value[index : index + 2] for index in range(len(value) - 1)]
    return characters + bigrams


def _code_tokens(value: str) -> list[str]:
    result = [value]
    for component in _CODE_SPLIT.split(value):
        if not component:
            continue
        result.append(component)
        result.extend(_CAMEL_PARTS.findall(component))
    return result
