"""Small token-budget helpers.

This intentionally estimates instead of depending on tokenizer packages. The
compressor only needs a stable budget signal.
"""

from __future__ import annotations

from agent_runtime.context.cards import ContextItem


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii_chars // 2)


def estimate_context_tokens(items: list[ContextItem]) -> int:
    return sum(estimate_tokens(item.content) for item in items)
