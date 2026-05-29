"""Prompt template loading utilities."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROMPT_ROOT = Path(__file__).resolve().parent
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@lru_cache(maxsize=128)
def load_prompt(relative_path: str) -> str:
    path = _safe_prompt_path(relative_path)
    return path.read_text(encoding="utf-8").strip()


def render_prompt(relative_path: str, **values: Any) -> str:
    """
        路由特定的 prompt 模版
    """
    template = load_prompt(relative_path)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Missing prompt variable `{key}` for {relative_path}")
        value = values[key]
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(replace, template)


def _safe_prompt_path(relative_path: str) -> Path:
    root = PROMPT_ROOT.resolve()
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Prompt path escapes prompt root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {relative_path}")
    return path
