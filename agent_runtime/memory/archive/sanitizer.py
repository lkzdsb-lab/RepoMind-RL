"""Recursive secret redaction and size bounding for durable task artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agent_runtime.memory.archive.security import is_sensitive_path


REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "private_key",
    "cookie",
    "set_cookie",
}
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|client[_-]?secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def sanitize(value: Any, *, max_text_chars: int = 200_000) -> Any:
    """Return a JSON-safe copy with secrets redacted and large strings bounded."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        sensitive_document = is_sensitive_path(
            str(value.get("file_path") or value.get("path") or "")
        )
        for raw_key, item in value.items():
            key = str(raw_key)
            redact_document_content = sensitive_document and key.lower() in {
                "content",
                "content_excerpt",
                "snapshot",
                "spans",
            }
            result[key] = REDACTED if _is_sensitive_key(key) or redact_document_content else sanitize(
                item, max_text_chars=max_text_chars
            )
        return result
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, max_text_chars=max_text_chars) for item in value]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return _sanitize_text(value, max_text_chars)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(str(value), max_text_chars)


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith("_password")


def _sanitize_text(text: str, max_text_chars: int) -> str:
    clean = _BEARER.sub("Bearer [REDACTED]", text)
    clean = _OPENAI_STYLE_KEY.sub(REDACTED, clean)
    clean = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}={REDACTED}", clean)
    if len(clean) <= max_text_chars:
        return clean
    removed = len(clean) - max_text_chars
    return f"{clean[:max_text_chars]}\n...[TRUNCATED {removed} CHARACTERS]"
