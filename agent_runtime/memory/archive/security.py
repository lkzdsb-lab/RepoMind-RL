"""Shared security classification used by archive capture and sanitization."""

from __future__ import annotations


def is_sensitive_path(path: str) -> bool:
    name = str(path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return (
        name == ".env"
        or name.startswith(".env.")
        or name in {"credentials.json", "id_rsa", "id_ed25519"}
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
    )
