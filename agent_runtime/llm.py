"""LLM client boundary used by runtime components.

The default client is intentionally inert. Configure provider, API base, model,
and API key env to enable a real call without changing compressor/executor code.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger


@dataclass
class LLMConfig:
    provider: str = "disabled"
    model: str = ""
    api_base: str = ""
    api_key_env: str = "LLM_API_KEY"
    timeout: int = 60
    temperature: float = 0.0
    max_output_chars: int = 12000


@dataclass
class LLMMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    response_format: dict | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    raw: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...


class DisabledLLMClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("LLM client is disabled or not configured.")


class OpenAICompatibleLLMClient:
    """Minimal chat-completions client using the standard library."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, request: LLMRequest) -> LLMResponse:
        endpoint = self._chat_endpoint()
        payload = {
            "model": request.model or self.config.model,
            "messages": [message.to_dict() for message in request.messages],
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.config.temperature
            ),
        }
        if request.response_format:
            payload["response_format"] = request.response_format

        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env) if self.config.api_key_env else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        http_request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        logger.info(
            "llm request started provider={} model={} endpoint={} messages={}",
            self.config.provider,
            payload["model"],
            endpoint,
            len(request.messages),
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.config.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.warning("llm http error status={} detail_chars={}", exc.code, len(detail))
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            logger.warning("llm request failed error={}", exc)
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = _extract_chat_content(raw)
        logger.info(
            "llm request completed model={} content_chars={}",
            raw.get("model") or payload["model"],
            len(content),
        )
        return LLMResponse(
            content=content[: self.config.max_output_chars],
            model=str(raw.get("model") or payload["model"]),
            raw=raw,
        )

    def _chat_endpoint(self) -> str:
        base = self.config.api_base.rstrip("/")
        if not base:
            raise RuntimeError("LLM api_base is required.")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


def build_llm_client(config: LLMConfig) -> LLMClient:
    provider = config.provider.strip().lower()
    if provider in {"", "disabled", "none"}:
        return DisabledLLMClient()
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        if not config.model:
            return DisabledLLMClient()
        if not config.api_base:
            return DisabledLLMClient()
        return OpenAICompatibleLLMClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _extract_chat_content(raw: dict) -> str:
    choices = raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""
