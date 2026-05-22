"""LLM client boundary used by runtime components."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from config import LLMConfig
from loguru import logger
from model.llm import LLMMessage, LLMRequest, LLMResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...


class DisabledLLMClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("LLM client is disabled or not configured.")


class OpenAICompatibleLLMClient:
    """
        llm 客户端
    """
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = os.getenv(config.api_key_env) if config.api_key_env else ""
        if not api_key:
            raise RuntimeError(
                f"LLM API key env var `{config.api_key_env}` is not set for provider `{config.provider}`."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.api_base or None,
            timeout=config.timeout,
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.config.model
        if not model:
            raise RuntimeError("LLM model is required.")

        messages = [self._message_to_dict(message) for message in request.messages]
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.config.temperature
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        logger.info(
            "llm request started provider={} model={} base_url={} messages={}",
            self.config.provider,
            model,
            self.config.api_base,
            len(messages),
        )
        try:
            if not _is_pydantic_response_model(request.response_format):
                raise RuntimeError(
                    "OpenAI SDK structured response parsing requires a Pydantic response_format."
                )
            kwargs["response_format"] = request.response_format
            completion = self.client.beta.chat.completions.parse(**kwargs)
            parsed = _extract_parsed_message(completion)
        except OpenAIError as exc:
            logger.warning("llm request failed error_type={} error={}", exc.__class__.__name__, exc)
            raise RuntimeError(f"LLM request failed via OpenAI SDK: {exc}") from exc

        raw = completion.model_dump()
        content = _extract_chat_content(raw)
        if not content and parsed is not None:
            content = json.dumps(_parsed_to_plain(parsed), ensure_ascii=False)
        logger.info(
            "llm request completed model={} content={} parsed={}",
            raw.get("model") or model,
            content,
            parsed is not None,
        )
        return LLMResponse(
            content=content[: self.config.max_output_chars],
            model=str(raw.get("model") or model),
            raw=raw,
            parsed=parsed,
        )

    def _message_to_dict(self, message: LLMMessage | dict[str, str]) -> dict[str, str]:
        if isinstance(message, dict):
            return {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        return message.to_dict()


def build_llm_client(config: LLMConfig) -> LLMClient:
    provider = config.provider.strip().lower()
    if provider in {"", "disabled", "none"}:
        return DisabledLLMClient()
    if provider in {"openai", "openai_compatible", "openai-compatible", "enable"}:
        if not config.model:
            return DisabledLLMClient()
        return OpenAICompatibleLLMClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _is_pydantic_response_model(response_format: Any) -> bool:
    return isinstance(response_format, type) and issubclass(response_format, BaseModel)


def _extract_parsed_message(completion: Any) -> Any:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    return getattr(message, "parsed", None)


def _parsed_to_plain(parsed: Any) -> Any:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    if isinstance(parsed, list):
        return [_parsed_to_plain(item) for item in parsed]
    if isinstance(parsed, dict):
        return {key: _parsed_to_plain(value) for key, value in parsed.items()}
    return parsed


def _extract_chat_content(raw: dict) -> str:
    choices = raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content or "")
