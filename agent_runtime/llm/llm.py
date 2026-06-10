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
from utils import _safe_int


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
        except Exception as exc:
            logger.warning(
                "structured parse failed; retrying with plain completion error_type={} error={}",
                exc.__class__.__name__,
                exc,
            )
            completion, parsed = self._retry_plain_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                response_model=request.response_format,
                original_error=exc,
            )

        raw = completion.model_dump()
        content = _extract_chat_content(raw)
        if not content and parsed is not None:
            content = json.dumps(_parsed_to_plain(parsed), ensure_ascii=False)
        logger.info(
            "llm request completed model={} content={} parsed={} usage={}",
            raw.get("model") or model,
            content,
            parsed is not None,
            raw.get("usage") or {},
        )
        return LLMResponse(
            content=content[: self.config.max_output_chars],
            model=str(raw.get("model") or model),
            raw=raw,
            parsed=parsed,
            usage=_extract_usage(raw),
        )

    def _message_to_dict(self, message: LLMMessage | dict[str, str]) -> dict[str, str]:
        if isinstance(message, dict):
            return {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        return message.to_dict()

    def _retry_plain_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        response_model: Any,
        original_error: Exception,
    ) -> tuple[Any, Any]:
        """ 降级以适配老模型传输格式出问题的场景 todo 后期换好模型考虑删除"""
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except OpenAIError as exc:
            logger.warning("llm request failed error_type={} error={}", exc.__class__.__name__, exc)
            raise RuntimeError(f"LLM request failed via OpenAI SDK: {exc}") from exc

        raw = completion.model_dump()
        content = _extract_chat_content(raw)
        try:
            parsed = _manual_parse_response(content, response_model)
        except Exception as manual_exc:
            raise RuntimeError(
                f"LLM structured parse failed: {original_error}; manual JSON recovery also failed: {manual_exc}"
            ) from manual_exc
        return completion, parsed


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


def _extract_usage(raw: dict) -> dict[str, Any]:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
        "completion_tokens": _safe_int(usage.get("completion_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
    }


def _manual_parse_response(content: str, response_model: Any) -> Any:
    if not _is_pydantic_response_model(response_model):
        raise RuntimeError("Manual structured parsing requires a Pydantic response model.")
    cleaned = _strip_markdown_fence(content)
    return response_model.model_validate_json(cleaned)


def _strip_markdown_fence(content: str) -> str:
    """ 模型兼容逻辑"""
    text = str(content or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if not lines:
        return text
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
