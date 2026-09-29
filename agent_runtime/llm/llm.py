"""LLM client boundary used by runtime components."""

from __future__ import annotations

import json
import os
import time
from threading import Lock
from typing import Any, Protocol

from config import LLMConfig, validate_llm_generation
from loguru import logger
from model.llm import LLMMessage, LLMRequest, LLMResponse
from openai import BadRequestError, OpenAI, OpenAIError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError
from utils import _safe_int


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...


class DisabledLLMClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("LLM client is disabled or not configured.")


class LLMResponseError(RuntimeError):
    """A completed but unusable response, retaining provider-reported usage."""

    def __init__(self, message: str, category: str, completion: Any) -> None:
        super().__init__(message)
        self.category = category
        raw = completion.model_dump()
        self.response = LLMResponse(content="", model=str(raw.get("model") or ""),
                                    raw=raw, parsed=None, usage=_extract_usage(raw))


_UNSUPPORTED_STRUCTURED_FORMATS: set[tuple[str, str]] = set()
_FORMAT_CACHE_LOCK = Lock()


class OpenAICompatibleLLMClient:
    """
        llm 客户端
    """
    def __init__(self, config: LLMConfig) -> None:
        validate_llm_generation(config)
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
            max_retries=config.max_retries,
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
        if self.config.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.config.max_completion_tokens
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.extra_body:
            kwargs["extra_body"] = dict(self.config.extra_body)
        node = str(request.metadata.get("node") or "unknown")

        logger.info(
            "llm request started provider={} model={} base_url={} messages={} format_mode={}",
            self.config.provider,
            model,
            self.config.api_base,
            len(messages),
            self.config.response_format_mode,
        )
        if not _is_pydantic_response_model(request.response_format):
            raise RuntimeError("Structured responses require a Pydantic response_format.")
        mode = self.config.response_format_mode.lower()
        format_key = (str(self.config.api_base or "").rstrip("/"), model)
        with _FORMAT_CACHE_LOCK:
            known_unsupported = format_key in _UNSUPPORTED_STRUCTURED_FORMATS
        if mode == "json" or (mode == "auto" and known_unsupported):
            completion, parsed = self._plain_completion(
                kwargs=kwargs, node=node,
                response_model=request.response_format,
            )
        else:
            try:
                completion = self._invoke_completion(
                    kwargs, node=node, response_model=request.response_format,
                )
                parsed = _extract_parsed_message(completion)
            except Exception as exc:
                if isinstance(exc, LLMResponseError):
                    raise
                unsupported = mode == "auto" and self.config.structured_fallback and _unsupported_structured_format(exc)
                if unsupported:
                    with _FORMAT_CACHE_LOCK:
                        _UNSUPPORTED_STRUCTURED_FORMATS.add(format_key)
                    logger.warning(
                        "structured response format unavailable; using JSON text mode endpoint={} model={}",
                        format_key[0], model,
                    )
                elif mode == "native" or isinstance(exc, OpenAIError) or not self.config.structured_fallback:
                    raise
                else:
                    logger.warning(
                        "structured parse failed; retrying with plain completion error_type={} error={}",
                        exc.__class__.__name__, exc,
                    )
                completion, parsed = self._plain_completion(
                    kwargs=kwargs, node=node,
                    response_model=request.response_format,
                    original_error=exc,
                )

        raw = completion.model_dump()
        content = _extract_chat_content(raw)
        if not content and parsed is not None:
            content = json.dumps(_parsed_to_plain(parsed), ensure_ascii=False)
        logged_content = _format_json_for_log(content)
        logger.info(
            "llm request completed model={} content=\n{}\nparsed={} usage={}",
            raw.get("model") or model,
            logged_content,
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

    def _plain_completion(
        self,
        *,
        kwargs: dict[str, Any],
        node: str,
        response_model: Any,
        original_error: Exception | None = None,
    ) -> tuple[Any, Any]:
        """Use the same model contract for JSON prompting and local validation."""
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        messages = list(kwargs["messages"]) + [{
            "role": "system",
            "content": "Return exactly one JSON object matching this JSON Schema. "
                       "Use empty strings/lists for empty optional text/list fields, not null. "
                       "Preserve required fields and enum values.\n" + schema,
        }]
        completion = self._invoke_completion({**kwargs, "messages": messages}, node=node)

        raw = completion.model_dump()
        content = _extract_chat_content(raw)
        try:
            parsed = _manual_parse_response(content, response_model)
        except Exception as manual_exc:
            issues = (
                [{"field": ".".join(map(str, error["loc"])), "type": error["type"]}
                 for error in manual_exc.errors(include_url=False, include_input=False)]
                if isinstance(manual_exc, ValidationError) else [{"type": type(manual_exc).__name__}]
            )
            logger.bind(llm_node=node).warning(
                "LLM response validation failed model={} issues={} response_excerpt={}",
                response_model.__name__, issues, content[:4000],
            )
            fields = ", ".join(f"{item.get('field', '<root>')} ({item['type']})" for item in issues[:5])
            raise LLMResponseError(
                f"{response_model.__name__}: {len(issues)} response validation error(s): {fields}",
                "response_format", completion,
            ) from manual_exc
        return completion, parsed

    def _invoke_completion(self, kwargs: dict[str, Any], *, node: str, response_model: Any = None) -> Any:
        """Shared transport for native and JSON responses, including format fallback."""
        started = time.perf_counter()
        request_log = logger.bind(llm_node=node)
        request_log.info(
            "llm generation request model={} max_completion_tokens={} reasoning_effort={} thinking_budget={} timeout={} max_retries={}",
            kwargs["model"], kwargs.get("max_completion_tokens"), kwargs.get("reasoning_effort"),
            (kwargs.get("extra_body") or {}).get("thinking_budget"), self.config.timeout, self.config.max_retries,
        )
        completion = None
        try:
            if response_model is None:
                completion = self.client.chat.completions.create(**kwargs)
            else:
                try:
                    completion = self.client.beta.chat.completions.parse(**kwargs, response_format=response_model)
                except LengthFinishReasonError as exc:
                    completion = exc.completion
                    raise LLMResponseError("LLM generation budget exhausted (finish_reason=length).", "generation_budget", completion) from exc
            if any(choice.finish_reason == "length" for choice in completion.choices):
                raise LLMResponseError("LLM generation budget exhausted (finish_reason=length).", "generation_budget", completion)
            return completion
        except Exception as exc:
            request_log.warning("llm generation failed error_type={} category={}", type(exc).__name__, getattr(exc, "category", type(exc).__name__))
            raise
        finally:
            raw = completion.model_dump() if completion is not None else {}
            request_log.info(
                "llm generation finished elapsed_ms={:.1f} finish_reason={} usage={}",
                (time.perf_counter() - started) * 1000,
                [choice.get("finish_reason") for choice in raw.get("choices", [])],
                raw.get("usage") if completion is not None else "unknown",
            )


def _unsupported_structured_format(exc: Exception) -> bool:
    if not isinstance(exc, BadRequestError):
        return False
    body = getattr(exc, "body", None)
    details = json.dumps(body, ensure_ascii=False, default=str) if body is not None else str(exc)
    details = details.lower()
    return ("response_format" in details or "json_schema" in details) and any(
        phrase in details for phrase in ("unavailable", "unsupported", "not support", "not available")
    )


def _format_json_for_log(content: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return content
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


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
