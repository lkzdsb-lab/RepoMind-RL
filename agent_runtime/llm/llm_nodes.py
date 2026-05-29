"""Reusable JSON-oriented LLM node runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from agent_runtime.llm.llm import LLMClient, build_llm_client
from agent_runtime.user_updates import emit_user_update
from config import LLMConfig
from loguru import logger
from model.agent.graph import AgentState
from model.llm import LLMMessage, LLMRequest
from pydantic import BaseModel
from utils import _safe_int


PromptBuilder = Callable[[AgentState, dict[str, Any]], str]
JsonFallback = Callable[[AgentState, dict[str, Any]], dict[str, Any]]
JsonNormalizer = Callable[[dict[str, Any], AgentState, dict[str, Any]], dict[str, Any]]


@dataclass
class LLMJsonNode:
    """
    Shared runner for LLM-backed runtime stages that expect a JSON object.

    Business nodes provide the prompt, fallback and optional normalizer. This
    class owns OpenAI-compatible invocation, JSON parsing, logging and fallback
    behavior so each stage does not need to reimplement those mechanics.
    """

    name: str
    llm_config: LLMConfig
    system_prompt: str
    build_prompt: PromptBuilder
    fallback: JsonFallback | None
    response_model: type[BaseModel]
    normalize: JsonNormalizer | None = None
    client: LLMClient | None = None
    raise_on_error: bool = False
    _client: LLMClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = self.client or build_llm_client(self.llm_config)

    def run(
        self,
        state: AgentState,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = dict(context or {})
        try:
            logger.bind(task_id=state.get("task_id"), llm_node=self.name).info(
                "llm json node requested provider={} model={}",
                self.llm_config.provider,
                self.llm_config.model,
            )
            response = self._client.complete(
                LLMRequest(
                    model=self.llm_config.model,
                    temperature=self.llm_config.temperature,
                    response_format=self.response_model,
                    metadata={"node": self.name},
                    messages=[
                        LLMMessage(role="system", content=self.system_prompt),
                        LLMMessage(role="user", content=self.build_prompt(state, context)),
                    ],
                )
            )
            _record_llm_usage(state, self.name, response)
            if response.parsed is None:
                raise ValueError("OpenAI structured parser returned no parsed payload")
            data = _parsed_payload_to_dict(response.parsed)
            if not isinstance(data, dict):
                raise ValueError("LLM response is not a JSON object")
            raw_user_update = _clean_user_update(data.get("user_update"))
            if self.normalize is not None:
                data = self.normalize(data, state, context)
            if not isinstance(data, dict):
                raise ValueError("normalized LLM response is not a JSON object")
            if raw_user_update and not _clean_user_update(data.get("user_update")):
                data["user_update"] = raw_user_update
            data.setdefault("source", "llm")
            data.setdefault("llm_node", self.name)
            user_update = _clean_user_update(data.get("user_update"))
            if user_update:
                data["user_update"] = user_update
                _append_user_update(state, self.name, user_update)
            return data
        except Exception as exc:
            _record_llm_error(state, self.name, exc)
            logger.bind(task_id=state.get("task_id"), llm_node=self.name).opt(
                exception=self.raise_on_error
            ).warning(
                "llm json node failed error={} fallback_enabled={}",
                exc,
                self.fallback is not None and not self.raise_on_error,
            )
            if self.raise_on_error or self.fallback is None:
                raise
            fallback_data = self._fallback(state, context)
            fallback_data.setdefault("source", "fallback")
            fallback_data.setdefault("llm_node", self.name)
            fallback_data.setdefault("fallback_reason", str(exc))
            fallback_data.setdefault("llm_error", _llm_error_payload(self.name, exc))
            return fallback_data

    def _fallback(self, state: AgentState, context: dict[str, Any]) -> dict[str, Any]:
        data = self.fallback(state, context)
        if not isinstance(data, dict):
            raise ValueError(f"{self.name} fallback must return a dict")
        return dict(data)


def _parsed_payload_to_dict(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return dict(parsed)
    raise ValueError(f"unsupported parsed payload type: {type(parsed).__name__}")


def _clean_user_update(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:240]


def _append_user_update(state: AgentState, source: str, message: str) -> None:
    update = {
        "source": source,
        "message": message,
        "level": "info",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shown": False,
    }
    updates = state.get("user_updates", [])
    if not isinstance(updates, list):
        updates = []
    if updates and isinstance(updates[-1], dict):
        previous = str(updates[-1].get("message") or "").strip()
        previous_source = str(updates[-1].get("source") or "").strip()
        if previous == message and previous_source == source:
            state["last_user_update"] = updates[-1]
            return
    update["shown"] = emit_user_update(update)
    updates = updates + [update]
    state["user_updates"] = updates
    state["last_user_update"] = update


def _record_llm_usage(state: AgentState, node: str, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if not isinstance(usage, dict) or not usage:
        return
    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    model = str(getattr(response, "model", "") or "").strip()
    item = {
        "node": node,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    calls = state.get("llm_calls")
    if not isinstance(calls, list):
        calls = []
    state["llm_calls"] = calls + [item]

    totals = state.get("llm_token_usage")
    if not isinstance(totals, dict):
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            "by_node": {},
        }
    by_node = totals.get("by_node")
    if not isinstance(by_node, dict):
        by_node = {}
    node_usage = by_node.get(node)
    if not isinstance(node_usage, dict):
        node_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        }
    for target in (totals, node_usage):
        target["prompt_tokens"] = _safe_int(target.get("prompt_tokens")) + prompt_tokens
        target["completion_tokens"] = _safe_int(target.get("completion_tokens")) + completion_tokens
        target["total_tokens"] = _safe_int(target.get("total_tokens")) + total_tokens
        target["request_count"] = _safe_int(target.get("request_count")) + 1
    by_node[node] = node_usage
    totals["by_node"] = by_node
    state["llm_token_usage"] = totals


def _record_llm_error(state: AgentState, node: str, exc: Exception) -> None:
    """
        捕获 api 的 error
    """
    errors = state.get("llm_errors")
    if not isinstance(errors, list):
        errors = []
    state["llm_errors"] = errors + [_llm_error_payload(node, exc)]


def _llm_error_payload(node: str, exc: Exception) -> dict[str, Any]:
    text = str(exc)
    return {
        "node": node,
        "type": exc.__class__.__name__,
        "category": _classify_llm_error(text),
        "message": text[:1200],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _classify_llm_error(message: str) -> str:
    text = message.lower()
    if any(keyword in text for keyword in ("arrearage", "overdue", "insufficient", "quota", "billing")):
        return "billing_or_quota"
    if "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "api key" in text or "unauthorized" in text or "access denied" in text:
        return "auth_or_access"
    if "timeout" in text:
        return "timeout"
    return "unknown"
