"""Reusable JSON-oriented LLM node runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_runtime.llm.llm import LLMClient, build_llm_client
from config import LLMConfig
from loguru import logger
from model.agent.graph import AgentState
from model.llm import LLMMessage, LLMRequest
from pydantic import BaseModel


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
            if response.parsed is None:
                raise ValueError("OpenAI structured parser returned no parsed payload")
            data = _parsed_payload_to_dict(response.parsed)
            if not isinstance(data, dict):
                raise ValueError("LLM response is not a JSON object")
            if self.normalize is not None:
                data = self.normalize(data, state, context)
            if not isinstance(data, dict):
                raise ValueError("normalized LLM response is not a JSON object")
            data.setdefault("source", "llm")
            data.setdefault("llm_node", self.name)
            return data
        except Exception as exc:
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
