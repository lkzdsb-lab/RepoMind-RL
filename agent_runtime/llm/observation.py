"""Observation synthesis stage for runtime tool results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import ObservationResponse


OBSERVATION_STATUSES = {"ok", "error", "inconclusive", "complete"}


class Observer(Protocol):
    def observe(self, state: AgentState) -> dict[str, Any]:
        ...


@dataclass
class DisabledObserver:
    """No-op observer used when LLM observation is not enabled."""

    def observe(self, state: AgentState) -> dict[str, Any]:
        latest = _latest_tool_call(state)
        return {
            "type": "llm_observation",
            "latest_tool": str(latest.get("name") or "unknown"),
            "status": "disabled",
            "summary": "",
            "new_findings": [],
            "hypotheses": [],
            "missing_context": [],
            "next_search_terms": [],
            "confidence": 0.0,
            "source": "disabled",
        }


@dataclass
class LLMObserver:
    llm_config: LLMConfig

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="observer",
            llm_config=self.llm_config,
            system_prompt=(
                "You synthesize the latest tool result for a coding agent. "
                "Return only JSON matching the requested schema. "
                "Be factual and do not invent files, test results, or code changes."
            ),
            build_prompt=_observation_prompt,
            fallback=None,
            response_model=ObservationResponse,
            normalize=_normalize_observation,
            raise_on_error=True,
        )

    def observe(self, state: AgentState) -> dict[str, Any]:
        return self.node.run(state)


def _observation_prompt(state: AgentState, context: dict[str, Any]) -> str:
    latest = _latest_tool_call(state)
    return (
        "Return JSON with keys: latest_tool, status, summary, new_findings, hypotheses, "
        "missing_context, next_search_terms, confidence.\n"
        "status must be one of ok, error, inconclusive, complete. "
        "Use short list fields. If uncertain, use empty lists instead of guessing.\n\n"
        f"title={state.get('title', '')}\n"
        f"description={state.get('description', '')}\n"
        f"task_analysis={json.dumps(state.get('task_analysis', {}), ensure_ascii=False)}\n"
        f"current_step={state.get('current_step', '')}\n"
        f"plan={json.dumps(state.get('plan', []), ensure_ascii=False)}\n"
        f"candidate_files={json.dumps(state.get('candidate_files', []), ensure_ascii=False)}\n"
        f"test_results_tail={json.dumps(state.get('test_results', [])[-2:], ensure_ascii=False, default=str)}\n"
        f"patch_summary={state.get('patch_summary')}\n"
        f"memory_context={str(state.get('memory_context', ''))[:1800]}\n"
        f"compressed_context={str(state.get('compressed_context', ''))[:1800]}\n"
        f"latest_tool_call={json.dumps(_trim_tool_call(latest), ensure_ascii=False, default=str)}\n"
    )


def _normalize_observation(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    latest = _latest_tool_call(state)
    tool = str(data.get("latest_tool") or latest.get("name") or "unknown").strip()
    status = str(data.get("status") or "").strip().lower()
    if status not in OBSERVATION_STATUSES:
        raise ValueError(f"invalid observer status from LLM: {status}")
    return {
        "type": "llm_observation",
        "latest_tool": tool[:120],
        "status": status,
        "summary": str(data.get("summary") or "").strip()[:500],
        "new_findings": _clean_list(data.get("new_findings"), 8),
        "hypotheses": _clean_list(data.get("hypotheses"), 6),
        "missing_context": _clean_list(data.get("missing_context"), 6),
        "next_search_terms": _clean_list(data.get("next_search_terms"), 10),
        "confidence": _clamp_float(data.get("confidence")),
    }


def _latest_tool_call(state: AgentState) -> dict[str, Any]:
    calls = state.get("tool_calls") or []
    if not calls:
        return {}
    latest = calls[-1]
    return latest if isinstance(latest, dict) else {}


def _trim_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    output = call.get("output")
    if isinstance(output, dict):
        trimmed_output = {
            key: _trim_value(value)
            for key, value in output.items()
        }
    else:
        trimmed_output = _trim_value(output)
    return {
        "name": call.get("name"),
        "input": call.get("input"),
        "error": call.get("error"),
        "output": trimmed_output,
    }


def _trim_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:2500]
    if isinstance(value, list):
        return [_trim_value(item) for item in value[:12]]
    if isinstance(value, dict):
        return {str(key): _trim_value(item) for key, item in list(value.items())[:20]}
    return value


def _clean_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:260])
        if len(cleaned) >= limit:
            break
    return cleaned


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid observer confidence from LLM: {value}")
    return max(0.0, min(1.0, parsed))
