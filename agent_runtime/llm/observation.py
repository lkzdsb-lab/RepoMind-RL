"""Observation synthesis stage for runtime tool results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.context.events import latest_tool_event
from agent_runtime.lifecycle.execution_queue import current_execution_item as queue_current_execution_item
from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from ext.tool_summaries import read_file_summaries
from model.agent.graph import AgentState
from model.llm import ObservationResponse
from prompts.templates import load_prompt, render_prompt
from utils import _clamp_float, _clean_string_list


OBSERVATION_STATUSES = {"ok", "error", "inconclusive", "complete"}
OBSERVATION_MAX_CHAR = 260


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
    use_delta: bool = True
    full_state_on_severe: bool = True
    write_threshold: float = 0.35

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="observer",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/observer.md"),
            build_prompt=_observation_prompt,
            fallback=_fallback_observation,
            response_model=ObservationResponse,
            normalize=_normalize_observation,
        )

    def observe(self, state: AgentState) -> dict[str, Any]:
        quick = _skip_observation_when_stable(state, self.use_delta)
        if quick is not None:
            return quick
        return self.node.run(
            state,
            {
                "use_delta": self.use_delta,
                "full_state_on_severe": self.full_state_on_severe,
                "write_threshold": self.write_threshold,
            },
        )


def build_action_limit_observation(
    selected_action: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    blocked = [event for event in events if isinstance(event, dict)]
    blocked_names = [str(event.get("action") or "").strip() for event in blocked if str(event.get("action") or "").strip()]
    blocked_signatures = [
        str(event.get("signature") or event.get("action") or "").strip()
        for event in blocked
        if str(event.get("signature") or event.get("action") or "").strip()
    ]
    window = max(
        [int(event.get("window_size", 0) or 0) for event in blocked if isinstance(event, dict)] or [0]
    )
    facts = []
    for event in blocked[:4]:
        action = str(event.get("action") or "").strip()
        count = int(event.get("count", 0) or 0)
        limit = int(event.get("limit", 0) or 0)
        signature = str(event.get("signature") or action).strip()
        if action:
            facts.append(
                f"{action} blocked at {count}/{limit} in last {window or '?'} steps: {signature}"
            )
    return {
        "type": "llm_observation",
        "latest_tool": "select_action",
        "status": "ok",
        "summary": (
            f"Action limit rerouted selection to `{selected_action}` after blocking repeated actions."
            if selected_action
            else "Action limit blocked repeated actions and forced a different route."
        )[:500],
        "facts": facts,
        "new_findings": [],
        "hypotheses": [],
        "invalidated_hypotheses": [],
        "risks": [],
        "next_actions": [selected_action] if selected_action else [],
        "memory_candidates": [],
        "missing_context": [],
        "next_search_terms": [],
        "confidence": 1.0,
        "source": "action_limit",
        "store": True,
        "delta_score": 0.7,
        "storage_reason": "action_limit_reroute",
        "blocked_actions": blocked_names,
        "blocked_signatures": blocked_signatures,
    }


def _observation_prompt(state: AgentState, context: dict[str, Any]) -> str:
    latest = _latest_tool_call(state)
    delta = _observation_delta(state)
    full_state = _should_use_full_state(
        state,
        delta,
        bool(context.get("full_state_on_severe", True)),
    )
    return render_prompt(
        "user/observer.md",
        observation_mode="full" if full_state else "delta",
        title=state.get("title", ""),
        description=state.get("description", ""),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False) if full_state else "{}",
        current_step=state.get("current_step", ""),
        plan=json.dumps(state.get("plan", []), ensure_ascii=False) if full_state else "[]",
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False) if full_state else "[]",
        test_results_tail=json.dumps(
            state.get("test_results", [])[-2:],
            ensure_ascii=False,
            default=str,
        ),
        patch_summary=state.get("patch_summary") if full_state else "",
        memory_context=str(state.get("memory_context", ""))[:1800] if full_state else "",
        compressed_context=str(state.get("compressed_context", ""))[:1800] if full_state else "",
        latest_context_event=json.dumps(
            _latest_context_event_dict(state),
            ensure_ascii=False,
            default=str,
        ),
        latest_tool_call=json.dumps(_trim_tool_call(latest), ensure_ascii=False, default=str),
        observation_delta=json.dumps(delta, ensure_ascii=False, default=str),
        current_execution=json.dumps(_current_execution_item(state), ensure_ascii=False, default=str),
        recent_observations=json.dumps(_recent_relevant_observations(state), ensure_ascii=False, default=str),
        read_file_context=json.dumps(
            read_file_summaries(state, limit=6, excerpt_chars=2500),
            ensure_ascii=False,
            default=str,
        ),
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
    observation = {
        "type": "llm_observation",
        "event_id": str(data.get("event_id") or _latest_context_event_dict(state).get("event_id") or ""),
        "event_type": str(data.get("event_type") or _latest_context_event_dict(state).get("event_type") or ""),
        "latest_tool": tool[:120],
        "status": status,
        "summary": str(data.get("summary") or "").strip()[:500],
        "facts": _clean_string_list(data.get("facts"), 8, OBSERVATION_MAX_CHAR),
        "new_findings": _clean_string_list(data.get("new_findings"), 8, OBSERVATION_MAX_CHAR),
        "hypotheses": _clean_string_list(data.get("hypotheses"), 6, OBSERVATION_MAX_CHAR),
        "invalidated_hypotheses": _clean_string_list(data.get("invalidated_hypotheses"), 6, OBSERVATION_MAX_CHAR),
        "risks": _clean_string_list(data.get("risks"), 6, OBSERVATION_MAX_CHAR),
        "next_actions": _clean_string_list(data.get("next_actions"), 6, OBSERVATION_MAX_CHAR),
        "memory_candidates": _clean_memory_candidates(data.get("memory_candidates")),
        "missing_context": _clean_string_list(data.get("missing_context"), 6, OBSERVATION_MAX_CHAR),
        "next_search_terms": _clean_string_list(data.get("next_search_terms"), 10, OBSERVATION_MAX_CHAR),
        "confidence": _clamp_float(data.get("confidence"), 0.5, "invalid observer confidence from LLM"),
        "user_update": str(data.get("user_update") or "").strip()[:1000],
    }
    score = _observation_delta_score(state, observation)
    # 可配置的 observe 写入阈值
    threshold = float(context.get("write_threshold", 0.35))
    previous = _last_llm_observation(state)
    store = True
    if previous is not None and _observation_signature(previous) == _observation_signature(observation):
        store = False
    elif score < threshold and not _is_severe_observation(state, observation):
        store = False
    observation["delta_score"] = score
    observation["store"] = store
    if not store:
        observation["storage_reason"] = "small_delta_or_duplicate"
    return observation


def _fallback_observation(state: AgentState, context: dict[str, Any]) -> dict[str, Any]:
    latest = _latest_tool_call(state)
    output = latest.get("output")
    if not isinstance(output, dict):
        output = {}
    status = "error" if output.get("error") else "inconclusive"
    latest_event = _latest_context_event_dict(state)
    facts = []
    if latest_event.get("summary"):
        facts.append(str(latest_event.get("summary")))
    return {
        "type": "llm_observation",
        "event_id": str(latest_event.get("event_id") or ""),
        "event_type": str(latest_event.get("event_type") or ""),
        "latest_tool": str(latest.get("name") or "unknown")[:120],
        "status": status,
        "summary": str(output.get("message") or output.get("error") or "LLM observer unavailable; used tool output directly.")[:500],
        "facts": facts[:8],
        "new_findings": [],
        "hypotheses": [],
        "invalidated_hypotheses": [],
        "risks": [str(output.get("error"))[:260]] if output.get("error") else [],
        "next_actions": [],
        "memory_candidates": [],
        "missing_context": [],
        "next_search_terms": [],
        "confidence": 0.0,
        "store": True,
        "delta_score": 1.0,
    }


def _latest_tool_call(state: AgentState) -> dict[str, Any]:
    calls = state.get("tool_calls") or []
    if not calls:
        return {}
    latest = calls[-1]
    return latest if isinstance(latest, dict) else {}


def _latest_context_event_dict(state: AgentState) -> dict[str, Any]:
    event = latest_tool_event(state)
    return event.to_dict() if event is not None else {}


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


def _trim_value(value: Any, max_chars: int = 2500) -> Any:
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, list):
        return [_trim_value(item) for item in value[:12]]
    if isinstance(value, dict):
        return {str(key): _trim_value(item) for key, item in list(value.items())[:20]}
    return value


def _clean_memory_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        cleaned.append(
            {
                "type": str(item.get("type") or "observation")[:80],
                "content": content[:800],
                "source": str(item.get("source") or "observer")[:80],
            }
        )
        if len(cleaned) >= 6:
            break
    return cleaned


def _observation_delta(state: AgentState) -> dict[str, Any]:
    latest = _latest_tool_call(state)
    output = latest.get("output") if isinstance(latest.get("output"), dict) else {}
    execution = _current_execution_item(state)
    return {
        "latest_tool": str(latest.get("name") or "unknown"),
        "latest_tool_input": _trim_value(latest.get("input"), 1200),
        "latest_tool_output": _trim_value(output, 1600),
        "execution_item": execution,
        "edited_files": list(state.get("edited_files", []) or [])[-5:],
        "verification_stale": bool(state.get("verification_stale", False)),
        "latest_test_result": _trim_value((state.get("test_results") or [])[-1] if state.get("test_results") else {}, 1000),
    }


def _should_use_full_state(
    state: AgentState,
    delta: dict[str, Any],
    full_state_on_severe: bool,
) -> bool:
    if not full_state_on_severe:
        return False
    latest = _latest_tool_call(state)
    output = latest.get("output") if isinstance(latest.get("output"), dict) else {}
    if output.get("fatal") or output.get("needs_user_input"):
        return True
    if output.get("error"):
        return True
    if bool(state.get("verification_stale", False)):
        return True
    execution = delta.get("execution_item")
    if isinstance(execution, dict) and str(execution.get("kind") or "") == "verify":
        return True
    return False


def _skip_observation_when_stable(state: AgentState, use_delta: bool) -> dict[str, Any] | None:
    """ skip"""
    if not use_delta:
        return None
    latest = _latest_tool_call(state)
    name = str(latest.get("name") or "")
    output = latest.get("output") if isinstance(latest.get("output"), dict) else {}
    if name == "read_file" and not output.get("error"):
        return {
            "type": "llm_observation",
            "latest_tool": name,
            "status": "ok",
            "summary": "Read-file delta is small; skipped full observation.",
            "new_findings": [],
            "hypotheses": [],
            "invalidated_hypotheses": [],
            "facts": [],
            "risks": [],
            "next_actions": [],
            "memory_candidates": [],
            "missing_context": [],
            "next_search_terms": [],
            "confidence": 0.0,
            "source": "delta_skip",
            "store": False,
            "delta_score": 0.0,
            "storage_reason": "read_file_small_delta",
        }
    previous = _last_llm_observation(state)
    if previous is not None and _observation_signature(previous) == _observation_signature_from_tool(state):
        return {
            "type": "llm_observation",
            "latest_tool": name or "unknown",
            "status": "ok",
            "summary": "Observation skipped because the latest delta matches the most recent stored observation.",
            "new_findings": [],
            "hypotheses": [],
            "invalidated_hypotheses": [],
            "facts": [],
            "risks": [],
            "next_actions": [],
            "memory_candidates": [],
            "missing_context": [],
            "next_search_terms": [],
            "confidence": 0.0,
            "source": "delta_skip",
            "store": False,
            "delta_score": 0.0,
            "storage_reason": "duplicate_delta",
        }
    return None


def _recent_relevant_observations(state: AgentState) -> list[dict[str, Any]]:
    """ 提取和最近 tool_call 相关的 observation"""
    latest_tool = str(_latest_tool_call(state).get("name") or "")
    relevant: list[dict[str, Any]] = []
    for item in reversed(state.get("llm_observations", []) or []):
        if not isinstance(item, dict):
            continue
        if latest_tool and str(item.get("latest_tool") or "") != latest_tool:
            continue
        relevant.append(
            {
                "latest_tool": str(item.get("latest_tool") or ""),
                "status": str(item.get("status") or ""),
                "summary": str(item.get("summary") or "")[:220],
                "delta_score": item.get("delta_score"),
            }
        )
        if len(relevant) >= 3:
            break
    return list(reversed(relevant))


def _current_execution_item(state: AgentState) -> dict[str, Any]:
    item = queue_current_execution_item(state)
    if isinstance(item, dict):
        return {
            "kind": str(item.get("kind") or ""),
            "status": str(item.get("status") or ""),
            "target_files": [
                str(path).strip()
                for path in item.get("target_files", []) or []
                if str(path).strip()
            ],
        }
    return {}


def _last_llm_observation(state: AgentState) -> dict[str, Any] | None:
    items = state.get("llm_observations", []) or []
    for item in reversed(items):
        if isinstance(item, dict):
            return item
    return None


def _observation_signature(observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "event_id": str(observation.get("event_id") or ""),
            "latest_tool": str(observation.get("latest_tool") or ""),
            "status": str(observation.get("status") or ""),
            "summary": str(observation.get("summary") or ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _observation_signature_from_tool(state: AgentState) -> str:
    latest = _latest_tool_call(state)
    output = latest.get("output") if isinstance(latest.get("output"), dict) else {}
    return json.dumps(
        {
            "event_id": str(_latest_context_event_dict(state).get("event_id") or ""),
            "latest_tool": str(latest.get("name") or ""),
            "status": "error" if output.get("error") else "ok",
            "summary": str(output.get("message") or output.get("error") or "")[:500],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _observation_delta_score(state: AgentState, observation: dict[str, Any]) -> float:
    """ 根据 tool_call 的结果以及最后一个 observation 进行打分"""
    latest = _latest_tool_call(state)
    output = latest.get("output") if isinstance(latest.get("output"), dict) else {}
    score = 0.1
    if output and output.get("error") or observation.get("status") in {"error", "inconclusive"}:
        score += 0.45
    if latest.get("name") in {"apply_code_patch", "run_shell_command", "run_tests"}:
        score += 0.25
    if observation.get("new_findings"):
        score += 0.2
    if observation.get("missing_context"):
        score += 0.1
    previous = _last_llm_observation(state)
    if previous and _observation_signature(previous) != _observation_signature(observation):
        score += 0.15
    return min(1.0, score)


def _is_severe_observation(state: AgentState, observation: dict[str, Any]) -> bool:
    latest = _latest_tool_call(state)
    output = latest.get("output") if isinstance(latest.get("output"), dict) else {}
    if output.get("fatal") or output.get("error"):
        return True
    if observation.get("status") in {"error", "inconclusive"}:
        return True
    execution = _current_execution_item(state)
    return str(execution.get("kind") or "") == "verify"
