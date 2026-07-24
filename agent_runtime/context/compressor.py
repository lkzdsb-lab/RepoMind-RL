"""Context compression implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.context.assembler import ContextAssembler
from agent_runtime.context.cards import ContextDigest, ContextItem
from agent_runtime.context.distiller import distill_context_events
from agent_runtime.context.events import collect_context_events
from agent_runtime.context.token_counter import estimate_context_tokens
from agent_runtime.llm.llm_nodes import LLMJsonNode
from model.agent.graph import AgentState
from model.llm import ContextCompressionResponse
from config import DebugAgentConfig, LLMConfig, resolve_llm_config
from loguru import logger
from prompts.templates import load_prompt, render_prompt

# 抽象接口
class ContextCompressor(Protocol):
    def compress(self, items: list[ContextItem], state: AgentState) -> ContextDigest:
        ...


@dataclass
class ContextCompressionPolicy:
    enabled: bool = True
    max_context_tokens: int = 32000
    threshold: float = 0.75
    recent_items: int = 8
    min_new_tokens: int = 1200

    def should_compress(self, items: list[ContextItem]) -> bool:
        if not self.enabled:
            return False
        return estimate_context_tokens(items) >= int(self.max_context_tokens * self.threshold)


class RuleBasedContextCompressor:
    def compress(self, items: list[ContextItem], state: AgentState) -> ContextDigest:
        tool_results = [_summarize_tool_call(call) for call in state.get("tool_calls", [])[-12:]]
        trajectory = state.get("trajectory", [])
        completed = [
            f"{step.get('node')}: {step.get('thought')}"
            for step in trajectory[-8:]
            if step.get("node") and step.get("thought")
        ]
        observations = _summarize_observations(state.get("observations", [])[-10:])
        memory_refs = _summarize_memory_refs(state.get("retrieved_memories", {}))
        code_changes = []
        if state.get("patch_summary"):
            code_changes.append(str(state["patch_summary"]))

        latest_error = state.get("error")
        constraints = [
            f"repo_path={state.get('repo_path', '')}",
            f"verification_required={bool(state.get('verification_required', True))}",
            f"verification_capabilities={json.dumps(state.get('verification_capabilities', {}), ensure_ascii=False, default=str)[:800]}",
        ]
        if latest_error:
            constraints.append(f"latest_error={latest_error}")

        return ContextDigest(
            summary=_first_non_empty(
                [
                    f"Task `{state.get('title', '')}` is in step `{state.get('current_step', '')}`.",
                    "Prior context was compressed from runtime state.",
                ]
            ),
            current_goal=" ".join(
                part for part in [state.get("title", ""), state.get("description", "")] if part
            ),
            constraints=constraints,
            decisions=[],
            completed_tasks=completed,
            open_tasks=_open_tasks(state),
            key_observations=observations,
            tool_results=tool_results,
            code_changes=code_changes,
            memory_refs=memory_refs,
            source_item_ids=[item.item_id for item in items],
            compression_method="rule_based",
        )


class LLMContextCompressor:
    def __init__(
        self,
        llm_config: LLMConfig,
        fallback: ContextCompressor | None = None,
    ) -> None:
        self.llm_config = llm_config
        self.fallback = fallback or RuleBasedContextCompressor()
        self.node = LLMJsonNode(
            name="context_compressor",
            llm_config=llm_config,
            system_prompt=load_prompt("system/context_compressor.md"),
            build_prompt=_build_llm_compression_prompt,
            fallback=_context_compression_fallback,
            response_model=ContextCompressionResponse,
            normalize=_normalize_context_compression,
        )

    def compress(self, items: list[ContextItem], state: AgentState) -> ContextDigest:
        fallback_digest = self.fallback.compress(items, state)
        logger.bind(task_id=state.get("task_id")).info(
            "llm context compression requested items={} model={} provider={}",
            len(items),
            self.llm_config.model,
            self.llm_config.provider,
        )
        data = self.node.run(
            state,
            {
                "items": items,
                "fallback_digest": fallback_digest,
            },
        )
        if data.get("source") == "fallback":
            return fallback_digest.with_error(
                str(data.get("fallback_reason") or "llm_context_compression_failed")
            )
        return ContextDigest.from_dict(
            {
                **fallback_digest.to_dict(),
                **data,
                "compression_method": "llm",
                "source_item_ids": fallback_digest.source_item_ids,
            }
        )


class ContextCompressionManager:
    def __init__(
        self,
        policy: ContextCompressionPolicy,
        compressor: ContextCompressor,
    ) -> None:
        self.policy = policy
        self.compressor = compressor
        self.assembler = ContextAssembler(max_tokens=policy.max_context_tokens)

    @classmethod
    def from_config(cls, config: DebugAgentConfig) -> "ContextCompressionManager":
        policy = ContextCompressionPolicy(
            enabled=config.context_compression_enabled,
            max_context_tokens=config.context_max_tokens,
            threshold=config.context_compression_threshold,
            recent_items=config.context_recent_items,
            min_new_tokens=config.context_min_new_tokens,
        )
        mode = (config.context_compressor_mode or "rule_based").strip().lower()
        compressor: ContextCompressor
        if mode == "disabled":
            policy.enabled = False
            compressor = RuleBasedContextCompressor()
            llm_config = config.llm_config
        elif mode == "llm":
            llm_config = resolve_llm_config(
                config.llm_config,
                config.context_compressor_llm_config,
            )
            compressor = LLMContextCompressor(llm_config)
        else:
            llm_config = config.llm_config
            compressor = RuleBasedContextCompressor()
        logger.info(
            "context compression configured enabled={} mode={} provider={} threshold={} max_tokens={}",
            policy.enabled,
            mode,
            llm_config.provider,
            policy.threshold,
            policy.max_context_tokens,
        )
        return cls(policy=policy, compressor=compressor)

    def prepare(self, state: AgentState) -> AgentState:
        items = collect_context_items(state)
        raw_items = [item for item in items if item.item_type != "compressed_context"]
        context_events = collect_context_events(state)
        distilled_events, memory_candidates = distill_context_events(context_events, state)
        assembled = self.assembler.assemble(distilled_events, state)
        token_estimate = estimate_context_tokens(raw_items)
        threshold_tokens = int(self.policy.max_context_tokens * self.policy.threshold)
        context_updates = {
            "context_events": [event.to_dict() for event in context_events],
            "distilled_events": [event.to_dict() for event in distilled_events],
            "working_context": assembled.working_context,
            "archive_context": assembled.archive_context,
            "context_sections": assembled.context_sections,
            "memory_candidates": memory_candidates,
            "compressed_context": _merge_context_text(
                assembled.working_context,
                _render_state_digest(state),
                assembled.archive_context,
            ),
        }
        if not self.policy.enabled:
            logger.bind(task_id=state.get("task_id")).debug(
                "context compression disabled items={} estimated_tokens={}",
                len(items),
                token_estimate,
            )
            return {
                **state,
                **context_updates,
                "context_items": [item.to_dict() for item in items[-self.policy.recent_items :]],
            }
        if token_estimate < threshold_tokens:
            logger.bind(task_id=state.get("task_id")).debug(
                "context compression skipped items={} estimated_tokens={} threshold_tokens={}",
                len(items),
                token_estimate,
                threshold_tokens,
            )
            return {
                **state,
                **context_updates,
                "context_items": [item.to_dict() for item in items[-self.policy.recent_items :]],
            }

        pinned = [item for item in items if item.pinned]
        recent = items[-self.policy.recent_items :]
        recent_ids = {item.item_id for item in recent}
        pinned_ids = {item.item_id for item in pinned}
        already_compressed = set(
            (state.get("context_digest") or {}).get("source_item_ids", [])
        )
        already_compressed.update(_string_set(state.get("compressed_context_item_ids")))
        compressible = [
            item
            for item in items
            if item.item_id not in recent_ids
            and item.item_id not in pinned_ids
            and item.item_id not in already_compressed
            and item.item_type != "compressed_context"
        ]
        if not compressible:
            logger.bind(task_id=state.get("task_id")).debug(
                "context compression skipped; no new compressible items items={}",
                len(items),
            )
            return {
                **state,
                **context_updates,
                "context_items": [item.to_dict() for item in items],
            }
        new_token_estimate = estimate_context_tokens(compressible)
        if state.get("context_digest") and new_token_estimate < self.policy.min_new_tokens:
            logger.bind(task_id=state.get("task_id")).debug(
                "context compression skipped; new compressible tokens below minimum new_tokens={} min_new_tokens={}",
                new_token_estimate,
                self.policy.min_new_tokens,
            )
            return {
                **state,
                **context_updates,
                "compressed_context_item_ids": sorted(already_compressed),
                "context_items": [item.to_dict() for item in pinned + recent],
            }

        logger.bind(task_id=state.get("task_id")).info(
            "context compression started items={} compressible={} estimated_tokens={} new_tokens={} threshold_tokens={}",
            len(items),
            len(compressible),
            token_estimate,
            new_token_estimate,
            threshold_tokens,
        )
        digest = self.compressor.compress(compressible, state)
        compressed_item_ids = sorted(already_compressed.union(digest.source_item_ids))
        logger.bind(task_id=state.get("task_id")).info(
            "context compression completed method={} source_items={} total_source_items={}",
            digest.compression_method,
            len(digest.source_item_ids),
            len(compressed_item_ids),
        )
        return {
            **state,
            **context_updates,
            "context_digest": digest.to_dict(),
            "compressed_context_item_ids": compressed_item_ids,
            "compressed_context": _merge_context_text(
                assembled.working_context,
                digest.render_for_prompt(),
                assembled.archive_context,
            ),
            "context_items": [item.to_dict() for item in pinned + recent],
        }


def collect_context_items(state: AgentState) -> list[ContextItem]:
    items: list[ContextItem] = []
    goal = " ".join(
        part for part in [state.get("title", ""), state.get("description", "")] if part
    )
    if goal:
        items.append(
            ContextItem(
                role="user",
                item_type="task",
                content=goal,
                pinned=True,
                metadata={"source": "task"},
                item_id="task:current",
            )
        )
    if state.get("task_analysis"):
        items.append(
            ContextItem(
                role="system",
                item_type="task_analysis",
                content=json.dumps(state["task_analysis"], ensure_ascii=False, default=str),
                pinned=True,
                metadata={"source": "task_analysis"},
                item_id="task:analysis",
            )
        )
    if state.get("memory_context"):
        items.append(
            ContextItem(
                role="system",
                item_type="memory_context",
                content=str(state["memory_context"]),
                pinned=True,
                metadata={"source": "memory"},
                item_id="memory:context",
            )
        )
    for index, observation in enumerate(state.get("llm_observations", [])):
        items.append(
            ContextItem(
                role="assistant",
                item_type="llm_observation",
                content=json.dumps(
                    _summarize_llm_observation_item(observation),
                    ensure_ascii=False,
                    default=str,
                ),
                metadata={"source": "llm_observation", "tool": observation.get("latest_tool")},
                item_id=f"llm_observation:{index}:{observation.get('latest_tool', 'unknown')}",
            )
        )
    for index, step in enumerate(state.get("trajectory", [])):
        items.append(
            ContextItem(
                role="assistant",
                item_type="trajectory",
                content=json.dumps(_summarize_trajectory_item(step), ensure_ascii=False, default=str),
                metadata={"source": "trajectory", "node": step.get("node")},
                item_id=f"trajectory:{step.get('step_id', index)}",
            )
        )
    for index, call in enumerate(state.get("tool_calls", [])):
        items.append(
            ContextItem(
                role="tool",
                item_type="tool_call",
                content=json.dumps(_summarize_tool_call_item(call), ensure_ascii=False, default=str),
                metadata={"source": "tool_call", "name": call.get("name")},
                item_id=f"tool_call:{index}:{call.get('name', 'unknown')}",
            )
        )
    for index, observation in enumerate(state.get("observations", [])):
        items.append(
            ContextItem(
                role="tool",
                item_type="observation",
                content=json.dumps(_summarize_observation_item(observation), ensure_ascii=False, default=str),
                metadata={"source": "observation", "type": observation.get("type")},
                item_id=f"observation:{index}:{observation.get('type', 'unknown')}",
            )
        )
    if state.get("compressed_context"):
        items.append(
            ContextItem(
                role="system",
                item_type="compressed_context",
                content=str(state["compressed_context"]),
                metadata={"source": "previous_digest"},
                item_id="compressed:current",
            )
        )
    return items


def _summarize_llm_observation_item(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"summary": str(observation)[:500]}
    return {
        "latest_tool": observation.get("latest_tool"),
        "status": observation.get("status"),
        "summary": str(observation.get("summary", ""))[:500],
        "facts": _short_list(observation.get("facts"), 6),
        "risks": _short_list(observation.get("risks"), 4),
        "next_actions": _short_list(observation.get("next_actions"), 6),
    }


def _summarize_trajectory_item(step: Any) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {"summary": str(step)[:500]}
    return {
        "step_id": step.get("step_id"),
        "node": step.get("node"),
        "thought": str(step.get("thought", ""))[:500],
        "action": step.get("action"),
        "action_input": _compact_value(step.get("action_input"), max_chars=800),
        "observation": _summarize_payload(step.get("observation")),
        "created_at": step.get("created_at"),
    }


def _summarize_tool_call_item(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {"summary": str(call)[:500]}
    output = call.get("output") if isinstance(call.get("output"), dict) else {}
    name = str(call.get("name") or "unknown")
    return {
        "name": name,
        "input": _compact_value(call.get("input"), max_chars=800),
        "error": call.get("error") or output.get("error"),
        "output_summary": _tool_output_summary(name, output),
        "output": _summarize_payload(output),
    }


def _summarize_observation_item(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"summary": str(observation)[:500]}
    return {
        "type": observation.get("type"),
        "tool": observation.get("tool"),
        "content": _summarize_payload(observation.get("content")),
    }


def _summarize_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value, max_chars=1000)
    if value.get("error"):
        return {"error": str(value.get("error"))[:1000]}
    if "matches" in value:
        matches = value.get("matches")
        return {
            "query": value.get("query"),
            "match_count": len(matches) if isinstance(matches, list) else 0,
            "sample_matches": _short_list(matches, 5, max_chars=240),
            "exit_code": value.get("exit_code"),
            "command": value.get("command"),
        }
    if "content" in value and "file_path" in value:
        content = str(value.get("content") or "")
        return {
            "file_path": value.get("file_path"),
            "content_chars": len(content),
            "excerpt": content[:1000],
            "truncated": value.get("truncated"),
            "start_line": value.get("start_line"),
            "end_line": value.get("end_line"),
            "total_lines": value.get("total_lines"),
        }
    if "files" in value:
        files = value.get("files")
        return {
            "file_count": len(files) if isinstance(files, list) else 0,
            "sample_files": _short_list(files, 20, max_chars=200),
            "truncated": value.get("truncated"),
            "ignored_dirs": value.get("ignored_dirs"),
        }
    if "diff" in value:
        diff = str(value.get("diff") or "")
        return {
            "diff_lines": len(diff.splitlines()),
            "excerpt": diff[:1200],
        }
    return {
        key: _compact_value(val, max_chars=800)
        for key, val in value.items()
        if key not in {"content", "matches", "files", "diff"}
    }


def _compact_value(value: Any, max_chars: int = 500) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_value(val, max_chars=max_chars) for key, val in value.items()}
    if isinstance(value, list):
        return [_compact_value(item, max_chars=max_chars) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value) if isinstance(value, str) else value
        return text[:max_chars] if isinstance(text, str) else text
    return str(value)[:max_chars]


def _short_list(value: Any, limit: int, max_chars: int = 300) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:max_chars] for item in value[:limit]]


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}


def _render_state_digest(state: AgentState) -> str:
    digest = state.get("context_digest")
    if not isinstance(digest, dict) or not digest:
        return ""
    try:
        return ContextDigest.from_dict(digest).render_for_prompt()
    except TypeError:
        return ""


def _merge_context_text(*parts: str) -> str:
    chunks: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in chunks:
            chunks.append(text)
    return "\n\n".join(chunks)


def _build_llm_compression_prompt(
    state: AgentState,
    context: dict[str, Any],
) -> str:
    items = context.get("items") or []
    fallback_digest = context.get("fallback_digest")
    if not isinstance(fallback_digest, ContextDigest):
        fallback_digest = RuleBasedContextCompressor().compress(items, state)
    item_text = "\n\n".join(
        f"[{idx}] role={item.role} type={item.item_type} pinned={item.pinned}\n{item.content[:4000]}"
        for idx, item in enumerate(items, start=1)
    )
    return render_prompt(
        "user/context_compressor.md",
        fallback_digest=json.dumps(fallback_digest.to_dict(), ensure_ascii=False),
        title=state.get("title", ""),
        description=state.get("description", ""),
        current_step=state.get("current_step", ""),
        status=state.get("status", ""),
        error=state.get("error", ""),
        item_text=item_text,
    )


def _context_compression_fallback(
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    fallback_digest = context.get("fallback_digest")
    if isinstance(fallback_digest, ContextDigest):
        return fallback_digest.to_dict()
    items = context.get("items") or []
    return RuleBasedContextCompressor().compress(items, state).to_dict()


def _normalize_context_compression(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    fallback = _context_compression_fallback(state, context)
    return {
        "summary": str(data.get("summary") or fallback.get("summary") or "").strip()[:1000],
        "current_goal": str(data.get("current_goal") or fallback.get("current_goal") or "").strip()[:600],
        "constraints": _clean_str_list(data.get("constraints"), fallback.get("constraints", []), 12),
        "decisions": _clean_str_list(data.get("decisions"), fallback.get("decisions", []), 12),
        "open_tasks": _clean_str_list(data.get("open_tasks"), fallback.get("open_tasks", []), 12),
        "completed_tasks": _clean_str_list(
            data.get("completed_tasks"),
            fallback.get("completed_tasks", []),
            12,
        ),
        "key_observations": _clean_str_list(
            data.get("key_observations"),
            fallback.get("key_observations", []),
            16,
        ),
        "tool_results": _clean_tool_results(
            data.get("tool_results"),
            fallback.get("tool_results", []),
            16,
        ),
        "code_changes": _clean_str_list(data.get("code_changes"), fallback.get("code_changes", []), 12),
        "memory_refs": _clean_str_list(data.get("memory_refs"), fallback.get("memory_refs", []), 12),
    }


def _clean_str_list(value: Any, fallback: list[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return [str(item) for item in fallback[:limit]]
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:500])
        if len(cleaned) >= limit:
            break
    return cleaned or [str(item) for item in fallback[:limit]]


def _clean_tool_results(value: Any, fallback: list[dict], limit: int) -> list[dict]:
    if not isinstance(value, list):
        return list(fallback)[:limit]
    cleaned: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "name": str(item.get("name", "unknown"))[:120],
                "status": str(item.get("status", "unknown"))[:80],
                "summary": str(item.get("summary", ""))[:500],
            }
        )
        if len(cleaned) >= limit:
            break
    return cleaned or list(fallback)[:limit]


def _summarize_tool_call(call: dict) -> dict:
    output = call.get("output") or {}
    status = "error" if call.get("error") or output.get("error") else "ok"
    return {
        "name": call.get("name", "unknown"),
        "status": status,
        "summary": _tool_output_summary(call.get("name", ""), output),
    }


def _tool_output_summary(name: str, output: dict) -> str:
    if output.get("error"):
        return str(output["error"])
    if name == "search_code":
        return f"{len(output.get('matches', []))} matches for query `{output.get('query', '')}`"
    if name == "read_file":
        return f"read {output.get('file_path', 'unknown file')}"
    if name == "run_tests":
        if output.get("skipped"):
            return f"skipped reason={output.get('reason', '')}"
        return f"exit_code={output.get('exit_code')} command={output.get('command', '')}"
    if name == "git_diff":
        return f"{len(str(output.get('diff', '')).splitlines())} diff lines"
    return f"keys={', '.join(sorted(output.keys()))}"


def _summarize_observations(observations: list[dict]) -> list[str]:
    summaries = []
    for observation in observations:
        kind = observation.get("type", "observation")
        content = observation.get("content", {})
        if isinstance(content, dict):
            summaries.append(f"{kind}: keys={', '.join(sorted(content.keys()))}")
        else:
            summaries.append(f"{kind}: {str(content)[:240]}")
    return summaries


def _summarize_memory_refs(memories: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(memories, dict):
        for tier, values in memories.items():
            if isinstance(values, list) and values:
                refs.append(f"{tier}: {len(values)} memories")
    elif isinstance(memories, list) and memories:
        refs.append(f"retrieved: {len(memories)} memories")
    return refs


def _open_tasks(state: AgentState) -> list[str]:
    open_tasks = []
    if not state.get("test_results"):
        open_tasks.append("Run verification command.")
    if state.get("patch_summary") is None:
        open_tasks.append("Inspect current git diff.")
    if not state.get("memory_written"):
        open_tasks.append("Write reward-gated memory.")
    return open_tasks


def _first_non_empty(values: list[str]) -> str:
    for value in values:
        if value:
            return value
    return ""
