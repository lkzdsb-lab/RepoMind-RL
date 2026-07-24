"""LLM-assisted memory query planning and reranking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.memory.cards import MemoryContextPack, MemorySearchResult
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import MemoryQueryPlanResponse, MemoryRerankResponse
from prompts.templates import load_prompt, render_prompt
from utils import _clamp_float


@dataclass
class MemoryQueryPlan:
    queries: list[str]
    source: str = "disabled"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRerankDecision:
    selected_ids: list[str] = field(default_factory=list)
    source: str = "disabled"
    rationale: str = ""
    selections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryQueryPlanner(Protocol):
    def plan(self, state: AgentState) -> MemoryQueryPlan:
        ...


class MemoryReranker(Protocol):
    def rerank(
        self,
        state: AgentState,
        query_plan: MemoryQueryPlan,
        candidates: MemoryContextPack,
    ) -> tuple[MemoryContextPack, MemoryRerankDecision]:
        ...


@dataclass
class DisabledMemoryQueryPlanner:
    def plan(self, state: AgentState) -> MemoryQueryPlan:
        query = _base_memory_query(state)
        return MemoryQueryPlan(
            queries=[query] if query else [],
            source="disabled",
            rationale="LLM memory query planning is disabled.",
        )


@dataclass
class LLMMemoryQueryPlanner:
    llm_config: LLMConfig
    max_queries: int = 4

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="memory_query_planner",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/memory_query_planner.md"),
            build_prompt=_memory_query_prompt,
            fallback=None,
            response_model=MemoryQueryPlanResponse,
            normalize=self._normalize,
            raise_on_error=True,
        )

    def plan(self, state: AgentState) -> MemoryQueryPlan:
        data = self.node.run(state, {"max_queries": self.max_queries})
        queries = [str(item).strip() for item in data.get("queries", []) if str(item).strip()]
        return MemoryQueryPlan(
            queries=queries[: self.max_queries],
            source="llm",
            rationale=str(data.get("rationale", "")).strip(),
        )

    def _normalize(
        self,
        data: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        max_queries = int(context.get("max_queries") or self.max_queries)
        raw_queries = data.get("queries")
        if not isinstance(raw_queries, list):
            raise ValueError("memory query planner response missing queries list")
        queries: list[str] = []
        for item in raw_queries:
            query = " ".join(str(item).split())
            if query and query not in queries:
                queries.append(query[:300])
            if len(queries) >= max_queries:
                break
        return {
            "queries": queries,
            "rationale": str(data.get("rationale", "")).strip()[:500],
        }


@dataclass
class DisabledMemoryReranker:
    selected_limit: int = 12

    def rerank(
        self,
        state: AgentState,
        query_plan: MemoryQueryPlan,
        candidates: MemoryContextPack,
    ) -> tuple[MemoryContextPack, MemoryRerankDecision]:
        selected = _select_top_candidates(candidates, self.selected_limit)
        return selected, MemoryRerankDecision(
            selected_ids=[result.card.memory_id for result in selected.all_results()],
            source="disabled",
            rationale="LLM memory reranking is disabled.",
        )


@dataclass
class LLMMemoryReranker:
    llm_config: LLMConfig
    selected_limit: int = 12
    candidate_limit: int = 24

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="memory_reranker",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/memory_reranker.md"),
            build_prompt=_memory_rerank_prompt,
            fallback=None,
            response_model=MemoryRerankResponse,
            normalize=self._normalize,
            raise_on_error=True,
        )

    def rerank(
        self,
        state: AgentState,
        query_plan: MemoryQueryPlan,
        candidates: MemoryContextPack,
    ) -> tuple[MemoryContextPack, MemoryRerankDecision]:
        candidate_results = _top_results(candidates, self.candidate_limit)
        if not candidate_results:
            return MemoryContextPack(), MemoryRerankDecision(source="llm", rationale="No memory candidates.")

        context = {
            "query_plan": query_plan,
            "candidates": candidate_results,
            "selected_limit": self.selected_limit,
        }
        data = self.node.run(state, context)
        selections = data.get("selected", [])
        if not isinstance(selections, list):
            raise ValueError("memory reranker response missing selected list")

        available = {result.card.memory_id: result for result in candidate_results}
        selected_ids: list[str] = []
        selected_payloads: list[dict[str, Any]] = []
        for item in selections:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", "")).strip()
            if memory_id not in available or memory_id in selected_ids:
                continue
            selected_ids.append(memory_id)
            selected_payloads.append(
                {
                    "memory_id": memory_id,
                    "relevance": _clamp_float(item.get("relevance"), 0.5, "invalid memory relevance from LLM"),
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
            if len(selected_ids) >= self.selected_limit:
                break

        selected_pack = _pack_from_results([available[memory_id] for memory_id in selected_ids])
        return selected_pack, MemoryRerankDecision(
            selected_ids=selected_ids,
            source="llm",
            rationale=str(data.get("rationale", "")).strip(),
            selections=selected_payloads,
        )

    def _normalize(
        self,
        data: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_ids = {
            result.card.memory_id
            for result in context.get("candidates", [])
            if isinstance(result, MemorySearchResult)
        }
        raw_selected = data.get("selected")
        if not isinstance(raw_selected, list):
            raise ValueError("memory reranker response missing selected list")
        selected: list[dict[str, Any]] = []
        for item in raw_selected:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", "")).strip()
            if memory_id not in candidate_ids:
                continue
            selected.append(
                {
                    "memory_id": memory_id,
                    "relevance": _clamp_float(item.get("relevance"), 0.5, "invalid memory relevance from LLM"),
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
            if len(selected) >= self.selected_limit:
                break
        if raw_selected and not selected:
            raise ValueError("memory reranker selected only unknown memory IDs")
        return {"selected": selected}


def merge_memory_packs(packs: list[MemoryContextPack]) -> MemoryContextPack:
    merged = MemoryContextPack()
    for pack in packs:
        for attr in ("short_term", "mid_term", "long_term", "skill"):
            values = getattr(merged, attr)
            values.extend(_dedupe_results(values, getattr(pack, attr)))
    for attr in ("short_term", "mid_term", "long_term", "skill"):
        values = getattr(merged, attr)
        values.sort(key=lambda item: item.score, reverse=True)
    return merged


def _base_memory_query(state: AgentState) -> str:
    analysis = state.get("task_analysis") or {}
    parts = [
        str(state.get("title", "")),
        str(state.get("description", "")),
    ]
    if isinstance(analysis, dict):
        parts.extend(
            [
                str(analysis.get("task_category", "")),
                " ".join(str(item) for item in _list_values(analysis.get("entities"))[:8]),
                " ".join(str(item) for item in _list_values(analysis.get("search_hints"))[:8]),
            ]
        )
    return " ".join(part for part in parts if part).strip()


def _memory_query_prompt(state: AgentState, context: dict[str, Any]) -> str:
    max_queries = int(context.get("max_queries") or 4)
    return render_prompt(
        "user/memory_query_planner.md",
        max_queries=max_queries,
        title=state.get("title", ""),
        description=state.get("description", ""),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False),
        selected_skills=json.dumps(state.get("selected_skills", []), ensure_ascii=False),
        current_step=state.get("current_step", ""),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
    )


def _memory_rerank_prompt(state: AgentState, context: dict[str, Any]) -> str:
    query_plan = context.get("query_plan")
    queries = query_plan.queries if isinstance(query_plan, MemoryQueryPlan) else []
    candidates = context.get("candidates") or []
    selected_limit = int(context.get("selected_limit") or 12)
    candidate_payload = [_candidate_payload(result) for result in candidates]
    return render_prompt(
        "user/memory_reranker.md",
        selected_limit=selected_limit,
        title=state.get("title", ""),
        description=state.get("description", ""),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False),
        queries=json.dumps(queries, ensure_ascii=False),
        candidates=json.dumps(candidate_payload, ensure_ascii=False),
    )


def _candidate_payload(result: MemorySearchResult) -> dict[str, Any]:
    card = result.card
    return {
        "memory_id": card.memory_id,
        "tier": card.tier,
        "source": result.source,
        "type": card.type,
        "status": card.status,
        "score": result.score,
        "trigger": card.trigger[:240],
        "content": card.content[:900],
        "tags": card.tags[:12],
        "skill_name": card.skill_name,
        "reward_credit": card.reward_credit,
        "last_used_at": card.last_used_at,
    }


def _select_top_candidates(candidates: MemoryContextPack, limit: int) -> MemoryContextPack:
    return _pack_from_results(_top_results(candidates, limit))


def _top_results(candidates: MemoryContextPack, limit: int) -> list[MemorySearchResult]:
    results = candidates.all_results()
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def _pack_from_results(results: list[MemorySearchResult]) -> MemoryContextPack:
    pack = MemoryContextPack()
    for result in results:
        attr = _pack_attr(result)
        getattr(pack, attr).append(result)
    return pack


def _pack_attr(result: MemorySearchResult) -> str:
    if result.card.tier == "short_term" or result.source == "short_term":
        return "short_term"
    if result.card.tier == "long_term" or result.source == "long_term":
        return "long_term"
    if result.card.tier == "skill" or result.source == "skill":
        return "skill"
    return "mid_term"


def _dedupe_results(
    existing: list[MemorySearchResult],
    incoming: list[MemorySearchResult],
) -> list[MemorySearchResult]:
    by_id = {result.card.memory_id: result for result in existing}
    deduped: list[MemorySearchResult] = []
    for result in incoming:
        current = by_id.get(result.card.memory_id)
        if current is None:
            by_id[result.card.memory_id] = result
            deduped.append(result)
            continue
        if result.score > current.score:
            current.score = result.score
            current.card = result.card
            current.source = result.source
    return deduped


def _list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]

