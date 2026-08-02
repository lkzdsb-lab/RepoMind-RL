"""LLM-assisted codebase context query planning and reranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from agent_runtime.search_query import SearchQueryPlanner
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import CodeContextQueryPlanResponse, CodeContextRerankResponse
from prompts.templates import load_prompt, render_prompt
from utils import _clamp_float, _is_informative_query, _truncate_text
from agent_runtime.codebase_context.models import CodeContextRerankDecision, CodeContextQueryPlan


CONTEXT_LIST_KEYS = (
    "files",
    "symbols",
    "functions",
    "api_routes",
    "db_models",
    "call_graph",
    "test_mappings",
    "embedding_matches",
)

CONTEXT_KINDS = {
    "files": "file",
    "symbols": "symbol",
    "functions": "function",
    "api_routes": "api_route",
    "db_models": "db_model",
    "call_graph": "call_graph",
    "test_mappings": "test_mapping",
    "embedding_matches": "embedding_match",
}


class CodeContextQueryPlanner(Protocol):
    def plan(self, state: AgentState, default_query: str = "") -> CodeContextQueryPlan:
        ...


class CodeContextReranker(Protocol):
    def rerank(
        self,
        state: AgentState,
        query_plan: CodeContextQueryPlan,
        candidates: dict[str, Any],
    ) -> tuple[dict[str, Any], CodeContextRerankDecision]:
        ...


@dataclass
class DisabledCodeContextQueryPlanner:
    default_query: str = "TODO"

    def __post_init__(self) -> None:
        self.query_planner = SearchQueryPlanner(default_query=self.default_query)

    def plan(self, state: AgentState, default_query: str = "") -> CodeContextQueryPlan:
        query = str(default_query or "").strip()
        if not query:
            query = self.query_planner.plan(state).query
        return CodeContextQueryPlan(
            queries=[query] if query else [],
            source="disabled",
            rationale="LLM code context query planning is disabled.",
            default_query=query,
        )


@dataclass
class LLMCodeContextQueryPlanner:
    llm_config: LLMConfig
    max_queries: int = 4

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="code_context_query_planner",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/code_context_query_planner.md"),
            build_prompt=_code_context_query_prompt,
            fallback=None,
            response_model=CodeContextQueryPlanResponse,
            normalize=self._normalize,
            raise_on_error=True,
        )

    def plan(self, state: AgentState, default_query: str = "") -> CodeContextQueryPlan:
        data = self.node.run(
            state,
            {
                "default_query": str(default_query or "").strip(),
                "max_queries": self.max_queries,
            },
        )
        queries = [str(item).strip() for item in data.get("queries", []) if str(item).strip()]
        if not queries:
            raise ValueError("LLM code context query planner returned no queries")
        return CodeContextQueryPlan(
            queries=queries[: self.max_queries],
            source="llm",
            rationale=str(data.get("rationale", "")).strip(),
            default_query=str(default_query or "").strip(),
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
            raise ValueError("code context query planner response missing queries list")
        queries: list[str] = []
        default_query = " ".join(str(context.get("default_query") or "").split())
        if _is_informative_query(default_query):
            queries.append(default_query[:300])
        for item in raw_queries:
            query = " ".join(str(item).split())
            if _is_informative_query(query) and query not in queries:
                queries.append(query[:300])
            if len(queries) >= max_queries:
                break
        if not queries:
            raise ValueError("code context query planner response returned empty queries")
        return {
            "queries": queries,
            "rationale": str(data.get("rationale", "")).strip()[:500],
        }


@dataclass
class DisabledCodeContextReranker:
    selected_limit: int = 12

    def rerank(
        self,
        state: AgentState,
        query_plan: CodeContextQueryPlan,
        candidates: dict[str, Any],
    ) -> tuple[dict[str, Any], CodeContextRerankDecision]:
        candidate_payload = build_code_context_candidates(candidates, self.selected_limit)
        selected_ids = [item["candidate_id"] for item in candidate_payload[: self.selected_limit]]
        selected_context = select_code_context(candidates, selected_ids)
        return selected_context, CodeContextRerankDecision(
            selected_ids=selected_ids,
            source="disabled",
            rationale="LLM code context reranking is disabled.",
        )


@dataclass
class LLMCodeContextReranker:
    llm_config: LLMConfig
    selected_limit: int = 12
    candidate_limit: int = 40

    def __post_init__(self) -> None:
        self.node = LLMJsonNode(
            name="code_context_reranker",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/code_context_reranker.md"),
            build_prompt=_code_context_rerank_prompt,
            fallback=None,
            response_model=CodeContextRerankResponse,
            normalize=self._normalize,
            raise_on_error=True,
        )

    def rerank(
        self,
        state: AgentState,
        query_plan: CodeContextQueryPlan,
        candidates: dict[str, Any],
    ) -> tuple[dict[str, Any], CodeContextRerankDecision]:
        candidate_payload = build_code_context_candidates(candidates, self.candidate_limit)
        if not candidate_payload:
            return {}, CodeContextRerankDecision(source="llm", rationale="No code context candidates.")

        data = self.node.run(
            state,
            {
                "query_plan": query_plan,
                "candidates": candidate_payload,
                "selected_limit": self.selected_limit,
            },
        )
        selections = data.get("selected", [])
        if not isinstance(selections, list):
            raise ValueError("code context reranker response missing selected list")

        available = {item["candidate_id"]: item for item in candidate_payload}
        selected_ids: list[str] = []
        selected_payloads: list[dict[str, Any]] = []
        for item in selections:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "")).strip()
            if candidate_id not in available or candidate_id in selected_ids:
                continue
            selected_ids.append(candidate_id)
            selected_payloads.append(
                {
                    "candidate_id": candidate_id,
                    "kind": available[candidate_id].get("kind", ""),
                    "file_path": available[candidate_id].get("file_path", ""),
                    "relevance": _clamp_float(item.get("relevance"), 0.5, "invalid code context relevance from LLM"),
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
            if len(selected_ids) >= self.selected_limit:
                break

        selected_context = select_code_context(candidates, selected_ids)
        return selected_context, CodeContextRerankDecision(
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
            str(item.get("candidate_id", ""))
            for item in context.get("candidates", [])
            if isinstance(item, dict)
        }
        raw_selected = data.get("selected")
        if not isinstance(raw_selected, list):
            raise ValueError("code context reranker response missing selected list")
        selected: list[dict[str, Any]] = []
        for item in raw_selected:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "")).strip()
            if candidate_id not in candidate_ids:
                continue
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "relevance": _clamp_float(item.get("relevance"), 0.5, "invalid code context relevance from LLM"),
                    "reason": str(item.get("reason", "")).strip()[:300],
                }
            )
            if len(selected) >= self.selected_limit:
                break
        if raw_selected and not selected:
            raise ValueError("code context reranker selected only unknown candidate IDs")
        return {
            "selected": selected,
            "rationale": str(data.get("rationale", "")).strip()[:500],
        }


def merge_code_context_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: [] for key in CONTEXT_LIST_KEYS}
    merged.update(
        {
            "query": "",
            "queries": [],
            "query_results": [],
            "flow": {},
            "metadata": {},
        }
    )
    errors: list[str] = []

    for output in outputs:
        if not isinstance(output, dict):
            continue
        query = str(output.get("query", "")).strip()
        if query and query not in merged["queries"]:
            merged["queries"].append(query)
        if output.get("error"):
            errors.append(str(output.get("error")))
        for key in CONTEXT_LIST_KEYS:
            merged[key] = _dedupe_context_items(
                key,
                merged.get(key, []),
                output.get(key, []),
            )
        if isinstance(output.get("metadata"), dict):
            merged["metadata"].update(output["metadata"])
        if isinstance(output.get("flow"), dict):
            merged.setdefault("flows", []).append({"query": query, "flow": output["flow"]})
            if not merged.get("flow"):
                merged["flow"] = output["flow"]
        merged["query_results"].append(_query_result_summary(output))

    merged["query"] = " | ".join(merged["queries"])
    if errors:
        merged["errors"] = errors
        if not any(merged.get(key) for key in CONTEXT_LIST_KEYS):
            merged["error"] = "; ".join(errors)
    return merged


def build_code_context_candidates(
    context: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in CONTEXT_LIST_KEYS:
        values = context.get(key, [])
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            candidate_id = _candidate_id(key, item, index)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "kind": CONTEXT_KINDS.get(key, key),
                    "source_key": key,
                    "file_path": _candidate_file_path(key, item),
                    "title": _candidate_title(key, item),
                    "payload": _trim_value(item, max_chars=900),
                }
            )
            if len(candidates) >= limit:
                return candidates
    return candidates


def select_code_context(
    context: dict[str, Any],
    selected_ids: list[str],
) -> dict[str, Any]:
    selected_set = set(selected_ids)
    selected: dict[str, Any] = {key: [] for key in CONTEXT_LIST_KEYS}
    selected["selected_ids"] = list(selected_ids)

    for key in CONTEXT_LIST_KEYS:
        values = context.get(key, [])
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            candidate_id = _candidate_id(key, item, index)
            if candidate_id not in selected_set:
                continue
            selected_item = dict(item)
            selected_item["candidate_id"] = candidate_id
            selected[key].append(selected_item)

    if isinstance(context.get("flow"), dict):
        selected["flow"] = context["flow"]
    if isinstance(context.get("metadata"), dict):
        selected["metadata"] = context["metadata"]
    return selected


def _code_context_query_prompt(state: AgentState, context: dict[str, Any]) -> str:
    max_queries = int(context.get("max_queries") or 4)
    return render_prompt(
        "user/code_context_query_planner.md",
        max_queries=max_queries,
        default_query=context.get("default_query", ""),
        title=state.get("title", ""),
        description=state.get("description", ""),
        project_profile=json.dumps(state.get("project_profile", {}), ensure_ascii=False),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False),
        selected_skills=json.dumps(state.get("selected_skills", []), ensure_ascii=False),
        skill_context=_truncate_text(json.dumps(state.get("skill_context", []), ensure_ascii=False), 1400),
        memory_context=_truncate_text(str(state.get("memory_context", "")), 3000),
        current_step=state.get("current_step", ""),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
    )


def _code_context_rerank_prompt(state: AgentState, context: dict[str, Any]) -> str:
    query_plan = context.get("query_plan")
    queries = query_plan.queries if isinstance(query_plan, CodeContextQueryPlan) else []
    candidates = context.get("candidates") or []
    selected_limit = int(context.get("selected_limit") or 12)
    return render_prompt(
        "user/code_context_reranker.md",
        selected_limit=selected_limit,
        title=state.get("title", ""),
        description=state.get("description", ""),
        project_profile=json.dumps(state.get("project_profile", {}), ensure_ascii=False),
        task_analysis=json.dumps(state.get("task_analysis", {}), ensure_ascii=False),
        selected_skills=json.dumps(state.get("selected_skills", []), ensure_ascii=False),
        queries=json.dumps(queries, ensure_ascii=False),
        candidates=json.dumps(candidates, ensure_ascii=False),
    )


def _dedupe_context_items(
    key: str,
    existing: list[Any],
    incoming: Any,
) -> list[dict[str, Any]]:
    result = [item for item in existing if isinstance(item, dict)]
    if not isinstance(incoming, list):
        return result
    seen = {
        _candidate_id(key, item, index)
        for index, item in enumerate(result)
        if isinstance(item, dict)
    }
    for index, item in enumerate(incoming):
        if not isinstance(item, dict):
            continue
        candidate_id = _candidate_id(key, item, index)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(item)
    return result


def _candidate_id(key: str, item: dict[str, Any], index: int) -> str:
    if key == "files":
        return f"file:{item.get('path', index)}"
    if key == "symbols":
        return f"symbol:{item.get('file_path', '')}:{item.get('line', '')}:{item.get('name', index)}"
    if key == "functions":
        return f"function:{item.get('file_path', '')}:{item.get('start_line', '')}:{item.get('full_name', item.get('name', index))}"
    if key == "api_routes":
        return f"api_route:{item.get('method', '')}:{item.get('path', '')}:{item.get('handler', index)}"
    if key == "db_models":
        return f"db_model:{item.get('file_path', '')}:{item.get('line', '')}:{item.get('name', index)}"
    if key == "call_graph":
        return f"call_graph:{item.get('file_path', '')}:{item.get('line', '')}:{item.get('caller', '')}->{item.get('callee', index)}"
    if key == "test_mappings":
        return f"test_mapping:{item.get('source_path', '')}->{item.get('test_path', index)}"
    if key == "embedding_matches":
        return f"embedding_match:{item.get('doc_id', index)}"
    return f"{key}:{index}"


def _candidate_file_path(key: str, item: dict[str, Any]) -> str:
    if key == "files":
        return str(item.get("path", ""))
    for field_name in ("file_path", "source_path", "test_path"):
        if item.get(field_name):
            return str(item[field_name])
    return ""


def _candidate_title(key: str, item: dict[str, Any]) -> str:
    if key == "files":
        return str(item.get("path", ""))
    if key == "api_routes":
        return f"{item.get('method', '')} {item.get('path', '')}".strip()
    if key == "call_graph":
        return f"{item.get('caller', '')} -> {item.get('callee', '')}".strip()
    if key == "test_mappings":
        return f"{item.get('source_path', '')} -> {item.get('test_path', '')}".strip()
    return str(item.get("full_name") or item.get("name") or item.get("title") or item.get("doc_id") or key)


def _query_result_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": output.get("query", ""),
        "error": output.get("error"),
        "files": len(output.get("files", []) or []),
        "symbols": len(output.get("symbols", []) or []),
        "functions": len(output.get("functions", []) or []),
        "api_routes": len(output.get("api_routes", []) or []),
        "db_models": len(output.get("db_models", []) or []),
        "embedding_matches": len(output.get("embedding_matches", []) or []),
    }


def _trim_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, dict):
        return {str(key): _trim_value(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_trim_value(item, max_chars) for item in value[:12]]
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    return value
