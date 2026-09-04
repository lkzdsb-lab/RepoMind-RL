"""Read-only orchestration from task state to bounded long-term-memory context."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from agent_runtime.memory.archive.repository import repository_id
from agent_runtime.memory.catalog import SQLiteMemoryCatalog
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.interfaces import MemoryCatalog, MemoryDocumentStore
from agent_runtime.memory.domain.models import (
    MemoryHit,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from agent_runtime.memory.retrieval.models import (
    MemoryRetrievalBatch,
    PlannedMemoryQuery,
    RetrievedMemory,
)
from agent_runtime.memory.retrieval.planner import RuleBasedMemoryQueryPlanner
from agent_runtime.memory.retrieval.renderer import MemoryContextRenderer
from model.agent.graph import AgentState


_DEFAULT_TYPE_LIMITS = {
    "preference": 2,
    "semantic": 4,
    "procedural": 2,
    "anti_pattern": 2,
    "episodic": 2,
}


class LongTermMemoryService:
    """The agent-facing capability intentionally exposes no write operations."""

    def __init__(
        self,
        *,
        catalog: MemoryCatalog,
        document_store: MemoryDocumentStore,
        repo_path: str,
        planner: RuleBasedMemoryQueryPlanner | None = None,
        renderer: MemoryContextRenderer | None = None,
        enabled: bool = True,
        retrieval_limit: int = 8,
        max_context_chars: int = 12000,
        min_score: float = 0.1,
        include_draft: bool = True,
        max_refreshes: int = 3,
        type_limits: Mapping[str, Any] | None = None,
    ) -> None:
        self._catalog = catalog
        self._document_store = document_store
        self.repo_path = str(repo_path or ".")
        self.repo_id = repository_id(self.repo_path)
        self.planner = planner or RuleBasedMemoryQueryPlanner()
        self.renderer = renderer or MemoryContextRenderer()
        self.enabled = bool(enabled)
        self.retrieval_limit = max(1, min(100, int(retrieval_limit)))
        self.max_context_chars = max(1000, int(max_context_chars))
        self.min_score = max(0.0, min(1.0, float(min_score)))
        self.include_draft = bool(include_draft)
        self.max_refreshes = max(1, min(10, int(max_refreshes)))
        configured_limits = dict(type_limits or {})
        self.type_limits = {
            memory_type: max(
                0,
                int(configured_limits.get(memory_type, default_limit)),
            )
            for memory_type, default_limit in _DEFAULT_TYPE_LIMITS.items()
        }

    @classmethod
    def from_config(cls, config: Any) -> "LongTermMemoryService":
        mode = str(
            getattr(config, "long_term_memory_retrieval_mode", "keyword")
            or "keyword"
        ).strip().lower()
        if mode != "keyword":
            raise ValueError(
                "long_term_memory_retrieval_mode must be 'keyword' for Phase 5"
            )
        return cls(
            catalog=SQLiteMemoryCatalog.from_config(config),
            document_store=MarkdownMemoryDocumentStore.from_config(config),
            repo_path=getattr(config, "repo_path", "."),
            planner=RuleBasedMemoryQueryPlanner(
                getattr(config, "long_term_memory_retrieval_max_queries", 4)
            ),
            enabled=getattr(config, "long_term_memory_retrieval_enabled", True),
            retrieval_limit=getattr(config, "long_term_memory_retrieval_limit", 8),
            max_context_chars=getattr(
                config, "long_term_memory_retrieval_max_chars", 12000
            ),
            min_score=getattr(config, "long_term_memory_retrieval_min_score", 0.1),
            include_draft=getattr(
                config, "long_term_memory_retrieval_include_draft", True
            ),
            max_refreshes=getattr(
                config, "long_term_memory_retrieval_max_refreshes", 3
            ),
            type_limits=getattr(
                config, "long_term_memory_retrieval_type_limits", _DEFAULT_TYPE_LIMITS
            ),
        )

    def retrieve_if_needed(
        self,
        state: AgentState,
        *,
        phase: str,
        force: bool = False,
    ) -> MemoryRetrievalBatch | None:
        if not self.enabled:
            return None
        revision = self.revision(state)
        if not force and revision == str(state.get("long_term_memory_revision") or ""):
            return None
        refresh_count = int(state.get("long_term_memory_refresh_count", 0))
        if not force and refresh_count >= self.max_refreshes:
            return None
        return self.retrieve(state, phase=phase, revision=revision)

    def revision(self, state: AgentState) -> str:
        """ 根据当前 state 计算指纹来判断是否需要更新长期记忆"""
        payload = {
            "repo_id": self.repo_id,
            "planner": self.planner.fingerprint_payload(state),
            "include_draft": self.include_draft,
            "limit": self.retrieval_limit,
            "min_score": self.min_score,
            "type_limits": self.type_limits,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def retrieve(
        self,
        state: AgentState,
        *,
        phase: str,
        revision: str | None = None,
    ) -> MemoryRetrievalBatch:
        revision = revision or self.revision(state)
        queries = tuple(self.planner.plan(state))
        warnings: list[str] = []
        merged: dict[str, dict[str, Any]] = {}
        statuses = () if self.include_draft else (MemoryStatus.VERIFIED,)
        for planned in queries:
            try:
                hits = self._catalog.keyword_search(
                    MemoryQuery(
                        text=planned.text,
                        repo_id=self.repo_id,
                        memory_types=tuple(
                            MemoryType(value) for value in planned.memory_types
                        ),
                        statuses=statuses,
                        scope_hints=planned.scope_hints,
                        tags=planned.tags,
                        limit=min(100, max(self.retrieval_limit * 3, self.retrieval_limit)),
                    )
                )
            except Exception as exc:
                warnings.append(f"query {planned.label!r} failed: {exc}")
                continue
            for rank, hit in enumerate(hits, start=1):
                current = merged.get(hit.memory_id)
                reasons = tuple(
                    dict.fromkeys(
                        (f"query={planned.label}", f"query_rank={rank}") + hit.reasons
                    )
                )
                if current is None:
                    merged[hit.memory_id] = {
                        "hit": hit,
                        "score": hit.score,
                        "reasons": reasons,
                        "queries": [planned.label],
                    }
                    continue
                current["score"] = max(float(current["score"]), hit.score)
                current["reasons"] = tuple(
                    dict.fromkeys(tuple(current["reasons"]) + reasons)
                )
                if planned.label not in current["queries"]:
                    current["queries"].append(planned.label)

        ordered = sorted(
            merged.items(),
            key=lambda item: (-float(item[1]["score"]), item[0]),
        )
        selected: list[RetrievedMemory] = []
        hit_records: list[dict[str, Any]] = []
        type_counts = {key: 0 for key in self.type_limits}
        for memory_id, record in ordered:
            score = float(record["score"])
            diagnostic = {
                "memory_id": memory_id,
                "score": score,
                "reasons": list(record["reasons"]),
                "queries": list(record["queries"]),
                "selected": False,
                "skip_reason": "",
            }
            if score < self.min_score:
                diagnostic["skip_reason"] = "below_min_score"
                hit_records.append(diagnostic)
                continue
            try:
                document = self._document_store.get(memory_id)
            except Exception as exc:
                diagnostic["skip_reason"] = "invalid_markdown"
                warnings.append(f"memory {memory_id} could not be loaded: {exc}")
                hit_records.append(diagnostic)
                continue
            if document is None:
                diagnostic["skip_reason"] = "missing_markdown"
                warnings.append(f"memory {memory_id} is indexed but Markdown is missing")
                hit_records.append(diagnostic)
                continue
            if not _status_allowed(document.status, include_draft=self.include_draft):
                diagnostic["skip_reason"] = f"status_{document.status.value}"
                hit_records.append(diagnostic)
                continue
            if not _scope_applies(document.scope, state, self.repo_id):
                diagnostic["skip_reason"] = "scope_mismatch"
                hit_records.append(diagnostic)
                continue
            try:
                current_hash = self._document_store.content_hash(document.memory_id)
            except (OSError, KeyError, ValueError) as exc:
                diagnostic["skip_reason"] = "markdown_hash_failed"
                warnings.append(f"memory {memory_id} could not be hashed: {exc}")
                hit_records.append(diagnostic)
                continue
            indexed_hash = str(record["hit"].content_hash or "")
            if not indexed_hash or indexed_hash != current_hash:
                diagnostic["skip_reason"] = "stale_projection"
                warnings.append(
                    f"memory {memory_id} Markdown differs from its Catalog projection"
                )
                hit_records.append(diagnostic)
                continue
            memory_type = document.memory_type.value
            if type_counts.get(memory_type, 0) >= self.type_limits.get(memory_type, 0):
                diagnostic["skip_reason"] = "type_quota"
                hit_records.append(diagnostic)
                continue
            if len(selected) >= self.retrieval_limit:
                diagnostic["skip_reason"] = "global_limit"
                hit_records.append(diagnostic)
                continue
            retrieved = RetrievedMemory(
                memory_id=document.memory_id,
                memory_type=memory_type,
                status=document.status.value,
                title=document.title,
                scope=document.scope.to_dict(),
                knowledge=document.knowledge[:6000],
                applicability=document.applicability[:2000],
                invalidation=document.invalidation[:1000],
                triggers=document.triggers,
                tags=document.tags,
                confidence=document.confidence,
                evidence_strength=document.evidence_strength,
                score=score,
                reasons=tuple(record["reasons"]),
                content_hash=current_hash,
                source_task_id=document.source.task_id,
            )
            selected.append(retrieved)
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            diagnostic["selected"] = True
            hit_records.append(diagnostic)

        context, context_ids = self.renderer.render_with_ids(
            selected,
            audience="all",
            max_chars=self.max_context_chars,
        )
        context_id_set = set(context_ids)
        selected = [item for item in selected if item.memory_id in context_id_set]
        for diagnostic in hit_records:
            if diagnostic["selected"] and diagnostic["memory_id"] not in context_id_set:
                diagnostic["selected"] = False
                diagnostic["skip_reason"] = "context_budget"
        rendered_sections = {
            audience: self.renderer.render_with_ids(
                selected,
                audience=audience,
                max_chars=self.max_context_chars,
            )
            for audience in ("analyzer", "skill", "code_search", "planner", "action")
        }
        sections = {key: value[0] for key, value in rendered_sections.items()}
        section_memory_ids = {
            key: value[1] for key, value in rendered_sections.items()
        }
        return MemoryRetrievalBatch(
            phase=str(phase or "retrieve")[:80],
            revision=revision,
            queries=queries,
            hits=tuple(hit_records[:100]),
            memories=tuple(selected),
            context=context,
            sections=sections,
            section_memory_ids=section_memory_ids,
            warnings=tuple(dict.fromkeys(warnings)),
        )


def _status_allowed(status: MemoryStatus, *, include_draft: bool) -> bool:
    if status == MemoryStatus.VERIFIED:
        return True
    if include_draft and status in {MemoryStatus.DRAFT, MemoryStatus.NEEDS_REVIEW}:
        return True
    return False


def _scope_applies(scope: Any, state: AgentState, repo_id: str) -> bool:
    if scope.level in {ScopeLevel.GLOBAL, ScopeLevel.USER}:
        return True
    if scope.repo_id != repo_id:
        return False
    if scope.level == ScopeLevel.REPO:
        return True
    analysis = state.get("task_analysis")
    if not isinstance(analysis, dict):
        analysis = {}
    values = [
        str(state.get("title") or ""),
        str(state.get("description") or ""),
        *[str(item) for item in analysis.get("entities", []) or []],
        *[str(item) for item in analysis.get("search_hints", []) or []],
        *[str(item) for item in state.get("candidate_files", []) or []],
    ]
    haystack = " ".join(values).replace("\\", "/").casefold()
    candidate_files = {
        str(item).replace("\\", "/").casefold()
        for item in state.get("candidate_files", []) or []
        if str(item).strip()
    }
    if scope.level == ScopeLevel.MODULE:
        module = scope.module.replace("\\", "/").casefold()
        module_path = module.replace(".", "/")
        return bool(
            module
            and (
                module in haystack
                or module_path in haystack
                or any(path.startswith(module_path + "/") for path in candidate_files)
            )
        )
    if scope.level == ScopeLevel.FILE:
        return any(
            normalized in candidate_files
            or normalized in haystack
            or normalized.rsplit("/", 1)[-1] in haystack
            for normalized in (
                str(path).replace("\\", "/").casefold() for path in scope.files
            )
        )
    if scope.level == ScopeLevel.SYMBOL:
        return any(str(symbol).casefold() in haystack for symbol in scope.symbols)
    return False
