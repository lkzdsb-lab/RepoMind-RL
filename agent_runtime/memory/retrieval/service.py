"""Read-only orchestration from task state to bounded long-term-memory context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from collections.abc import Mapping
from typing import Any

from agent_runtime.memory.archive.repository import repository_id
from agent_runtime.memory.catalog import SQLiteMemoryCatalog
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.interfaces import MemoryCatalog, MemoryDocumentStore
from agent_runtime.memory.domain.models import (
    MemoryQuery,
    MemoryStatus,
    MemoryType,
)
from agent_runtime.memory.retrieval.models import (
    MemoryRetrievalBatch,
    RetrievedMemory,
)
from agent_runtime.memory.retrieval.planner import RuleBasedMemoryQueryPlanner
from agent_runtime.memory.retrieval.renderer import MemoryContextRenderer
from agent_runtime.memory.retrieval.scope import (
    ScopeContext, ScopeContextBuilder, ScopeMatcher, SCOPE_POLICY_VERSION,
)
from agent_runtime.memory.retrieval.channels import (
    ChannelResult, KeywordRetriever, SemanticRetriever, validated_candidates,
)
from agent_runtime.memory.semantic.embeddings import OpenAICompatibleEmbeddingClient
from agent_runtime.memory.semantic.sqlite import SQLiteMemorySemanticIndex
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
        mode: str = "keyword",
        semantic_retriever: SemanticRetriever | None = None,
        semantic_error: str = "",
        rrf_k: int = 60,
        keyword_weight: float = 1.0,
        semantic_weight: float = 1.0,
        scope_matched_weight: float = 1.1,
        scope_unknown_weight: float = 0.5,
        scope_unknown_limit: int = 2,
    ) -> None:
        self._catalog = catalog
        self._document_store = document_store
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("unsupported memory retrieval mode")
        self.mode = mode
        self.keyword_retriever = KeywordRetriever(catalog)
        self.semantic_retriever = semantic_retriever
        self.semantic_error = semantic_error
        self.rrf_k = max(1, int(rrf_k))
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.repo_path = str(repo_path or ".")
        self.repo_id = repository_id(self.repo_path)
        self.scope_builder = ScopeContextBuilder(self.repo_path, self.repo_id)
        self.scope_matcher = ScopeMatcher()
        self.scope_matched_weight = scope_matched_weight
        self.scope_unknown_weight = scope_unknown_weight
        self.scope_unknown_limit = max(0, int(scope_unknown_limit))
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
        semantic_retriever = None
        semantic_error = ""
        if mode != "keyword" and getattr(config, "long_term_memory_retrieval_enabled", True):
            try:
                embedding = config.memory_embedding_config
                if embedding.provider in {"", "disabled", "none"}:
                    raise ValueError("embedding provider disabled")
                timeout = getattr(config, "long_term_memory_semantic_timeout", 5.0)
                client = OpenAICompatibleEmbeddingClient(
                    replace(embedding, timeout=timeout, batch_size=max(8, embedding.batch_size)),
                    max_retries=0,
                )
                semantic_retriever = SemanticRetriever(
                    SQLiteMemorySemanticIndex.from_config(config, embedding_client=client),
                    timeout=timeout,
                    min_score=getattr(config, "long_term_memory_semantic_min_score", 0.5),
                    candidate_limit=getattr(config, "long_term_memory_semantic_candidates", 24),
                    cache_size=getattr(config, "long_term_memory_query_cache_size", 256),
                )
            except Exception as exc:
                semantic_error = str(exc)
        return cls(
            mode=mode, semantic_retriever=semantic_retriever, semantic_error=semantic_error,
            rrf_k=getattr(config, "long_term_memory_rrf_k", 60),
            keyword_weight=getattr(config, "long_term_memory_keyword_weight", 1.0),
            semantic_weight=getattr(config, "long_term_memory_semantic_weight", 1.0),
            scope_matched_weight=getattr(config, "long_term_memory_scope_matched_weight", 1.1),
            scope_unknown_weight=getattr(config, "long_term_memory_scope_unknown_weight", 0.5),
            scope_unknown_limit=getattr(config, "long_term_memory_scope_unknown_limit", 2),
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
        scope_context = self.scope_builder.build(state)
        revision = self.revision(state, scope_context)
        if not force and revision == str(state.get("long_term_memory_revision") or ""):
            return None
        refresh_count = int(state.get("long_term_memory_refresh_count", 0))
        if not force and refresh_count >= self.max_refreshes:
            return None
        return self.retrieve(state, phase=phase, revision=revision, scope_context=scope_context)

    def revision(self, state: AgentState, scope_context: ScopeContext | None = None) -> str:
        """ 根据当前 state 计算指纹来"""
        payload = {
            "scope_context": (scope_context or self.scope_builder.build(state)).to_dict(),
            "scope_policy": [SCOPE_POLICY_VERSION, self.scope_matched_weight,
                             self.scope_unknown_weight, self.scope_unknown_limit],
            "mode": self.mode,
            "max_context_chars": self.max_context_chars,
            "fusion": [self.rrf_k, self.keyword_weight, self.semantic_weight],
            "semantic": (
                [self.semantic_retriever.index.embedding_client.model_fingerprint,
                 self.semantic_retriever.min_score, self.semantic_retriever.candidate_limit,
                 self.semantic_retriever.timeout]
                if self.semantic_retriever else None
            ),
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
        scope_context: ScopeContext | None = None,
    ) -> MemoryRetrievalBatch:
        scope_context = scope_context or self.scope_builder.build(state)
        revision = revision or self.revision(state, scope_context)
        # 从 analyze task 后的 state 中生成 queries
        queries = tuple(self.planner.plan(state, scope_context))
        warnings: list[str] = []
        statuses = () if self.include_draft else (MemoryStatus.VERIFIED,)
        requests = [(planned.label, MemoryQuery(
            text=planned.text, repo_id=self.repo_id,
            memory_types=tuple(MemoryType(value) for value in planned.memory_types),
            statuses=statuses, tags=planned.tags,
            limit=min(100, self.retrieval_limit * 3),
        )) for planned in queries]
        channels: dict[str, ChannelResult] = {}
        if self.mode != "semantic":
            channels["keyword"] = self.keyword_retriever.retrieve(requests)
        if self.mode != "keyword":
            channels["semantic"] = (
                self.semantic_retriever.retrieve(requests) if self.semantic_retriever else
                ChannelResult(warnings=["semantic unavailable: " + self.semantic_error],
                              diagnostics={"status": "unavailable"})
            )
        for result in channels.values():
            warnings.extend(result.warnings)
        ordered, documents, rejected = validated_candidates(
            channels, self._document_store, mode=self.mode, min_keyword_score=self.min_score,
            rrf_k=self.rrf_k, semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight, warnings=warnings,
        )
        selected: list[RetrievedMemory] = []
        hit_records: list[dict[str, Any]] = []
        # 判断 scope，调整选择顺序
        for rank, (memory_id, record) in enumerate(ordered, 1):
            match = self.scope_matcher.match(documents[memory_id].scope, scope_context)
            record["scope_match"] = match
            record["base_score"] = record["score"]
            record["base_rank"] = rank
            record["score"] *= (self.scope_matched_weight if match.status == "matched"
                                else self.scope_unknown_weight)
        # Confirmed memories get first access to global/type/context budgets.
        ordered.sort(key=lambda pair: (pair[1]["scope_match"].status != "matched",
                                      -pair[1]["score"], pair[0]))
        unknown_count = 0
        type_counts = {key: 0 for key in self.type_limits}
        # 检查状态、配额和文件一致性
        for rank, (memory_id, record) in enumerate(ordered, 1):
            score = float(record["score"])
            diagnostic = {
                "memory_id": memory_id,
                "score": score,
                "reasons": list(record["reasons"]),
                "queries": list(record["queries"]),
                "channels": record["channels"],
                "scope_status": record["scope_match"].status,
                "scope_reason": record["scope_match"].reason,
                "matched_by": list(record["scope_match"].matched_by),
                "base_score": record["base_score"],
                "base_rank": record["base_rank"],
                "adjusted_rank": rank,
                "selected": False,
                "skip_reason": "",
            }
            try:
                document = documents[memory_id]
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
            match = record["scope_match"]
            if match.status == "mismatched":
                diagnostic["skip_reason"] = "scope_mismatch"
                hit_records.append(diagnostic)
                continue
            if match.status == "unknown":
                if document.memory_type == MemoryType.PREFERENCE:
                    diagnostic["skip_reason"] = "preference_scope_unconfirmed"
                    hit_records.append(diagnostic)
                    continue
                if unknown_count >= self.scope_unknown_limit:
                    diagnostic["skip_reason"] = "unknown_scope_quota"
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
                scope_status=match.status,
                scope_reason=match.reason,
                scope_matched_by=match.matched_by,
            )
            selected.append(retrieved)
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            unknown_count += int(match.status == "unknown")
            diagnostic["selected"] = True
            hit_records.append(diagnostic)
        # 生成有长度限制的上下文
        context, context_ids = self.renderer.render_with_ids(
            selected,
            audience="all",
            max_chars=self.max_context_chars,
        )
        # 按节点分发，并记录使用情况，为 analyzer、skill、code_search、planner、action 等节点生成各自允许使用的记忆上下文
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
            hits=tuple((hit_records + rejected)[:100]),
            memories=tuple(selected),
            context=context,
            sections=sections,
            section_memory_ids=section_memory_ids,
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics={"mode": self.mode, "scope_context": scope_context.to_dict(),
                         "scope_policy": SCOPE_POLICY_VERSION, "channels": {
                name: result.diagnostics for name, result in channels.items()
            }},
        )


def _status_allowed(status: MemoryStatus, *, include_draft: bool) -> bool:
    if status == MemoryStatus.VERIFIED:
        return True
    if include_draft and status in {MemoryStatus.DRAFT, MemoryStatus.NEEDS_REVIEW}:
        return True
    return False
