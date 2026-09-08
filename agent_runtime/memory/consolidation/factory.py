"""Composition root for offline memory consolidation."""

from __future__ import annotations

from typing import Any

from agent_runtime.memory.archive import TaskArchiveStoreImpl
from agent_runtime.memory.catalog import SQLiteMemoryCatalog
from agent_runtime.memory.consolidation.candidates import CandidateSeedBuilder
from agent_runtime.memory.consolidation.evidence import ArchiveEvidenceCollector
from agent_runtime.memory.consolidation.evolution import MemoryEvolutionPolicy
from agent_runtime.memory.consolidation.extractors import (
    LLMMemoryExtractor,
    RuleBasedMemoryExtractor,
)
from agent_runtime.memory.consolidation.matching import MemoryMatchFinder
from agent_runtime.memory.consolidation.pipeline import ConsolidationPipeline
from agent_runtime.memory.consolidation.policy import MemoryDocumentBuilder
from agent_runtime.memory.consolidation.projection import MemoryProjectionCoordinator
from agent_runtime.memory.consolidation.relations import LLMMemoryRelationResolver
from agent_runtime.memory.consolidation.resolution import CandidateEvidenceResolver
from agent_runtime.memory.consolidation.runs import ConsolidationRunStore
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.semantic import SQLiteMemorySemanticIndex, build_embedding_client
from config import resolve_llm_config


def build_consolidation_pipeline(
    config: Any,
    *,
    extractor_mode: str | None = None,
) -> ConsolidationPipeline:
    archive = TaskArchiveStoreImpl.from_config(config)
    document_store = MarkdownMemoryDocumentStore.from_config(config)
    rule_extractor = RuleBasedMemoryExtractor()
    mode = str(
        extractor_mode or getattr(config, "memory_extractor_mode", "rule_based")
    ).strip().lower()
    if mode == "llm":
        llm_config = resolve_llm_config(
            config.llm_config,
            config.memory_extractor_llm_config,
        )
        if str(llm_config.provider).strip().lower() in {"", "disabled", "none"}:
            raise ValueError("LLM memory extraction requires an enabled LLM provider")
        if not str(llm_config.model).strip():
            raise ValueError("LLM memory extraction requires a model")
        extractor = LLMMemoryExtractor(
            llm_config,
            fallback=rule_extractor,
        )
    elif mode == "rule_based":
        extractor = rule_extractor
    else:
        raise ValueError(f"unsupported memory extractor mode: {mode}")
    semantic_mode = str(
        getattr(config, "memory_semantic_merge_mode", "disabled")
    ).strip().lower()
    semantic_index = None
    relation_resolver = None
    if semantic_mode in {"observe", "apply"}:
        embedding_client = build_embedding_client(config.memory_embedding_config)
        if embedding_client is None:
            raise ValueError("semantic memory merge requires an embedding provider")
        semantic_index = SQLiteMemorySemanticIndex.from_config(
            config,
            embedding_client=embedding_client,
        )
        relation_config = resolve_llm_config(
            config.llm_config,
            config.memory_relation_llm_config,
        )
        if str(relation_config.provider).strip().lower() in {"", "disabled", "none"}:
            raise ValueError("semantic memory merge requires an enabled relation LLM")
        relation_resolver = LLMMemoryRelationResolver(relation_config)
    catalog = SQLiteMemoryCatalog.from_config(config)
    projection = MemoryProjectionCoordinator(
        document_store=document_store,
        catalog=catalog,
        semantic_index=semantic_index,
    )
    return ConsolidationPipeline(
        collector=ArchiveEvidenceCollector(archive),
        candidate_builder=CandidateSeedBuilder(
            getattr(config, "memory_consolidation_max_candidates", 24)
        ),
        evidence_resolver=CandidateEvidenceResolver(),
        extractor=extractor,
        document_builder=MemoryDocumentBuilder(),
        document_store=document_store,
        projection=projection,
        match_finder=MemoryMatchFinder(
            semantic_index,
            catalog=catalog,
            document_store=document_store,
            top_k=getattr(config, "memory_semantic_top_k", 5),
            min_similarity=getattr(config, "memory_semantic_min_similarity", 0.78),
        ),
        evolution_policy=MemoryEvolutionPolicy(
            min_confidence=getattr(
                config, "memory_semantic_relation_min_confidence", 0.85
            )
        ),
        relation_resolver=relation_resolver,
        run_store=ConsolidationRunStore(
            getattr(config, "consolidation_run_path", ".repomind/consolidation/runs"),
            repo_path=getattr(config, "repo_path", "."),
        ),
        semantic_merge_mode=semantic_mode,
        pipeline_version=getattr(
            config, "consolidation_pipeline_version", "consolidation-v3"
        ),
    )
