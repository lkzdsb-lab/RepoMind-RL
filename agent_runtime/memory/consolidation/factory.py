"""Composition root for offline memory consolidation."""

from __future__ import annotations

from typing import Any

from agent_runtime.memory.archive import TaskArchiveStoreImpl
from agent_runtime.memory.catalog import SQLiteMemoryCatalog
from agent_runtime.memory.consolidation.candidates import CandidateSeedBuilder
from agent_runtime.memory.consolidation.evidence import ArchiveEvidenceCollector
from agent_runtime.memory.consolidation.extractors import (
    LLMMemoryExtractor,
    RuleBasedMemoryExtractor,
)
from agent_runtime.memory.consolidation.pipeline import ConsolidationPipeline
from agent_runtime.memory.consolidation.policy import MemoryDocumentBuilder
from agent_runtime.memory.consolidation.resolution import CandidateEvidenceResolver
from agent_runtime.memory.consolidation.runs import ConsolidationRunStore
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
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
    return ConsolidationPipeline(
        collector=ArchiveEvidenceCollector(archive),
        candidate_builder=CandidateSeedBuilder(
            getattr(config, "memory_consolidation_max_candidates", 24)
        ),
        evidence_resolver=CandidateEvidenceResolver(),
        extractor=extractor,
        document_builder=MemoryDocumentBuilder(),
        document_store=document_store,
        catalog=SQLiteMemoryCatalog.from_config(config),
        run_store=ConsolidationRunStore(
            getattr(config, "consolidation_run_path", ".repomind/consolidation/runs"),
            repo_path=getattr(config, "repo_path", "."),
        ),
        pipeline_version=getattr(
            config, "consolidation_pipeline_version", "consolidation-v2"
        ),
    )
