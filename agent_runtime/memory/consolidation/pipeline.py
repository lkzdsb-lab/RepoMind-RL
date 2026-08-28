"""Replayable offline consolidation from Task Archive to Markdown memory."""

from __future__ import annotations

from agent_runtime.memory.consolidation.candidates import RuleBasedCandidateBuilder
from agent_runtime.memory.consolidation.evidence import ArchiveEvidenceCollector
from agent_runtime.memory.consolidation.extractors import MemoryExtractor
from agent_runtime.memory.consolidation.policy import (
    MemoryDocumentBuilder,
    save_with_exact_dedup,
)
from agent_runtime.memory.consolidation.runs import ConsolidationRunStore
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.models import ConsolidationResult, MemoryDocument


class ConsolidationPipeline:
    def __init__(
        self,
        *,
        collector: ArchiveEvidenceCollector,
        candidate_builder: RuleBasedCandidateBuilder,
        extractor: MemoryExtractor,
        document_builder: MemoryDocumentBuilder,
        document_store: MarkdownMemoryDocumentStore,
        run_store: ConsolidationRunStore,
        pipeline_version: str = "consolidation-v1",
    ) -> None:
        self.collector = collector
        self.candidate_builder = candidate_builder
        self.extractor = extractor
        self.document_builder = document_builder
        self.document_store = document_store
        self.run_store = run_store
        self.pipeline_version = pipeline_version

    def consolidate(self, task_id: str) -> ConsolidationResult:
        existing = self.run_store.load(task_id, self.pipeline_version)
        if existing is not None:
            payload = self.run_store.read_artifact(
                task_id, self.pipeline_version, "documents"
            )
            if isinstance(payload, dict):
                for raw in payload.get("documents", []):
                    if not isinstance(raw, dict):
                        continue
                    document = MemoryDocument.from_dict(raw)
                    if self.document_store.get(document.memory_id) is None:
                        self.document_store.save(document)
            return existing
        bundle = self.collector.collect(task_id)
        candidates = self.candidate_builder.build(bundle)
        extraction = self.extractor.extract(bundle, candidates)
        documents, rejected, warnings = self.document_builder.build(
            bundle,
            candidates,
            extraction.memories,
            pipeline_version=self.pipeline_version,
        )
        saved = [
            save_with_exact_dedup(self.document_store, document)
            for document in documents
        ]
        if extraction.source == "rule_based_fallback":
            warnings.append("LLM extraction failed; rule-based fallback was used")
        destination = self.run_store.run_path(task_id, self.pipeline_version)
        outcome = ConsolidationResult(
            task_id=task_id,
            pipeline_version=self.pipeline_version,
            status="completed" if not warnings else "partial",
            memory_ids=tuple(document.memory_id for document in saved),
            rejected_candidates=rejected,
            extractor_source=extraction.source,
            warnings=tuple(warnings),
            run_path=destination.as_posix(),
        )
        self.run_store.save(
            outcome,
            {
                "request": {
                    "task_id": task_id,
                    "pipeline_version": self.pipeline_version,
                    "extractor": extraction.source,
                },
                "evidence_bundle": bundle.to_dict(),
                "candidates": {"candidates": [item.to_dict() for item in candidates]},
                "extraction": extraction.to_dict(),
                "documents": {"documents": [item.to_dict() for item in saved]},
            },
        )
        return outcome
