"""Replayable offline consolidation from Task Archive to durable memory."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from agent_runtime.memory.consolidation.candidates import CandidateSeedBuilder
from agent_runtime.memory.consolidation.evidence import ArchiveEvidenceCollector
from agent_runtime.memory.consolidation.evolution import MemoryEvolutionPolicy
from agent_runtime.memory.consolidation.extractors import MemoryExtractor
from agent_runtime.memory.consolidation.matching import MemoryMatchFinder
from agent_runtime.memory.consolidation.models import MatchedMemory, RelationResolution
from agent_runtime.memory.consolidation.policy import MemoryDocumentBuilder
from agent_runtime.memory.consolidation.projection import MemoryProjectionCoordinator
from agent_runtime.memory.consolidation.resolution import CandidateEvidenceResolver
from agent_runtime.memory.consolidation.runs import ConsolidationRunStore
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.models import ConsolidationResult, MemoryDocument


class MemoryRelationResolver(Protocol):
    def resolve(
        self,
        proposed: MemoryDocument,
        matches: list[MatchedMemory],
    ) -> RelationResolution: ...


class ConsolidationPipeline:
    def __init__(
        self,
        *,
        collector: ArchiveEvidenceCollector,
        candidate_builder: CandidateSeedBuilder,
        evidence_resolver: CandidateEvidenceResolver,
        extractor: MemoryExtractor,
        document_builder: MemoryDocumentBuilder,
        document_store: MarkdownMemoryDocumentStore,
        projection: MemoryProjectionCoordinator,
        match_finder: MemoryMatchFinder,
        evolution_policy: MemoryEvolutionPolicy,
        relation_resolver: MemoryRelationResolver | None,
        run_store: ConsolidationRunStore,
        semantic_merge_mode: str = "disabled",
        pipeline_version: str = "consolidation-v3",
    ) -> None:
        self.collector = collector
        self.candidate_builder = candidate_builder
        self.evidence_resolver = evidence_resolver
        self.extractor = extractor
        self.document_builder = document_builder
        self.document_store = document_store
        self.projection = projection
        self.match_finder = match_finder
        self.evolution_policy = evolution_policy
        self.relation_resolver = relation_resolver
        self.run_store = run_store
        self.semantic_merge_mode = semantic_merge_mode
        self.pipeline_version = pipeline_version

    def consolidate(self, task_id: str) -> ConsolidationResult:
        # 检查当前 task id 的归档 pipeline 是否存在
        existing = self.run_store.load(task_id, self.pipeline_version)
        if existing is not None:
            return self._replay(existing)

        bundle = self.collector.collect(task_id)
        seeds = self.candidate_builder.build(bundle)
        candidates, resolutions = self.evidence_resolver.resolve_all(
            seeds,
            bundle.evidence,
        )
        extraction = self.extractor.extract(bundle, candidates)
        # 先将本次的 md 文件生成保存在内存里
        proposed, rejected, warnings = self.document_builder.build(
            bundle,
            candidates,
            extraction.memories,
            pipeline_version=self.pipeline_version,
        )

        # 后续根据 knowledge 和并相似的 md 文件
        saved_by_id: dict[str, MemoryDocument] = {}
        semantic_audit: list[dict] = []
        relation_audit: list[dict] = []
        mutation_audit: list[dict] = []
        self.projection.retry_pending(warnings)
        for document in sorted(proposed, key=_candidate_sort_key):
            self.projection.require_catalog_ready()
            exact = self.match_finder.exact(document)
            matches: list[MatchedMemory] = []
            relation: RelationResolution | None = None
            if exact is None and self.semantic_merge_mode in {"observe", "apply"}:
                try:
                    matches = self.match_finder.semantic(document)
                except Exception as exc:
                    warnings.append(
                        f"semantic matching failed for {document.memory_id}: {exc}"
                    )
                semantic_audit.append(
                    {
                        "proposed_memory_id": document.memory_id,
                        "candidate_id": document.extensions.get("candidate_id", ""),
                        "matches": [item.to_dict() for item in matches],
                    }
                )
                if matches and self.relation_resolver is not None:
                    # 交给 llm 来进一步判断相似度，这里只返回审查结果，不做直接的判断
                    relation = self.relation_resolver.resolve(document, matches)
                    relation_audit.append(
                        {
                            "proposed_memory_id": document.memory_id,
                            **relation.to_dict(),
                        }
                    )
                    if relation.error:
                        warnings.append(
                            f"semantic relation resolution failed for {document.memory_id}: "
                            f"{relation.error}"
                        )

            # 这里做直接的判断
            evolved = self.evolution_policy.apply(
                document,
                exact=exact,
                resolution=relation,
                mode=self.semantic_merge_mode,
            )
            stored = self.projection.persist(evolved.document, warnings)
            saved_by_id[stored.memory_id] = stored
            mutation_record = evolved.mutation.to_dict()
            mutation_record["embedding_model_fingerprints"] = sorted(
                {item.model_fingerprint for item in matches if item.model_fingerprint}
            )
            if relation is not None:
                selected = next(
                    (
                        item
                        for item in relation.assessments
                        if item.target_memory_id == stored.memory_id
                    ),
                    None,
                )
                if selected is not None:
                    mutation_record["semantic_relation"] = selected.relation
                    mutation_record["relation_confidence"] = selected.confidence
            mutation_audit.append(mutation_record)

        if extraction.source == "rule_based_fallback":
            warnings.append("LLM extraction failed; rule-based fallback was used")
        saved = list(saved_by_id.values())
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
        # 记录这一次 consolidation 的流程与结果
        self.run_store.save(
            outcome,
            {
                "request": {
                    "task_id": task_id,
                    "pipeline_version": self.pipeline_version,
                    "extractor": extraction.source,
                    "semantic_merge_mode": self.semantic_merge_mode,
                },
                "evidence_bundle": bundle.to_dict(),
                "candidate_seeds": {"candidates": [item.to_dict() for item in seeds]},
                "candidates": {"candidates": [item.to_dict() for item in candidates]},
                "evidence_resolution": {
                    "resolutions": [item.to_dict() for item in resolutions]
                },
                "extraction": extraction.to_dict(),
                "documents": {"documents": [item.to_dict() for item in saved]},
                "semantic_matches": {"matches": semantic_audit},
                "relation_resolutions": {"resolutions": relation_audit},
                "memory_mutations": {"mutations": mutation_audit},
            },
        )
        return outcome

    def _replay(self, existing: ConsolidationResult) -> ConsolidationResult:
        warnings: list[str] = []
        payload = self.run_store.read_artifact(
            existing.task_id,
            self.pipeline_version,
            "documents",
        )
        # 检测该归档文件生成的相应 长期记忆 md 文件是否存在并恢复
        if isinstance(payload, dict):
            for raw in payload.get("documents", []):
                if not isinstance(raw, dict):
                    continue
                document = MemoryDocument.from_dict(raw)
                stored = self.document_store.get(document.memory_id)
                if stored is None:
                    self.projection.persist(document, warnings)
                else:
                    self.projection.synchronize(stored, warnings)
        if not warnings:
            return existing
        return replace(
            existing,
            status="partial",
            warnings=tuple(dict.fromkeys(existing.warnings + tuple(warnings))),
        )


def _candidate_sort_key(document: MemoryDocument) -> tuple[str, str]:
    return str(document.extensions.get("candidate_id") or ""), document.memory_id
