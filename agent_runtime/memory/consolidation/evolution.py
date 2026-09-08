"""Deterministic write policy for durable memory evolution."""

from __future__ import annotations

from dataclasses import replace

from agent_runtime.memory.consolidation.models import (
    EvolutionResult,
    MemoryMutation,
    RelationAssessment,
    RelationResolution,
)
from agent_runtime.memory.domain.models import MemoryDocument, MemoryStatus
from utils import utc_now


_STATUS_RANK = {
    MemoryStatus.DRAFT: 0,
    MemoryStatus.NEEDS_REVIEW: 1,
    MemoryStatus.VERIFIED: 2,
}


class MemoryEvolutionPolicy:
    """ 根据置信度、证据引用和作用域等规则决定是否执行合并"""
    def __init__(self, *, min_confidence: float = 0.85) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def apply(
        self,
        proposed: MemoryDocument,
        *,
        exact: MemoryDocument | None = None,
        resolution: RelationResolution | None = None,
        mode: str = "disabled",
    ) -> EvolutionResult:
        """ 接收新记忆、精确匹配结果、语义判断结果和运行模式"""
        candidate_id = str(proposed.extensions.get("candidate_id") or "")
        # 如果有旧记忆
        if exact is not None:
            merged = self._merge_common(exact, proposed)
            return self._result(
                merged,
                action="exact_merged",
                candidate_id=candidate_id,
                before=exact,
                reason="memory type, scope, and normalized knowledge hash matched",
            )

        if mode != "apply" or resolution is None:
            return self._result(
                proposed,
                action="created" if mode != "observe" else "observed_created",
                candidate_id=candidate_id,
                reason="no exact merge was applied",
            )
        # 在 apply 模式下，只把达到置信度门槛的 纳入后续决策
        decisive = [
            item
            for item in resolution.assessments
            if item.confidence >= self.min_confidence
            and item.relation in {"duplicate", "refine", "conflict"}
        ]
        mergeable = [item for item in decisive if item.relation in {"duplicate", "refine"}]
        conflicts = [item for item in decisive if item.relation == "conflict"]
        if conflicts or len(mergeable) > 1:
            targets = tuple(sorted({item.target_memory_id for item in conflicts + mergeable}))
            extensions = dict(proposed.extensions)
            extensions["semantic_review_targets"] = list(targets)
            reviewed = replace(proposed, status=MemoryStatus.NEEDS_REVIEW, extensions=extensions)
            reason = (
                "semantic conflict requires review"
                if conflicts
                else "multiple semantic merge targets require review"
            )
            return self._result(
                reviewed,
                action="conflict_created" if conflicts else "ambiguous_created",
                candidate_id=candidate_id,
                reason=reason,
            )
        if not mergeable:
            return self._result(
                proposed,
                action="created",
                candidate_id=candidate_id,
                reason="no high-confidence semantic merge relation",
            )

        assessment = mergeable[0]
        target = self._target(resolution, assessment.target_memory_id)
        if target is None:
            return self._reviewed(proposed, candidate_id, "semantic target was not available")
        if target.memory_type != proposed.memory_type or target.scope != proposed.scope:
            return self._reviewed(proposed, candidate_id, "semantic target changed type or scope")
        if assessment.relation == "duplicate":
            # 与精确合并类似：保留旧 ID、旧正文，只补充证据等信息
            merged = self._merge_common(target, proposed)
            return self._result(
                merged,
                action="semantic_merged",
                candidate_id=candidate_id,
                before=target,
                reason=assessment.reason,
            )
        return self._refine(target, proposed, assessment, candidate_id)

    def _refine(
        self,
        target: MemoryDocument,
        proposed: MemoryDocument,
        assessment: RelationAssessment,
        candidate_id: str,
    ) -> EvolutionResult:
        """ 使用 LLM 提出的新正文更新旧记忆"""
        if not all(
            (
                assessment.merged_title,
                assessment.merged_knowledge,
                assessment.merged_applicability,
            )
        ):
            return self._reviewed(proposed, candidate_id, "refinement fields were incomplete")
        allowed_evidence = {item.evidence_id for item in target.evidence + proposed.evidence}
        cited_evidence = set(assessment.supporting_evidence_ids)
        proposed_evidence = {item.evidence_id for item in proposed.evidence}
        if not cited_evidence or not cited_evidence.intersection(proposed_evidence):
            return self._reviewed(
                proposed,
                candidate_id,
                "refinement did not cite current candidate evidence",
            )
        if not cited_evidence.issubset(allowed_evidence):
            return self._reviewed(proposed, candidate_id, "refinement cited unknown evidence")
        common = self._merge_common(target, proposed)
        refined = replace(
            common,
            title=assessment.merged_title,
            knowledge=assessment.merged_knowledge,
            knowledge_hash="",
            applicability=assessment.merged_applicability,
            invalidation=assessment.merged_invalidation,
        )
        if refined.knowledge_hash == target.knowledge_hash:
            return self._reviewed(proposed, candidate_id, "refinement did not change knowledge")
        return self._result(
            refined,
            action="refined",
            candidate_id=candidate_id,
            before=target,
            reason=assessment.reason,
        )

    def _merge_common(
        self,
        target: MemoryDocument,
        proposed: MemoryDocument,
    ) -> MemoryDocument:
        evidence = {item.evidence_id: item for item in target.evidence}
        evidence.update({item.evidence_id: item for item in proposed.evidence})
        extensions = dict(target.extensions)
        source_tasks = _string_list(extensions.get("source_tasks"))
        for task_id in (
            *_string_list(proposed.extensions.get("source_tasks")),
            proposed.source.task_id,
        ):
            if task_id and task_id not in source_tasks:
                source_tasks.append(task_id)
        extensions["source_tasks"] = source_tasks
        status = max(
            (target.status, proposed.status),
            key=lambda value: _STATUS_RANK.get(value, -1),
        )
        return replace(
            target,
            status=status,
            triggers=tuple(dict.fromkeys(target.triggers + proposed.triggers)),
            tags=tuple(sorted(set(target.tags + proposed.tags))),
            evidence=tuple(evidence.values()),
            confidence=max(target.confidence, proposed.confidence),
            evidence_strength=max(target.evidence_strength, proposed.evidence_strength),
            revision=target.revision + 1,
            updated_at=utc_now(),
            extensions=extensions,
        )

    @staticmethod
    def _target(
        resolution: RelationResolution,
        memory_id: str,
    ) -> MemoryDocument | None:
        target = resolution.target_documents.get(memory_id)
        return target if isinstance(target, MemoryDocument) else None

    def _reviewed(
        self,
        proposed: MemoryDocument,
        candidate_id: str,
        reason: str,
    ) -> EvolutionResult:
        return self._result(
            replace(proposed, status=MemoryStatus.NEEDS_REVIEW),
            action="ambiguous_created",
            candidate_id=candidate_id,
            reason=reason,
        )

    @staticmethod
    def _result(
        document: MemoryDocument,
        *,
        action: str,
        candidate_id: str,
        reason: str,
        before: MemoryDocument | None = None,
    ) -> EvolutionResult:
        return EvolutionResult(
            document=document,
            mutation=MemoryMutation(
                action=action,
                source_candidate_id=candidate_id,
                target_memory_id=document.memory_id,
                before_revision=before.revision if before else 0,
                after_revision=document.revision,
                before_knowledge_hash=before.knowledge_hash if before else "",
                after_knowledge_hash=document.knowledge_hash,
                reason=reason[:2000],
            ),
        )


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
