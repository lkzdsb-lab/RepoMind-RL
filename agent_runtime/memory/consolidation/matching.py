"""Exact and semantic candidate discovery for memory consolidation."""

from __future__ import annotations

from agent_runtime.memory.consolidation.models import MatchedMemory
from agent_runtime.memory.domain.models import MemoryDocument, MemoryStatus
from agent_runtime.memory.semantic import SQLiteMemorySemanticIndex
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.interfaces import MemoryCatalog


_MATCHABLE = {
    MemoryStatus.DRAFT,
    MemoryStatus.VERIFIED,
    MemoryStatus.NEEDS_REVIEW,
}


class MemoryMatchFinder:
    def __init__(
        self,
        semantic_index: SQLiteMemorySemanticIndex | None,
        *,
        catalog: MemoryCatalog,
        document_store: MarkdownMemoryDocumentStore,
        top_k: int = 5,
        min_similarity: float = 0.78,
    ) -> None:
        self.semantic_index = semantic_index
        self.catalog = catalog
        self.document_store = document_store
        self.top_k = max(1, int(top_k))
        self.min_similarity = max(0.0, min(1.0, float(min_similarity)))

    def exact(
        self,
        proposed: MemoryDocument,
    ) -> MemoryDocument | None:
        """ 查找正文精确相同的旧记忆"""
        for hit in self.catalog.find_exact(proposed):
            document = self.document_store.get(hit.memory_id)
            if document is None:
                raise ValueError(f"missing exact-match Markdown: {hit.memory_id}; rebuild Catalog")
            if self.document_store.content_hash(hit.memory_id) != hit.content_hash:
                raise ValueError(f"stale exact-match projection: {hit.memory_id}; sync Catalog")
            if (document.status in _MATCHABLE and document.scope == proposed.scope
                    and document.memory_type == proposed.memory_type
                    and document.knowledge_hash == proposed.knowledge_hash):
                return document
        return None

    def semantic(
        self,
        proposed: MemoryDocument,
    ) -> list[MatchedMemory]:
        """ 查找语义相似的旧记忆"""
        if self.semantic_index is None:
            return []
        matches = self.semantic_index.semantic_search(
            proposed,
            limit=self.top_k,
            min_score=self.min_similarity,
        )
        result: list[MatchedMemory] = []
        for match in matches:
            document = self.document_store.get(match.memory_id)
            # 检查旧记忆的文件状态
            if document is None or document.status not in _MATCHABLE:
                continue
            if (
                match.knowledge_hash != document.knowledge_hash
                or match.revision != document.revision
            ):
                continue
            if document.memory_type != proposed.memory_type or document.scope != proposed.scope:
                continue
            result.append(
                MatchedMemory(
                    document=document,
                    score=match.score,
                    source="semantic",
                    model_fingerprint=match.model_fingerprint,
                )
            )
        return result
