"""Incremental projection writes with bounded durable repair."""

from agent_runtime.memory.consolidation.pending import PendingProjections
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.interfaces import MemoryCatalog
from agent_runtime.memory.domain.models import MemoryDocument
from agent_runtime.memory.semantic import SQLiteMemorySemanticIndex


class MemoryProjectionCoordinator:
    def __init__(self, *, document_store: MarkdownMemoryDocumentStore,
                 catalog: MemoryCatalog | None,
                 semantic_index: SQLiteMemorySemanticIndex | None) -> None:
        self.document_store = document_store
        self.catalog = catalog
        self.semantic_index = semantic_index
        self.pending = PendingProjections(document_store.root / 'projection_pending.sqlite3')

    def _targets(self):
        targets = {}
        if self.catalog is not None:
            targets['catalog'] = lambda document: self.catalog.synchronize(
                document, self.document_store.document_path(document))
        if self.semantic_index is not None:
            targets['semantic'] = self.semantic_index.synchronize
        return targets

    def persist(self, document: MemoryDocument, warnings: list[str]) -> MemoryDocument:
        # Register before Markdown replacement, so crashes cannot lose repair intent.
        for name in self._targets():
            self.pending.enqueue(document.memory_id, name)
        self.document_store.save(document)
        self._sync(document, warnings)
        return document

    def synchronize(self, document: MemoryDocument, warnings: list[str]) -> None:
        for name in self._targets():
            self.pending.enqueue(document.memory_id, name)
        self._sync(document, warnings)

    def _sync(self, document, warnings, only=None):
        for name, synchronize in self._targets().items():
            if only is not None and name != only:
                continue
            try:
                synchronize(document)
            except Exception as exc:
                self.pending.fail(document.memory_id, name, exc)
                warnings.append(f'{name} synchronization failed for {document.memory_id}: {exc}')
            else:
                self.pending.complete(document.memory_id, name)

    def retry_pending(self, warnings: list[str], *, limit: int = 16) -> None:
        if self.semantic_index is not None:
            try:
                issue = self.semantic_index.readiness_issue()
                if issue:
                    warnings.append(issue)
            except Exception as exc:
                warnings.append(f'semantic index readiness check failed: {exc}')
        targets = self._targets()
        if not targets:
            return
        for item in self.pending.batch(tuple(targets), limit):
            try:
                document = self.document_store.get(item['memory_id'])
                if document is None:
                    # A write intent can survive a failed first Markdown save.
                    self.pending.complete(item['memory_id'], item['projection'])
                else:
                    self._sync(document, warnings, only=item['projection'])
            except Exception as exc:
                self.pending.fail(item['memory_id'], item['projection'], exc)
                warnings.append(f"projection repair failed for {item['memory_id']}: {exc}")
        self.require_catalog_ready()
        if self.pending.has_pending('semantic'):
            warnings.append('semantic projection repair remains pending; retrieval may be incomplete')

    def require_catalog_ready(self) -> None:
        if self.catalog is not None and self.pending.has_pending('catalog'):
            raise RuntimeError('Catalog repairs remain pending; retry consolidation after repair to avoid duplicate memories')
