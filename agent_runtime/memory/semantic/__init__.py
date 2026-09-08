"""Knowledge-only semantic projection used by offline consolidation."""

from agent_runtime.memory.semantic.embeddings import (
    EmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    build_embedding_client,
)
from agent_runtime.memory.semantic.models import (
    EmbeddingBatch,
    SemanticMatch,
    SemanticSyncResult,
)
from agent_runtime.memory.semantic.sqlite import SQLiteMemorySemanticIndex

__all__ = [
    "EmbeddingBatch",
    "EmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "SemanticMatch",
    "SemanticSyncResult",
    "SQLiteMemorySemanticIndex",
    "build_embedding_client",
]
