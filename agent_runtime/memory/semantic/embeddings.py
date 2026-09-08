"""Embedding provider boundary and OpenAI-compatible implementation."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from agent_runtime.memory.io import compact_json
from agent_runtime.memory.semantic.models import EmbeddingBatch
from config import EmbeddingConfig


class EmbeddingClient(Protocol):
    @property
    def provider(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def model_fingerprint(self) -> str: ...
    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


class OpenAICompatibleEmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        api_key = os.getenv(config.api_key_env) if config.api_key_env else ""
        if not api_key:
            raise RuntimeError(
                f"embedding API key env var `{config.api_key_env}` is not set"
            )
        if not str(config.model or "").strip():
            raise RuntimeError("embedding model is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.api_base or None,
            timeout=config.timeout,
        )

    @property
    def provider(self) -> str:
        return str(self.config.provider)

    @property
    def model(self) -> str:
        return str(self.config.model)

    @property
    def model_fingerprint(self) -> str:
        return _embedding_fingerprint(self.config)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        values = [str(text or "").strip() for text in texts]
        if not values or any(not value for value in values):
            raise ValueError("embedding input must contain non-empty text")
        vectors: list[tuple[float, ...]] = []
        batch_size = max(1, int(self.config.batch_size))
        for offset in range(0, len(values), batch_size):
            kwargs = {
                "model": self.config.model,
                "input": values[offset : offset + batch_size],
            }
            if int(self.config.dimensions) > 0:
                kwargs["dimensions"] = int(self.config.dimensions)
            response = self.client.embeddings.create(**kwargs)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(tuple(float(value) for value in item.embedding) for item in ordered)
        if len(vectors) != len(values):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        dimensions = len(vectors[0])
        if dimensions <= 0 or any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("embedding provider returned inconsistent dimensions")
        return EmbeddingBatch(
            vectors=tuple(vectors),
            provider=self.config.provider,
            model=self.config.model,
            dimensions=dimensions,
            model_fingerprint=self.model_fingerprint,
        )


def build_embedding_client(config: EmbeddingConfig) -> EmbeddingClient | None:
    provider = str(config.provider or "").strip().lower()
    if provider in {"", "disabled", "none"}:
        return None
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        return OpenAICompatibleEmbeddingClient(config)
    raise ValueError(f"unsupported embedding provider: {config.provider}")


def _embedding_fingerprint(config: EmbeddingConfig) -> str:
    return hashlib.sha256(
        compact_json(
            {
                "provider": str(config.provider).strip().lower(),
                "model": str(config.model).strip(),
                "dimensions": int(config.dimensions),
                "api_base": str(config.api_base).strip(),
            }
        ).encode("utf-8")
    ).hexdigest()[:24]
