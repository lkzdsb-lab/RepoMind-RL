"""Search codebase context indexes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent_runtime.codebase_context.builder import CodebaseContextBuilder
from agent_runtime.codebase_context.models import (
    CodebaseContextIndex,
    EmbeddingDoc,
)
from agent_runtime.codebase_context.store import CodebaseContextStore
from loguru import logger
from utils import _tokens


class CodebaseContextSearcher:
    def __init__(self, index: CodebaseContextIndex) -> None:
        self.index = index

    @classmethod
    def from_repo(
        cls,
        repo_path: str,
        index_path: str,
        force_rebuild: bool = False,
    ) -> "CodebaseContextSearcher":
        store = CodebaseContextStore(repo_path, index_path=index_path)
        if force_rebuild or store.is_stale():
            logger.info(
                "codebase context index rebuild requested repo_path={} index_path={} force_rebuild={}",
                repo_path,
                index_path,
                force_rebuild,
            )
            index = CodebaseContextBuilder().build(repo_path)
            store.save(index)
        else:
            logger.debug("codebase context index loaded repo_path={} index_path={}", repo_path, index_path)
            index = store.load()
        return cls(index)

    def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        query_tokens = set(_tokens(query))
        docs = self._rank_docs(query_tokens, limit=max(limit * 3, 20))
        file_scores = self._file_scores(docs)
        files = [
            {
                "path": path,
                "score": score,
                "layer": self._tree_by_path().get(path, {}).get("layer", "unknown"),
            }
            for path, score in file_scores[:limit]
        ]
        symbols = self._rank_records(self.index.symbols, query_tokens, limit)
        functions = self._rank_records(self.index.functions, query_tokens, limit)
        routes = self._rank_records(self.index.api_routes, query_tokens, limit)
        db_models = self._rank_records(self.index.db_models, query_tokens, limit)
        tests = self._related_tests([item["path"] for item in files], limit=limit)
        call_graph = self._related_call_edges(functions, limit=limit)
        flow = self._flow(files, functions, db_models, routes)

        result = {
            "query": query,
            "files": files,
            "symbols": [_record_to_dict(item) for item, _ in symbols],
            "functions": [_record_to_dict(item) for item, _ in functions],
            "api_routes": [_record_to_dict(item) for item, _ in routes],
            "db_models": [_record_to_dict(item) for item, _ in db_models],
            "call_graph": [_record_to_dict(edge) for edge in call_graph],
            "test_mappings": [_record_to_dict(mapping) for mapping in tests],
            "embedding_matches": [
                {
                    "doc_id": doc.doc_id,
                    "kind": doc.kind,
                    "title": doc.title,
                    "file_path": doc.file_path,
                    "symbol": doc.symbol,
                    "score": score,
                }
                for doc, score in docs[:limit]
            ],
            "flow": flow,
            "metadata": self.index.metadata,
        }
        logger.info(
            "codebase context searched query_chars={} files={} functions={} routes={} db_models={}",
            len(query),
            len(files),
            len(functions),
            len(routes),
            len(db_models),
        )
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "files": len(self.index.tree),
            "symbols": len(self.index.symbols),
            "functions": len(self.index.functions),
            "api_routes": len(self.index.api_routes),
            "db_models": len(self.index.db_models),
            "call_edges": len(self.index.call_graph),
            "test_mappings": len(self.index.test_mappings),
            "embedding_docs": len(self.index.embeddings),
            "metadata": self.index.metadata,
        }

    def _rank_docs(
        self,
        query_tokens: set[str],
        limit: int,
    ) -> list[tuple[EmbeddingDoc, float]]:
        """
            对文件和用户 token 进行匹配打分，最后获取得分排在前的文件
        """
        scored: list[tuple[EmbeddingDoc, float]] = []
        for doc in self.index.embeddings:
            score = _token_score(query_tokens, set(doc.tokens))
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _file_scores(self, docs: list[tuple[EmbeddingDoc, float]]) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for doc, score in docs:
            if doc.file_path:
                scores[doc.file_path] = scores.get(doc.file_path, 0.0) + score
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _rank_records(
        self,
        records: list[Any],
        query_tokens: set[str],
        limit: int,
    ) -> list[tuple[Any, float]]:
        scored: list[tuple[Any, float]] = []
        for record in records:
            tokens = set(_tokens(" ".join(str(value) for value in _record_to_dict(record).values())))
            score = _token_score(query_tokens, tokens)
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _related_tests(self, file_paths: list[str], limit: int) -> list[Any]:
        selected = set(file_paths)
        mappings = [
            mapping
            for mapping in self.index.test_mappings
            if mapping.source_path in selected or mapping.test_path in selected
        ]
        return sorted(mappings, key=lambda item: item.confidence, reverse=True)[:limit]

    def _related_call_edges(self, functions: list[tuple[Any, float]], limit: int) -> list[Any]:
        names = {function.full_name for function, _ in functions}
        names.update(function.name for function, _ in functions)
        edges = [
            edge
            for edge in self.index.call_graph
            if edge.caller in names or edge.callee in names or edge.callee.split(".")[-1] in names
        ]
        return edges[:limit]

    def _flow(
        self,
        files: list[dict],
        functions: list[tuple[Any, float]],
        db_models: list[tuple[Any, float]],
        routes: list[tuple[Any, float]],
    ) -> dict[str, Any]:
        layer_paths: dict[str, list[str]] = {}
        for item in files:
            layer_paths.setdefault(item["layer"], []).append(item["path"])
        hint = self.index.metadata.get("go_flow_hint") or "entrypoint -> domain logic -> persistence/model"
        return {
            "hint": hint,
            "layers": layer_paths,
            "routes": [route.path for route, _ in routes],
            "entrypoints": [function.full_name for function, _ in functions if function.layer in {"handler", "route"}],
            "db_models": [model.name for model, _ in db_models],
        }

    def _tree_by_path(self) -> dict[str, dict]:
        return {entry.path: _record_to_dict(entry) for entry in self.index.tree}


def build_or_load_index(
    repo_path: str,
    index_path: str,
    force_rebuild: bool = False,
) -> tuple[CodebaseContextIndex, Path]:
    store = CodebaseContextStore(repo_path, index_path=index_path)
    if force_rebuild or store.is_stale():
        logger.info(
            "building codebase context index repo_path={} index_path={} force_rebuild={}",
            repo_path,
            index_path,
            force_rebuild,
        )
        index = CodebaseContextBuilder().build(repo_path)
        path = store.save(index)
        logger.info("codebase context index saved path={}", path)
        return index, path
    logger.debug("using existing codebase context index path={}", store.index_path)
    return store.load(), store.index_path


def _record_to_dict(record: Any) -> dict:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if hasattr(record, "__dataclass_fields__"):
        return {
            key: _record_to_dict(value)
            if hasattr(value, "__dataclass_fields__")
            else [_record_to_dict(item) if hasattr(item, "__dataclass_fields__") else item for item in value]
            if isinstance(value, list)
            else value
            for key, value in record.__dict__.items()
        }
    if isinstance(record, dict):
        return record
    return {"value": str(record)}


def _token_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """
        Score = 精准匹配数 / 搜索词总数 + (模糊匹配数 * 0.05)
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    exact = len(query_tokens.intersection(doc_tokens))
    fuzzy = 0
    for query in query_tokens:
        if any(query in token or token in query for token in doc_tokens if len(query) >= 3 and len(token) >= 3):
            fuzzy += 1
    return exact / len(query_tokens) + fuzzy * 0.05
