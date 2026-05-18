"""Tool adapters for the codebase context index."""

from __future__ import annotations

from typing import Any, Dict

from agent_runtime.codebase_context.search import (
    CodebaseContextSearcher,
    build_or_load_index,
)
from agent_runtime.codebase_context.store import DEFAULT_INDEX_PATH
from loguru import logger


def build_codebase_context(
    repo_path: str,
    index_path: str = DEFAULT_INDEX_PATH,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    logger.info(
        "build_codebase_context tool started repo_path={} index_path={} force_rebuild={}",
        repo_path,
        index_path,
        force_rebuild,
    )
    index, path = build_or_load_index(
        repo_path,
        index_path=index_path,
        force_rebuild=force_rebuild,
    )
    result = {
        "index_path": path.as_posix(),
        "summary": CodebaseContextSearcher(index).summary(),
    }
    logger.info(
        "build_codebase_context tool completed index_path={} files={} functions={}",
        path.as_posix(),
        result["summary"].get("files"),
        result["summary"].get("functions"),
    )
    return result


def search_code_context(
    repo_path: str,
    query: str,
    limit: int = 10,
    index_path: str = DEFAULT_INDEX_PATH,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    if not query:
        logger.warning("search_code_context tool called with empty query repo_path={}", repo_path)
        return {
            "query": query,
            "files": [],
            "symbols": [],
            "functions": [],
            "api_routes": [],
            "db_models": [],
            "call_graph": [],
            "test_mappings": [],
            "embedding_matches": [],
            "flow": {},
        }

    logger.info(
        "search_code_context tool started repo_path={} query_chars={} limit={} index_path={}",
        repo_path,
        len(query),
        limit,
        index_path,
    )
    searcher = CodebaseContextSearcher.from_repo(
        repo_path,
        index_path=index_path,
        force_rebuild=force_rebuild,
    )
    result = searcher.search(query, limit=limit)
    logger.info(
        "search_code_context tool completed files={} functions={} routes={} db_models={}",
        len(result.get("files", [])),
        len(result.get("functions", [])),
        len(result.get("api_routes", [])),
        len(result.get("db_models", [])),
    )
    return result
