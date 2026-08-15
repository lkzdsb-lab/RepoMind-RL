"""Build a local codebase context index."""

from __future__ import annotations

from pathlib import Path
from agent_runtime.codebase_context.models import (
    CallGraphEdge,
    CodebaseContextIndex,
    RepoTreeEntry,
)
from loguru import logger

from agent_runtime.codebase_context.analyzers.analyzers import AnalyzerRegistry, SourceFile
from config import CompressionConfig, FileConfig
from utils import (
    _embedding_doc,
    _language_counts,
)

config = CompressionConfig()

class CodebaseContextBuilder:
    def __init__(
        self,
        max_file_bytes: int = FileConfig.MAX_FILE_BYTES_LIMIT,
        analyzer_registry: AnalyzerRegistry | None = None,
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.analyzer_registry = analyzer_registry or AnalyzerRegistry.default()

    def build(self, repo_path: str | Path) -> CodebaseContextIndex:
        repo = Path(repo_path).resolve()
        logger.info("codebase context build started repo_path={}", repo)
        index = CodebaseContextIndex(repo_path=repo.as_posix())
        files = self._iter_files(repo)
        sources: list[SourceFile] = []

        for path in files:
            rel = path.relative_to(repo).as_posix()
            language = config.INDEXED_EXTENSIONS.get(path.suffix, "text")
            content = self._read_text(path)
            layer = _layer_for_path(rel)
            sources.append(
                SourceFile(
                    path=path,
                    rel_path=rel,
                    language=language,
                    content=content,
                    layer=layer,
                )
            )
            index.tree.append(
                RepoTreeEntry(
                    path=rel,
                    language=language,
                    size_bytes=path.stat().st_size,
                    lines=content.count("\n") + 1 if content else 0,
                    layer=layer,
                )
            )
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"file:{rel}",
                    kind="file",
                    title=rel,
                    content=content[:12000],
                    file_path=rel,
                    metadata={"language": language, "layer": layer},
                )
            )
        self.analyzer_registry.analyze(index, sources)

        self._build_call_graph_edges(index)
        self._build_architecture_metadata(index)
        index.metadata.update(
            {
                "file_count": len(index.tree),
                "symbol_count": len(index.symbols),
                "function_count": len(index.functions),
                "api_route_count": len(index.api_routes),
                "db_model_count": len(index.db_models),
                "call_edge_count": len(index.call_graph),
                "languages": _language_counts(index),
            }
        )
        logger.info(
            "codebase context build completed files={} symbols={} functions={} routes={} db_models={} call_edges={}",
            len(index.tree),
            len(index.symbols),
            len(index.functions),
            len(index.api_routes),
            len(index.db_models),
            len(index.call_graph),
        )
        return index

    def _iter_files(self, repo: Path) -> list[Path]:
        """
            迭代获取所有满足要求的文件
        """
        files: list[Path] = []
        for path in repo.rglob("*"):
            if not path.is_file():
                continue
            if set(path.relative_to(repo).parts).intersection(config.IGNORED_DIRS):
                continue
            if path.stat().st_size > self.max_file_bytes:
                continue
            if path.suffix in config.INDEXED_EXTENSIONS:
                files.append(path)
        return sorted(files)

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _build_call_graph_edges(self, index: CodebaseContextIndex) -> None:
        known = {function.name for function in index.functions}
        known.update(function.full_name for function in index.functions)
        for function in index.functions:
            for callee in function.calls:
                normalized = callee.split(".")[-1]
                kind = "known" if callee in known or normalized in known else "external"
                index.call_graph.append(
                    CallGraphEdge(
                        caller=function.full_name,
                        callee=callee,
                        file_path=function.file_path,
                        line=function.start_line,
                        kind=kind,
                    )
                )

    def _build_architecture_metadata(self, index: CodebaseContextIndex) -> None:
        layer_counts: dict[str, int] = {}
        for entry in index.tree:
            layer_counts[entry.layer] = layer_counts.get(entry.layer, 0) + 1
        index.metadata["layers"] = layer_counts


def _layer_for_path(path: str) -> str:
    parts = [part.lower() for part in Path(path).parts]
    joined = "/".join(parts)
    if path.endswith("_test.go") or "/test/" in joined or "/tests/" in joined:
        return "test"
    if any(part in {"handler", "handlers", "controller", "controllers", "api"} for part in parts):
        return "handler"
    if any(part in {"router", "routes", "route"} for part in parts):
        return "route"
    if any(part in {"middleware", "middlewares"} for part in parts):
        return "middleware"
    if any(part in {"service", "services", "usecase", "usecases"} for part in parts):
        return "service"
    if any(part in {"repository", "repositories", "repo", "dao", "store"} for part in parts):
        return "repository"
    if any(part in {"model", "models", "entity", "entities", "domain"} for part in parts):
        return "model"
    return "unknown"
