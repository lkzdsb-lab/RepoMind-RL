"""Build a local codebase context index."""

from __future__ import annotations

import re
from pathlib import Path

from agent_runtime.codebase_context.models import (
    ApiRouteEntry,
    CallGraphEdge,
    CodebaseContextIndex,
    DbModelEntry,
    DbModelField,
    EmbeddingDoc,
    FunctionEntry,
    RepoTreeEntry,
    SymbolEntry,
    TestFileMapping,
)
from loguru import logger
from config import CompressionConfig
from utils import _tokens

config = CompressionConfig()

class CodebaseContextBuilder:
    def __init__(self, max_file_bytes: int = 500_000) -> None:
        self.max_file_bytes = max_file_bytes

    def build(self, repo_path: str | Path) -> CodebaseContextIndex:
        repo = Path(repo_path).resolve()
        logger.info("codebase context build started repo_path={}", repo)
        index = CodebaseContextIndex(repo_path=repo.as_posix())
        files = self._iter_files(repo)
        go_files: list[tuple[Path, str, str]] = []

        for path in files:
            rel = path.relative_to(repo).as_posix()
            language = config.INDEXED_EXTENSIONS.get(path.suffix, "text")
            content = self._read_text(path)
            package = _go_package(content) if path.suffix == ".go" else ""
            layer = _layer_for_path(rel)
            index.tree.append(
                RepoTreeEntry(
                    path=rel,
                    language=language,
                    size_bytes=path.stat().st_size,
                    lines=content.count("\n") + 1 if content else 0,
                    package=package,
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
                    metadata={"language": language, "layer": layer, "package": package},
                )
            )
            if path.suffix == ".go":
                go_files.append((path, rel, content))

        table_names = self._go_table_names(go_files)
        for path, rel, content in go_files:
            self._index_go_file(index, rel, content, table_names)

        self._build_call_graph_edges(index)
        self._build_test_mappings(index)
        self._build_go_architecture_metadata(index)
        index.metadata.update(
            {
                "file_count": len(index.tree),
                "symbol_count": len(index.symbols),
                "function_count": len(index.functions),
                "api_route_count": len(index.api_routes),
                "db_model_count": len(index.db_models),
                "call_edge_count": len(index.call_graph),
            }
        )
        logger.info(
            "codebase context build completed files={} go_files={} symbols={} functions={} routes={} db_models={} call_edges={}",
            len(index.tree),
            len(go_files),
            len(index.symbols),
            len(index.functions),
            len(index.api_routes),
            len(index.db_models),
            len(index.call_graph),
        )
        return index

    def _iter_files(self, repo: Path) -> list[Path]:
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

    def _index_go_file(
        self,
        index: CodebaseContextIndex,
        rel: str,
        content: str,
        table_names: dict[str, str],
    ) -> None:
        package = _go_package(content)
        layer = _layer_for_path(rel)
        for symbol in self._go_symbols(rel, content, package, layer):
            index.symbols.append(symbol)
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"symbol:{symbol.file_path}:{symbol.full_name}:{symbol.line}",
                    kind="symbol",
                    title=symbol.full_name,
                    content=f"{symbol.kind} {symbol.signature}",
                    file_path=symbol.file_path,
                    symbol=symbol.full_name,
                    metadata={"layer": symbol.layer, "package": symbol.package},
                )
            )

        functions = self._go_functions(rel, content, package, layer)
        index.functions.extend(functions)
        for function in functions:
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"function:{function.file_path}:{function.full_name}:{function.start_line}",
                    kind="function",
                    title=function.full_name,
                    content=" ".join([function.signature, " ".join(function.calls)]),
                    file_path=function.file_path,
                    symbol=function.full_name,
                    metadata={
                        "layer": function.layer,
                        "package": function.package,
                        "receiver": function.receiver,
                    },
                )
            )

        index.api_routes.extend(self._go_api_routes(rel, content))
        models = self._go_db_models(rel, content, package, table_names)
        index.db_models.extend(models)
        for model in models:
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"db_model:{model.file_path}:{model.name}",
                    kind="db_model",
                    title=model.name,
                    content=" ".join(
                        [
                            model.name,
                            model.table,
                            " ".join(f"{field.name} {field.type}" for field in model.fields),
                        ]
                    ),
                    file_path=model.file_path,
                    symbol=model.name,
                    metadata={"table": model.table, "package": model.package},
                )
            )

    def _go_symbols(
        self,
        rel: str,
        content: str,
        package: str,
        layer: str,
    ) -> list[SymbolEntry]:
        symbols: list[SymbolEntry] = []
        for match in re.finditer(r"(?m)^type\s+([A-Za-z_]\w*)\s+(struct|interface)\b", content):
            symbols.append(
                SymbolEntry(
                    name=match.group(1),
                    kind=match.group(2),
                    file_path=rel,
                    line=_line_number(content, match.start()),
                    package=package,
                    signature=_line_at(content, match.start()).strip(),
                    layer=layer,
                )
            )
        for match in re.finditer(r"(?m)^(?:var|const)\s+([A-Za-z_]\w*)\b", content):
            symbols.append(
                SymbolEntry(
                    name=match.group(1),
                    kind="value",
                    file_path=rel,
                    line=_line_number(content, match.start()),
                    package=package,
                    signature=_line_at(content, match.start()).strip(),
                    layer=layer,
                )
            )
        return symbols

    def _go_functions(
        self,
        rel: str,
        content: str,
        package: str,
        layer: str,
    ) -> list[FunctionEntry]:
        functions: list[FunctionEntry] = []
        pattern = re.compile(r"(?m)^func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(")
        for match in pattern.finditer(content):
            brace = content.find("{", match.end())
            if brace == -1:
                continue
            end = _find_matching_brace(content, brace)
            if end == -1:
                continue
            signature = _collapse_ws(content[match.start() : brace])
            body = content[brace + 1 : end]
            receiver = _receiver_type(match.group("recv") or "")
            name = match.group("name")
            full_name = f"{receiver}.{name}" if receiver else f"{package}.{name}"
            calls = _extract_go_calls(body)
            functions.append(
                FunctionEntry(
                    name=name,
                    full_name=full_name,
                    file_path=rel,
                    start_line=_line_number(content, match.start()),
                    end_line=_line_number(content, end),
                    signature=signature,
                    package=package,
                    receiver=receiver,
                    layer=layer,
                    calls=calls,
                )
            )
        return functions

    def _go_api_routes(self, rel: str, content: str) -> list[ApiRouteEntry]:
        routes: list[ApiRouteEntry] = []
        route_pattern = re.compile(
            r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s*\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][\w\.]*)",
            re.IGNORECASE,
        )
        for match in route_pattern.finditer(content):
            routes.append(
                ApiRouteEntry(
                    method=match.group(1).upper(),
                    path=match.group(2),
                    handler=match.group(3),
                    file_path=rel,
                    line=_line_number(content, match.start()),
                    framework="gin/echo",
                    middleware=_nearby_middlewares(content, match.start()),
                )
            )

        handle_func_pattern = re.compile(
            r"HandleFunc\s*\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][\w\.]*)\s*\)(?:\.Methods\(\s*\"([A-Z]+)\"\s*\))?"
        )
        for match in handle_func_pattern.finditer(content):
            routes.append(
                ApiRouteEntry(
                    method=(match.group(3) or "ANY").upper(),
                    path=match.group(1),
                    handler=match.group(2),
                    file_path=rel,
                    line=_line_number(content, match.start()),
                    framework="net/http/gorilla",
                    middleware=_nearby_middlewares(content, match.start()),
                )
            )

        http_pattern = re.compile(r"http\.HandleFunc\s*\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][\w\.]*)")
        for match in http_pattern.finditer(content):
            routes.append(
                ApiRouteEntry(
                    method="ANY",
                    path=match.group(1),
                    handler=match.group(2),
                    file_path=rel,
                    line=_line_number(content, match.start()),
                    framework="net/http",
                )
            )
        return routes

    def _go_table_names(self, go_files: list[tuple[Path, str, str]]) -> dict[str, str]:
        table_names: dict[str, str] = {}
        pattern = re.compile(
            r"func\s*\(\s*\w+\s+\*?([A-Za-z_]\w*)\s*\)\s*TableName\s*\(\s*\)\s*string\s*\{[^}]*return\s+\"([^\"]+)\"",
            re.DOTALL,
        )
        for _, _, content in go_files:
            for match in pattern.finditer(content):
                table_names[match.group(1)] = match.group(2)
        return table_names

    def _go_db_models(
        self,
        rel: str,
        content: str,
        package: str,
        table_names: dict[str, str],
    ) -> list[DbModelEntry]:
        models: list[DbModelEntry] = []
        struct_pattern = re.compile(r"(?m)^type\s+([A-Za-z_]\w*)\s+struct\s*\{")
        for match in struct_pattern.finditer(content):
            brace = content.find("{", match.end() - 1)
            end = _find_matching_brace(content, brace)
            if end == -1:
                continue
            name = match.group(1)
            body = content[brace + 1 : end]
            fields = _parse_go_struct_fields(body)
            is_model = (
                bool(table_names.get(name))
                or _layer_for_path(rel) == "model"
                or any(field.tags.get("gorm") or field.tags.get("db") for field in fields)
                or name.lower().endswith(("model", "entity", "record"))
            )
            if not is_model:
                continue
            models.append(
                DbModelEntry(
                    name=name,
                    table=table_names.get(name, _snake_plural(name)),
                    file_path=rel,
                    line=_line_number(content, match.start()),
                    package=package,
                    fields=fields,
                )
            )
        return models

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

    def _build_test_mappings(self, index: CodebaseContextIndex) -> None:
        paths = {entry.path for entry in index.tree}
        go_files = [entry.path for entry in index.tree if entry.path.endswith(".go")]
        tests = [path for path in go_files if path.endswith("_test.go")]
        tests_by_dir: dict[str, list[str]] = {}
        for test in tests:
            tests_by_dir.setdefault(str(Path(test).parent), []).append(test)

        for path in go_files:
            if path.endswith("_test.go"):
                continue
            direct = path[:-3] + "_test.go"
            if direct in paths:
                index.test_mappings.append(
                    TestFileMapping(
                        source_path=path,
                        test_path=direct,
                        confidence=1.0,
                        reason="same file stem",
                    )
                )
                continue
            directory = str(Path(path).parent)
            for test in tests_by_dir.get(directory, []):
                index.test_mappings.append(
                    TestFileMapping(
                        source_path=path,
                        test_path=test,
                        confidence=0.55,
                        reason="same package directory",
                    )
                )

    def _build_go_architecture_metadata(self, index: CodebaseContextIndex) -> None:
        layer_counts: dict[str, int] = {}
        for entry in index.tree:
            layer_counts[entry.layer] = layer_counts.get(entry.layer, 0) + 1
        index.metadata["go_layers"] = layer_counts
        index.metadata["go_flow_hint"] = "handler -> service -> repository -> model"


def _go_package(content: str) -> str:
    match = re.search(r"(?m)^package\s+([A-Za-z_]\w*)", content)
    return match.group(1) if match else ""


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


def _line_number(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _line_at(content: str, offset: int) -> str:
    start = content.rfind("\n", 0, offset) + 1
    end = content.find("\n", offset)
    if end == -1:
        end = len(content)
    return content[start:end]


def _find_matching_brace(content: str, start: int) -> int:
    if start < 0 or start >= len(content) or content[start] != "{":
        return -1
    depth = 0
    in_string = ""
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = ""
            continue
        if char in {'"', "'", "`"}:
            in_string = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _receiver_type(receiver: str) -> str:
    if not receiver:
        return ""
    parts = receiver.replace("*", " ").split()
    return parts[-1] if parts else ""


def _collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_go_calls(body: str) -> list[str]:
    calls: list[str] = []
    for match in re.finditer(r"\b(?:(?P<prefix>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)\s*\(", body):
        name = match.group("name")
        prefix = match.group("prefix")
        if name in config.CALL_EXCLUDE:
            continue
        callee = f"{prefix}.{name}" if prefix else name
        if callee not in calls:
            calls.append(callee)
    return calls


def _nearby_middlewares(content: str, offset: int) -> list[str]:
    window = content[max(0, offset - 800) : offset]
    middlewares: list[str] = []
    for match in re.finditer(r"\.Use\s*\(([^)]*)\)", window):
        for item in match.group(1).split(","):
            item = item.strip()
            if item:
                middlewares.append(item)
    return middlewares[-5:]


def _parse_go_struct_fields(body: str) -> list[DbModelField]:
    fields: list[DbModelField] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        tag = ""
        if "`" in line:
            before, _, rest = line.partition("`")
            tag, _, _ = rest.partition("`")
            line = before.strip()
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        field_type = parts[1]
        if not re.match(r"[A-Za-z_]\w*$", name):
            continue
        fields.append(DbModelField(name=name, type=field_type, tags=_parse_go_tags(tag)))
    return fields


def _parse_go_tags(tag: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for match in re.finditer(r'(\w+):"([^"]*)"', tag):
        tags[match.group(1)] = match.group(2)
    return tags


def _snake_plural(name: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    if snake.endswith("y"):
        return snake[:-1] + "ies"
    if snake.endswith("s"):
        return snake
    return snake + "s"


def _embedding_doc(
    doc_id: str,
    kind: str,
    title: str,
    content: str,
    file_path: str = "",
    symbol: str = "",
    metadata: dict | None = None,
) -> EmbeddingDoc:
    return EmbeddingDoc(
        doc_id=doc_id,
        kind=kind,
        title=title,
        content=content[:4000],
        file_path=file_path,
        symbol=symbol,
        tokens=sorted(set(_tokens(" ".join([title, content])))),
        metadata=metadata or {},
    )

