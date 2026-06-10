import re

from pathlib import Path
from agent_runtime.codebase_context import CodebaseContextIndex
from agent_runtime.codebase_context.analyzers.analyzers import (
    SourceFile,
    _line_number,
    _nearby_middlewares,
    _snake_plural,
    _find_matching_brace,
    _embedding_doc,
    _line_at,
    _collapse_ws,
    _receiver_type)
from agent_runtime.codebase_context.models import ApiRouteEntry, TestFileMapping, DbModelEntry, DbModelField, SymbolEntry, \
    FunctionEntry


class GoLanguageAnalyzer:
    """Go language analyzer: packages, symbols, functions, models, and tests."""

    name = "go"

    def supports(self, source: SourceFile) -> bool:
        return source.path.suffix == ".go"

    def analyze_file(self, index: CodebaseContextIndex, source: SourceFile) -> None:
        package = _go_package(source.content)
        for symbol in _go_symbols(source, package):
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
        for function in _go_functions(source, package):
            index.functions.append(function)
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

    def finalize(self, index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
        go_sources = [source for source in sources if source.path.suffix == ".go"]
        table_names = _go_table_names(go_sources)
        for source in go_sources:
            package = _go_package(source.content)
            for model in _go_db_models(source, package, table_names):
                index.db_models.append(model)
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
        _build_go_test_mappings(index)


class GoWebFrameworkAnalyzer:
    """Go web framework analyzer for route and flow metadata extraction."""

    name = "go_web"

    def supports(self, source: SourceFile) -> bool:
        return source.path.suffix == ".go"

    def analyze_file(self, index: CodebaseContextIndex, source: SourceFile) -> None:
        index.api_routes.extend(_go_api_routes(source))

    def finalize(self, index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
        if not any(source.path.suffix == ".go" for source in sources):
            return
        layer_counts: dict[str, int] = {}
        for entry in index.tree:
            layer_counts[entry.layer] = layer_counts.get(entry.layer, 0) + 1
        index.metadata["go_layers"] = layer_counts
        index.metadata["go_flow_hint"] = "handler -> service -> repository -> model"


def _go_package(content: str) -> str:
    match = re.search(r"(?m)^package\s+([A-Za-z_]\w*)", content)
    return match.group(1) if match else ""


def _go_symbols(source: SourceFile, package: str) -> list[SymbolEntry]:
    symbols: list[SymbolEntry] = []
    for match in re.finditer(r"(?m)^type\s+([A-Za-z_]\w*)\s+(struct|interface)\b", source.content):
        symbols.append(
            SymbolEntry(
                name=match.group(1),
                kind=match.group(2),
                file_path=source.rel_path,
                line=_line_number(source.content, match.start()),
                package=package,
                signature=_line_at(source.content, match.start()).strip(),
                layer=source.layer,
            )
        )
    for match in re.finditer(r"(?m)^(?:var|const)\s+([A-Za-z_]\w*)\b", source.content):
        symbols.append(
            SymbolEntry(
                name=match.group(1),
                kind="value",
                file_path=source.rel_path,
                line=_line_number(source.content, match.start()),
                package=package,
                signature=_line_at(source.content, match.start()).strip(),
                layer=source.layer,
            )
        )
    return symbols


def _go_functions(source: SourceFile, package: str) -> list[FunctionEntry]:
    functions: list[FunctionEntry] = []
    pattern = re.compile(r"(?m)^func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(")
    for match in pattern.finditer(source.content):
        brace = source.content.find("{", match.end())
        if brace == -1:
            continue
        end = _find_matching_brace(source.content, brace)
        if end == -1:
            continue
        signature = _collapse_ws(source.content[match.start(): brace])
        body = source.content[brace + 1: end]
        receiver = _receiver_type(match.group("recv") or "")
        name = match.group("name")
        full_name = f"{receiver}.{name}" if receiver else f"{package}.{name}"
        functions.append(
            FunctionEntry(
                name=name,
                full_name=full_name,
                file_path=source.rel_path,
                start_line=_line_number(source.content, match.start()),
                end_line=_line_number(source.content, end),
                signature=signature,
                package=package,
                receiver=receiver,
                layer=source.layer,
                calls=_extract_go_calls(body),
            )
        )
    return functions


def _go_table_names(sources: list[SourceFile]) -> dict[str, str]:
    table_names: dict[str, str] = {}
    pattern = re.compile(
        r"func\s*\(\s*\w+\s+\*?([A-Za-z_]\w*)\s*\)\s*TableName\s*\(\s*\)\s*string\s*\{[^}]*return\s+\"([^\"]+)\"",
        re.DOTALL,
    )
    for source in sources:
        for match in pattern.finditer(source.content):
            table_names[match.group(1)] = match.group(2)
    return table_names


def _go_db_models(
        source: SourceFile,
        package: str,
        table_names: dict[str, str],
) -> list[DbModelEntry]:
    models: list[DbModelEntry] = []
    struct_pattern = re.compile(r"(?m)^type\s+([A-Za-z_]\w*)\s+struct\s*\{")
    for match in struct_pattern.finditer(source.content):
        brace = source.content.find("{", match.end() - 1)
        end = _find_matching_brace(source.content, brace)
        if end == -1:
            continue
        name = match.group(1)
        body = source.content[brace + 1: end]
        fields = _parse_go_struct_fields(body)
        is_model = (
                bool(table_names.get(name))
                or source.layer == "model"
                or any(field.tags.get("gorm") or field.tags.get("db") for field in fields)
                or name.lower().endswith(("model", "entity", "record"))
        )
        if not is_model:
            continue
        models.append(
            DbModelEntry(
                name=name,
                table=table_names.get(name, _snake_plural(name)),
                file_path=source.rel_path,
                line=_line_number(source.content, match.start()),
                package=package,
                fields=fields,
            )
        )
    return models


def _go_api_routes(source: SourceFile) -> list[ApiRouteEntry]:
    routes: list[ApiRouteEntry] = []
    route_pattern = re.compile(
        r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s*\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][\w.]*)",
        re.IGNORECASE,
    )
    for match in route_pattern.finditer(source.content):
        routes.append(
            ApiRouteEntry(
                method=match.group(1).upper(),
                path=match.group(2),
                handler=match.group(3),
                file_path=source.rel_path,
                line=_line_number(source.content, match.start()),
                framework="gin/echo",
                middleware=_nearby_middlewares(source.content, match.start()),
            )
        )

    handle_func_pattern = re.compile(
        r"HandleFunc\s*\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][\w.]*)\s*\)(?:\.Methods\(\s*\"([A-Z]+)\"\s*\))?"
    )
    for match in handle_func_pattern.finditer(source.content):
        routes.append(
            ApiRouteEntry(
                method=(match.group(3) or "ANY").upper(),
                path=match.group(1),
                handler=match.group(2),
                file_path=source.rel_path,
                line=_line_number(source.content, match.start()),
                framework="net/http/gorilla",
                middleware=_nearby_middlewares(source.content, match.start()),
            )
        )

    http_pattern = re.compile(r"http\.HandleFunc\s*\(\s*\"([^\"]+)\"\s*,\s*([A-Za-z_][\w.]*)")
    for match in http_pattern.finditer(source.content):
        routes.append(
            ApiRouteEntry(
                method="ANY",
                path=match.group(1),
                handler=match.group(2),
                file_path=source.rel_path,
                line=_line_number(source.content, match.start()),
                framework="net/http",
            )
        )
    return routes


def _build_go_test_mappings(index: CodebaseContextIndex) -> None:
    paths = {entry.path for entry in index.tree}
    go_files = [entry.path for entry in index.tree if entry.path.endswith(".go")]
    tests = [path for path in go_files if path.endswith("_test.go")]
    tests_by_dir: dict[str, list[str]] = {}
    existing = {(item.source_path, item.test_path) for item in index.test_mappings}
    for test in tests:
        tests_by_dir.setdefault(str(Path(test).parent), []).append(test)

    for path in go_files:
        if path.endswith("_test.go"):
            continue
        direct = path[:-3] + "_test.go"
        if direct in paths and (path, direct) not in existing:
            index.test_mappings.append(
                TestFileMapping(
                    source_path=path,
                    test_path=direct,
                    confidence=1.0,
                    reason="same file stem",
                )
            )
            existing.add((path, direct))
            continue
        directory = str(Path(path).parent)
        for test in tests_by_dir.get(directory, []):
            if (path, test) in existing:
                continue
            index.test_mappings.append(
                TestFileMapping(
                    source_path=path,
                    test_path=test,
                    confidence=0.55,
                    reason="same package directory",
                )
            )
            existing.add((path, test))


def _extract_go_calls(body: str) -> list[str]:
    calls: list[str] = []
    excluded = {
        "if",
        "for",
        "switch",
        "select",
        "return",
        "defer",
        "go",
        "func",
        "make",
        "new",
        "append",
        "len",
        "cap",
        "copy",
        "delete",
        "panic",
        "recover",
    }
    for match in re.finditer(r"\b(?:(?P<prefix>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)\s*\(", body):
        name = match.group("name")
        prefix = match.group("prefix")
        if name in excluded:
            continue
        callee = f"{prefix}.{name}" if prefix else name
        if callee not in calls:
            calls.append(callee)
    return calls


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
