import re

from agent_runtime.codebase_context import CodebaseContextIndex
from agent_runtime.codebase_context.analyzers.analyzers import SourceFile, _module_name, _embedding_doc, _symbol, _line_number, \
    _line_at, _is_test_path, _test_candidates
from agent_runtime.codebase_context.models import SymbolEntry, FunctionEntry, TestFileMapping


class PythonLanguageAnalyzer:
    """Python language analyzer for modules, imports, classes, functions, and tests."""

    name = "python"

    def supports(self, source: SourceFile) -> bool:
        return source.path.suffix == ".py"

    def analyze_file(self, index: CodebaseContextIndex, source: SourceFile) -> None:
        package = _module_name(source.rel_path)
        for symbol in _python_symbols(source, package):
            index.symbols.append(symbol)
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"symbol:{symbol.file_path}:{symbol.full_name}:{symbol.line}",
                    kind="symbol",
                    title=symbol.full_name,
                    content=f"{symbol.kind} {symbol.signature}",
                    file_path=symbol.file_path,
                    symbol=symbol.full_name,
                    metadata={"language": source.language, "layer": source.layer, "package": package},
                )
            )
        for function in _python_functions(source, package):
            index.functions.append(function)
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"function:{function.file_path}:{function.full_name}:{function.start_line}",
                    kind="function",
                    title=function.full_name,
                    content=function.signature,
                    file_path=function.file_path,
                    symbol=function.full_name,
                    metadata={"language": source.language, "layer": source.layer, "package": package},
                )
            )
        for imported in _python_imports(source):
            index.embeddings.append(
                _embedding_doc(
                    doc_id=f"import:{source.rel_path}:{imported}",
                    kind="import",
                    title=imported,
                    content=f"{source.rel_path} imports {imported}",
                    file_path=source.rel_path,
                    metadata={"language": source.language, "layer": source.layer},
                )
            )

    def finalize(self, index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
        _build_python_test_mappings(index, sources)
        python_sources = [source for source in sources if source.path.suffix == ".py"]
        if python_sources:
            index.metadata["python_modules"] = len(python_sources)


def _python_symbols(source: SourceFile, package: str) -> list[SymbolEntry]:
    symbols: list[SymbolEntry] = []
    for match in re.finditer(r"(?m)^class\s+([A-Za-z_]\w*)\b", source.content):
        symbols.append(_symbol(source, match.group(1), "class", match.start(), package))
    for match in re.finditer(r"(?m)^([A-Z][A-Z0-9_]+)\s*=", source.content):
        symbols.append(_symbol(source, match.group(1), "constant", match.start(), package))
    for imported in _python_imports(source):
        symbols.append(
            SymbolEntry(
                name=imported.split(".")[-1],
                kind="import",
                file_path=source.rel_path,
                line=1,
                package=package,
                signature=f"import {imported}",
                layer=source.layer,
            )
        )
    return symbols


def _python_functions(source: SourceFile, package: str) -> list[FunctionEntry]:
    functions: list[FunctionEntry] = []
    pattern = re.compile(r"(?m)^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)")
    class_pattern = re.compile(r"(?m)^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_]\w*)\b")
    classes = [
        (
            match.start(),
            len(match.group("indent").replace("\t", "    ")),
            match.group("name"),
        )
        for match in class_pattern.finditer(source.content)
    ]

    for match in pattern.finditer(source.content):
        indent = len(match.group("indent").replace("\t", "    "))
        name = match.group("name")
        receiver = _nearest_python_class(classes, match.start(), indent)
        full_name = f"{package}.{receiver}.{name}" if receiver else f"{package}.{name}"
        line = _line_number(source.content, match.start())
        functions.append(
            FunctionEntry(
                name=name,
                full_name=full_name,
                file_path=source.rel_path,
                start_line=line,
                end_line=line,
                signature=_line_at(source.content, match.start()).strip(),
                package=package,
                receiver=receiver,
                layer=source.layer,
                calls=[],
            )
        )
    return functions


def _nearest_python_class(classes: list[tuple[int, int, str]], offset: int, indent: int) -> str:
    receiver = ""
    receiver_indent = -1
    for class_offset, class_indent, class_name in classes:
        if class_offset >= offset:
            break
        if class_indent < indent and class_indent >= receiver_indent:
            receiver = class_name
            receiver_indent = class_indent
    return receiver


def _python_imports(source: SourceFile) -> list[str]:
    imports: list[str] = []
    for match in re.finditer(r"(?m)^\s*import\s+([A-Za-z_][\w.]*)", source.content):
        value = match.group(1)
        if value not in imports:
            imports.append(value)
    for match in re.finditer(r"(?m)^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+", source.content):
        value = match.group(1)
        if value not in imports:
            imports.append(value)
    return imports


def _build_python_test_mappings(index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
    py_paths = {source.rel_path for source in sources if source.path.suffix == ".py"}
    existing = {(item.source_path, item.test_path) for item in index.test_mappings}
    for source in sources:
        if source.path.suffix != ".py":
            continue
        rel = source.rel_path
        if _is_test_path(rel):
            continue
        candidates = _test_candidates(rel)
        for candidate, reason, confidence in candidates:
            if candidate in py_paths and (rel, candidate) not in existing:
                index.test_mappings.append(
                    TestFileMapping(
                        source_path=rel,
                        test_path=candidate,
                        confidence=confidence,
                        reason=reason,
                    )
                )
                existing.add((rel, candidate))
                break
