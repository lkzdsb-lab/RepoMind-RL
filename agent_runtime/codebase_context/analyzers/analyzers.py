"""Pluggable analyzers for language-agnostic codebase context.

The builder owns scanning and persistence. Analyzers only add facts to the
shared CodebaseContextIndex schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_runtime.codebase_context.models import (
    CodebaseContextIndex,
    EmbeddingDoc,
    SymbolEntry,
)
from utils import _tokens

@dataclass
class SourceFile:
    path: Path
    rel_path: str
    language: str
    content: str
    layer: str
    package: str = ""


class CodeAnalyzer(Protocol):
    name: str

    def supports(self, source: SourceFile) -> bool:
        ...

    def analyze_file(self, index: CodebaseContextIndex, source: SourceFile) -> None:
        ...

    def finalize(self, index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
        ...


class AnalyzerRegistry:
    def __init__(self, analyzers: list[CodeAnalyzer]) -> None:
        self.analyzers = analyzers

    @classmethod
    def default(cls) -> "AnalyzerRegistry":
        from agent_runtime.codebase_context.analyzers.GoAnalyzer import (
            GoLanguageAnalyzer,
            GoWebFrameworkAnalyzer,
        )
        from agent_runtime.codebase_context.analyzers.PythonAnalyzer import PythonLanguageAnalyzer

        return cls(
            [
                GenericAnalyzer(),
                GoLanguageAnalyzer(),
                GoWebFrameworkAnalyzer(),
                PythonLanguageAnalyzer(),
            ]
        )

    def analyze(self, index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
        used: set[str] = set()
        for source in sources:
            for analyzer in self.analyzers:
                if analyzer.supports(source):
                    analyzer.analyze_file(index, source)
                    used.add(analyzer.name)
        for analyzer in self.analyzers:
            analyzer.finalize(index, sources)
        index.metadata["analyzers"] = sorted(used)

class GenericAnalyzer:
    """Language-agnostic analyzer for top-level file and directory hints."""

    name = "generic"

    def supports(self, source: SourceFile) -> bool:
        return True

    def analyze_file(self, index: CodebaseContextIndex, source: SourceFile) -> None:
        return None

    def finalize(self, index: CodebaseContextIndex, sources: list[SourceFile]) -> None:
        index.metadata["generic_layers"] = _layer_counts(index)
        index.metadata["generic_languages"] = _language_counts(index)

def _test_candidates(path: str) -> list[tuple[str, str, float]]:
    p = Path(path)
    stem = p.stem
    suffix = p.suffix
    parent = p.parent.as_posix()
    same_dir = "" if parent == "." else parent + "/"
    candidates = [
        (f"{same_dir}{stem}_test{suffix}", "same file stem", 1.0),
        (f"{same_dir}test_{stem}{suffix}", "same file stem", 1.0),
        (f"{same_dir}{stem}.test{suffix}", "same file stem", 1.0),
        (f"{same_dir}{stem}.spec{suffix}", "same file stem", 1.0),
    ]
    if parent != ".":
        candidates.extend(
            [
                (f"tests/{path}", "tests mirror path", 0.75),
                (f"test/{path}", "test mirror path", 0.75),
                (f"__tests__/{path}", "__tests__ mirror path", 0.75),
            ]
        )
    return candidates


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        "/test/" in lowered
        or "/tests/" in lowered
        or "/__tests__/" in lowered
        or lowered.endswith(("_test.py", "_test.go", ".test.ts", ".spec.ts", ".test.js", ".spec.js"))
        or Path(path).name.startswith("test_")
    )


def _symbol(source: SourceFile, name: str, kind: str, offset: int, package: str) -> SymbolEntry:
    return SymbolEntry(
        name=name,
        kind=kind,
        file_path=source.rel_path,
        line=_line_number(source.content, offset),
        package=package,
        signature=_line_at(source.content, offset).strip(),
        layer=source.layer,
    )


def _module_name(path: str) -> str:
    p = Path(path)
    without_suffix = p.with_suffix("").as_posix()
    return without_suffix.replace("/", ".")


def _layer_counts(index: CodebaseContextIndex) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in index.tree:
        counts[entry.layer] = counts.get(entry.layer, 0) + 1
    return counts


def _language_counts(index: CodebaseContextIndex) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in index.tree:
        counts[entry.language] = counts.get(entry.language, 0) + 1
    return counts


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


def _nearby_middlewares(content: str, offset: int) -> list[str]:
    window = content[max(0, offset - 800) : offset]
    middlewares: list[str] = []
    for match in re.finditer(r"\.Use\s*\(([^)]*)\)", window):
        for item in match.group(1).split(","):
            item = item.strip()
            if item:
                middlewares.append(item)
    return middlewares[-5:]


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
