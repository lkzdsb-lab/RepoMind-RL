"""Shared, evidence-aware online scope interpretation (not consolidation identity)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from agent_runtime.memory.domain.models import MemoryScope, ScopeLevel

SCOPE_POLICY_VERSION = "scope-v2"


def _strings(value: Any, limit: int = 50) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip()[:500] for item in value[:limit] if str(item).strip()))


def relative_name(value: str) -> str:
    """Lexical normalization preserves case and rejects escaping/absolute paths."""
    text = str(value).strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or ":" in text:
        return ""
    return path.as_posix() if path.as_posix() != "." else ""


@dataclass(frozen=True, order=True)
class ScopedSymbol:
    file: str
    name: str


@dataclass(frozen=True)
class ScopeContext:
    repo_id: str
    files: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    symbols: tuple[ScopedSymbol, ...] = ()
    text_hints: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def query_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.files, *self.modules, *(s.name for s in self.symbols))))


class ScopeContextBuilder:
    def __init__(self, repo_path: str, repo_id: str) -> None:
        self.root = Path(repo_path).resolve()
        self.repo_id = repo_id

    def _file(self, value: str, directories: dict[Path, set[str]]) -> str:
        name = relative_name(value)
        if not name:
            return ""
        # No global scan; verify only explicitly supplied paths, preserving actual case.
        current = self.root
        try:
            for component in PurePosixPath(name).parts:
                if current not in directories:
                    directories[current] = {entry.name for entry in current.iterdir()}
                names = directories[current]
                if component not in names:
                    return ""
                current = current / component
                if not current.resolve().is_relative_to(self.root):
                    return ""
            resolved = current.resolve()
            if not resolved.is_relative_to(self.root) or not resolved.is_file():
                return ""
            return name
        except OSError:
            return ""

    def build(self, state: Mapping[str, Any]) -> ScopeContext:
        """
        - Current repository ID.
        - Explicit candidate paths verified against the current filesystem, preserving
          case, normalizing separators and rejecting paths outside the repository.
        - Directory ancestors of confirmed paths (not guessed package names).
        - File-qualified symbols from selected code context, falling back to retrieved
          code context. The existing symbol record's file and line are rechecked for its
          name. This is a bounded location check, not an AST or semantic proof.
        - Unverified entities/search hints and ambiguous filenames/symbol names.
        """
        analysis = state.get("task_analysis")
        analysis = analysis if isinstance(analysis, dict) else {}
        candidates = _strings(state.get("candidate_files"))
        hints = list(dict.fromkeys((*_strings(analysis.get("entities")),
                                  *_strings(analysis.get("search_hints")), *candidates)))
        checked: dict[str, str] = {}
        directories: dict[Path, set[str]] = {}

        def verified(value: str) -> str:
            if value not in checked:
                checked[value] = self._file(value, directories)
            return checked[value]

        files = {name for value in candidates if (name := verified(value))}
        symbols: set[ScopedSymbol] = set()
        context = state.get("selected_code_context") or state.get("code_context") or {}
        if isinstance(context, dict):
            records = context.get("symbols", [])
            for record in (records[:100] if isinstance(records, list) else []):
                if not isinstance(record, dict):
                    continue
                file = verified(str(record.get("file_path") or ""))
                name = str(record.get("name") or "").strip()
                try:
                    line = int(record.get("line") or 0)
                    if not file or not name or not 1 <= line <= 10000:
                        continue
                    # Recheck the indexed location; do not treat a bare LLM entity as a symbol.
                    with (self.root / file).open(encoding="utf-8", errors="replace") as source:
                        text = ""
                        for number, text in enumerate(source, 1):
                            if number == line:
                                break
                        else:
                            continue
                    if not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
                        continue
                except (OSError, ValueError, TypeError):
                    continue
                owner = str(record.get("receiver") or record.get("package") or "").strip()
                qualified = f"{owner}.{name}" if owner else name
                files.add(file)
                symbols.add(ScopedSymbol(file, qualified))
        modules = {str(parent) for file in files for parent in PurePosixPath(file).parents if str(parent) != "."}
        ambiguities = [f"unverified_file:{value}" for value in candidates if not verified(value)]
        for basename in {PurePosixPath(file).name for file in files}:
            if sum(PurePosixPath(file).name == basename for file in files) > 1:
                ambiguities.append(f"ambiguous_basename:{basename}")
        for name in {symbol.name for symbol in symbols}:
            if sum(symbol.name == name for symbol in symbols) > 1:
                ambiguities.append(f"ambiguous_symbol:{name}")
        return ScopeContext(self.repo_id, tuple(sorted(files)), tuple(sorted(modules)),
                            tuple(sorted(symbols)), tuple(hints), tuple(sorted(ambiguities)))


@dataclass(frozen=True)
class ScopeMatch:
    status: Literal["matched", "unknown", "mismatched"]
    reason: str
    matched_by: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class ScopeMatcher:
    def match(self, scope: MemoryScope, context: ScopeContext) -> ScopeMatch:
        if scope.level in {ScopeLevel.USER, ScopeLevel.GLOBAL}:
            return ScopeMatch("matched", "shared_scope", (scope.level.value,))
        if scope.repo_id != context.repo_id:
            return ScopeMatch("mismatched", "different_repository")
        if scope.level == ScopeLevel.REPO:
            return ScopeMatch("matched", "same_repository", (context.repo_id,))
        if scope.level == ScopeLevel.FILE:
            matched = tuple(sorted(set(filter(None, map(relative_name, scope.files))) & set(context.files)))
        elif scope.level == ScopeLevel.MODULE:
            module = relative_name(scope.module)
            # Dotted module names are accepted only when the derived directory is confirmed.
            alternatives = {module, relative_name(scope.module.replace(".", "/"))} - {""}
            matched = tuple(sorted(alternatives & set(context.modules)))
        elif scope.level == ScopeLevel.SYMBOL:
            files = set(filter(None, map(relative_name, scope.files)))
            matched = tuple(sorted(f"{item.file}:{item.name}" for item in context.symbols
                                   if item.file in files and item.name in scope.symbols))
        else:
            return ScopeMatch("mismatched", "unsupported_scope")
        if matched:
            return ScopeMatch("matched", "confirmed_location", matched)
        return ScopeMatch("unknown", "insufficient_or_ambiguous_location")
