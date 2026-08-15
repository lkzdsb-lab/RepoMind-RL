"""Data models for codebase context indexes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from utils import utc_now


@dataclass
class RepoTreeEntry:
    path: str
    language: str
    size_bytes: int
    lines: int
    package: str = ""
    layer: str = "unknown"


@dataclass
class SymbolEntry:
    name: str
    kind: str
    file_path: str
    line: int
    package: str = ""
    receiver: str = ""
    signature: str = ""
    layer: str = "unknown"

    @property
    def full_name(self) -> str:
        if self.receiver:
            return f"{self.receiver}.{self.name}"
        if self.package:
            return f"{self.package}.{self.name}"
        return self.name


@dataclass
class FunctionEntry:
    name: str
    full_name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    package: str = ""
    receiver: str = ""
    layer: str = "unknown"
    calls: list[str] = field(default_factory=list)


@dataclass
class ApiRouteEntry:
    method: str
    path: str
    handler: str
    file_path: str
    line: int
    framework: str = "unknown"
    middleware: list[str] = field(default_factory=list)


@dataclass
class DbModelField:
    name: str
    type: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class DbModelEntry:
    name: str
    table: str
    file_path: str
    line: int
    package: str = ""
    fields: list[DbModelField] = field(default_factory=list)


@dataclass
class CallGraphEdge:
    caller: str
    callee: str
    file_path: str
    line: int
    kind: str = "call"


@dataclass
class TestFileMapping:
    source_path: str
    test_path: str
    confidence: float
    reason: str


@dataclass
class EmbeddingDoc:
    doc_id: str
    kind: str
    title: str
    content: str
    file_path: str = ""
    symbol: str = ""
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodebaseContextIndex:
    repo_path: str
    created_at: str = field(default_factory=utc_now)
    tree: list[RepoTreeEntry] = field(default_factory=list)
    symbols: list[SymbolEntry] = field(default_factory=list)
    functions: list[FunctionEntry] = field(default_factory=list)
    api_routes: list[ApiRouteEntry] = field(default_factory=list)
    db_models: list[DbModelEntry] = field(default_factory=list)
    call_graph: list[CallGraphEdge] = field(default_factory=list)
    test_mappings: list[TestFileMapping] = field(default_factory=list)
    embeddings: list[EmbeddingDoc] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CodebaseContextIndex":
        return cls(
            repo_path=str(data.get("repo_path", "")),
            created_at=str(data.get("created_at", utc_now())),
            tree=[RepoTreeEntry(**item) for item in data.get("tree", [])],
            symbols=[SymbolEntry(**item) for item in data.get("symbols", [])],
            functions=[FunctionEntry(**item) for item in data.get("functions", [])],
            api_routes=[ApiRouteEntry(**item) for item in data.get("api_routes", [])],
            db_models=[
                DbModelEntry(
                    **{
                        **item,
                        "fields": [
                            DbModelField(**field)
                            for field in item.get("fields", [])
                        ],
                    }
                )
                for item in data.get("db_models", [])
            ],
            call_graph=[CallGraphEdge(**item) for item in data.get("call_graph", [])],
            test_mappings=[
                TestFileMapping(**item)
                for item in data.get("test_mappings", [])
            ],
            embeddings=[EmbeddingDoc(**item) for item in data.get("embeddings", [])],
            metadata=dict(data.get("metadata", {})),
        )

@dataclass
class CodeContextRerankDecision:
    selected_ids: list[str] = field(default_factory=list)
    source: str = "disabled"
    rationale: str = ""
    selections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class CodeContextQueryPlan:
    queries: list[str]
    source: str = "disabled"
    rationale: str = ""
    default_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
