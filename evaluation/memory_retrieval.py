"""Isolated Recall@K evaluation using the production retrieval service."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_runtime.memory.archive.repository import repository_id
from agent_runtime.memory.catalog import SQLiteMemoryCatalog
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.domain.models import MemoryDocument
from agent_runtime.memory.retrieval.service import LongTermMemoryService
from agent_runtime.memory.semantic.embeddings import build_embedding_client
from agent_runtime.memory.semantic.sqlite import SQLiteMemorySemanticIndex
from config import debug_agent_config_from_dict, load_config_payload, validate_memory_retrieval_config, load_env_file
from evaluation.workspace import copy_fixture, snapshot_digest, snapshot_workspace


class RetrievalQuery(BaseModel):
    query_id: str
    phase: str = "refined"
    state: dict[str, Any]
    relevant_memory_ids: list[str] = Field(min_length=1)
    forbidden_memory_ids: list[str] = Field(default_factory=list)
    expected_scope_status: dict[str, Literal["matched", "unknown", "mismatched"]] = Field(default_factory=dict)


class MemoryRetrievalCase(BaseModel):
    kind: Literal["memory_retrieval"]
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    version: int = 1
    fixture: str
    documents: list[dict[str, Any]] = Field(min_length=1)
    queries: list[RetrievalQuery] = Field(min_length=1)
    modes: list[Literal["keyword", "semantic", "hybrid"]] = Field(default_factory=lambda: ["keyword", "semantic", "hybrid"], min_length=1)
    k: int = Field(default=5, ge=1, le=100)
    minimum_recall: float = Field(default=0.0, ge=0, le=1)


def recall_at_k(expected: list[str], actual: list[str], k: int) -> float:
    gold = set(expected)
    if not gold:
        raise ValueError("Recall@K requires at least one relevant memory")
    ranked = list(dict.fromkeys(actual))[:k]
    return len(gold.intersection(ranked)) / len(gold)


def run_memory_evaluation(case_path: str | Path, *, config_path: str | Path,
                          keep_workspace: bool = False, timeout_override: int | None = None) -> dict:
    # Share the existing evaluator's run identity and secret-safe fingerprinting.
    from evaluation.runner import _run_id, _config_fingerprint

    case = MemoryRetrievalCase.model_validate_json(Path(case_path).read_text(encoding="utf-8"))
    project = Path(__file__).resolve().parent.parent
    run_id = _run_id()
    run_dir = project / ".repomind" / "evaluation" / case.case_id / run_id
    workspace = run_dir / "workspace"
    copy_fixture((project / case.fixture).resolve(), workspace)
    payload = load_config_payload(config_path, require_exists=True)
    from evaluation.worker import _env_path
    load_env_file(_env_path(Path(config_path).resolve(), payload), override=False)
    config = debug_agent_config_from_dict(payload)
    config.repo_path = str(workspace)
    config.long_term_memory_path = str(run_dir / "memory")
    config.memory_catalog_path = str(run_dir / "catalog.sqlite3")
    config.memory_semantic_index_path = str(run_dir / "semantic.sqlite3")
    config.long_term_memory_retrieval_enabled = True
    if timeout_override is not None:
        config.long_term_memory_semantic_timeout = timeout_override
    validate_memory_retrieval_config(config)
    if case.k > config.long_term_memory_retrieval_limit:
        raise ValueError("evaluation k exceeds configured retrieval_limit")
    store = MarkdownMemoryDocumentStore.from_config(config)
    catalog = SQLiteMemoryCatalog.from_config(config)
    documents = []
    for item in case.documents:
        data = dict(item)
        scope = dict(data["scope"])
        if scope.get("repo_id") == "$current":
            scope["repo_id"] = repository_id(workspace)
        data["scope"] = scope
        documents.append(MemoryDocument.from_dict(data))
    ids = {doc.memory_id for doc in documents}
    if len(ids) != len(documents) or len({q.query_id for q in case.queries}) != len(case.queries):
        raise ValueError("duplicate document or query IDs")
    for query in case.queries:
        if not set(query.relevant_memory_ids) <= ids:
            raise ValueError(f"unknown relevant memory in {query.query_id}")
        if not (set(query.forbidden_memory_ids) | set(query.expected_scope_status)) <= ids:
            raise ValueError(f"unknown scope assertion memory in {query.query_id}")
        if set(query.relevant_memory_ids) & set(query.forbidden_memory_ids):
            raise ValueError(f"conflicting gold labels in {query.query_id}")
    for document in documents:
        catalog.synchronize(document, store.save(document))
    setup_error = ""
    model_fingerprint = ""
    if any(mode != "keyword" for mode in case.modes):
        try:
            client = build_embedding_client(config.memory_embedding_config)
            if client is None:
                raise ValueError("embedding provider disabled")
            model_fingerprint = client.model_fingerprint
            SQLiteMemorySemanticIndex.from_config(config, embedding_client=client).rebuild(documents)
        except Exception as exc:
            setup_error = str(exc)
    rows = []
    failures = []
    traces = []
    for mode in dict.fromkeys(case.modes):
        config.long_term_memory_retrieval_mode = mode
        service = LongTermMemoryService.from_config(config)
        for query in case.queries:
            batch = service.retrieve(query.state, phase=query.phase)
            actual = [memory.memory_id for memory in batch.memories][:case.k]
            errors = list(batch.warnings)
            forbidden = sorted(set(query.forbidden_memory_ids) & {m.memory_id for m in batch.memories})
            if forbidden:
                errors.append(f"forbidden memories injected: {forbidden}")
            scope_context = service.scope_builder.build(query.state)
            scope_results = {
                doc.memory_id: service.scope_matcher.match(doc.scope, scope_context).status
                for doc in documents if doc.memory_id in query.expected_scope_status
            }
            for memory_id, expected in query.expected_scope_status.items():
                if scope_results.get(memory_id) != expected:
                    errors.append(f"scope {memory_id}: expected {expected}, got {scope_results.get(memory_id)}")
            if mode != "keyword" and setup_error:
                errors.append(setup_error)
            if errors:
                failures.append(f"{mode}/{query.query_id}: " + "; ".join(errors))
            rows.append({"mode": mode, "query_id": query.query_id,
                         "recall": recall_at_k(query.relevant_memory_ids, actual, case.k),
                         "returned": actual,
                         "forbidden_returned": forbidden,
                         "scope_assertions": scope_results,
                         "missing": sorted(set(query.relevant_memory_ids) - set(actual)),
                         "errors": errors})
            traces.append({"mode": mode, "query_id": query.query_id, **batch.to_dict()})
    means = {mode: sum(row["recall"] for row in rows if row["mode"] == mode) / len(case.queries)
             for mode in dict.fromkeys(case.modes)}
    result = {"kind": case.kind, "case_id": case.case_id, "run_id": run_id,
              "metric": f"Recall@{case.k}", "recall": means, "queries": rows,
              "passed": not failures and all(value >= case.minimum_recall for value in means.values()),
              "failures": failures, "config_fingerprint": _config_fingerprint(Path(config_path)),
              "fixture_hash": snapshot_digest(snapshot_workspace(workspace)),
              "artifacts": {"report": str(run_dir / "report.md"), "run_dir": str(run_dir)}}
    artifacts = {
        "request.json": {"case": case.model_dump(), "model_fingerprint": model_fingerprint,
                         "retrieval": {field.name: getattr(config, field.name)
                                       for field in fields(config)
                                       if field.name.startswith("long_term_memory_")}},
        "corpus_manifest.json": [{"memory_id": doc.memory_id,
                                   "content_hash": store.content_hash(doc.memory_id),
                                   "scope": doc.scope.to_dict()} for doc in documents],
        "result.json": result,
    }
    for filename, content in artifacts.items():
        (run_dir / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "retrievals.jsonl").write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces), encoding="utf-8")
    report = [f"# Memory evaluation: {case.case_id}", "", f"Passed: {result['passed']}", "",
              *[f"- {mode}: Recall@{case.k} = {value:.4f}" for mode, value in means.items()],
              "", "## Queries", "",
              *[f"- {row['mode']}/{row['query_id']}: {row['recall']:.4f}; missing={row['missing']}" for row in rows],
              "", "## Failures", "", *[f"- {failure}" for failure in failures]]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Keep the small isolated corpus/workspace for exact replay and diagnosis.
    return result
