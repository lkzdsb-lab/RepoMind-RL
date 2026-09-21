"""Recall adapters, bounded semantic execution, and deterministic rank fusion."""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

from agent_runtime.memory.domain.models import MemoryHit, MemoryQuery, normalize_memory_knowledge


@dataclass
class ChannelResult:
    hits: dict[str, list[MemoryHit]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class KeywordRetriever:
    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def retrieve(self, queries: list[tuple[str, MemoryQuery]]) -> ChannelResult:
        started = time.monotonic()
        result = ChannelResult()
        for label, query in queries:
            try:
                result.hits[label] = self.catalog.keyword_search(query)
            except Exception as exc:
                result.warnings.append(f"keyword query {label!r} failed: {exc}")
        result.diagnostics = {"elapsed_ms": (time.monotonic() - started) * 1000,
                              "status": "partial" if result.warnings else "ok"}
        return result


class SemanticRetriever:
    """At most one worker per service; timed-out work cannot accumulate or publish results."""
    def __init__(self, index: Any, *, timeout: float = 5, min_score: float = 0.5,
                 candidate_limit: int = 24, cache_size: int = 256) -> None:
        self.index = index
        self.timeout = float(timeout)
        self.min_score = float(min_score)
        self.candidate_limit = max(1, min(100, int(candidate_limit)))
        self.cache_size = max(1, int(cache_size))
        # 缓存已经 embedding 过的 query
        self._cache: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
        self._busy = threading.Lock()

    def retrieve(self, queries: list[tuple[str, MemoryQuery]]) -> ChannelResult:
        started = time.monotonic()
        if not self._busy.acquire(blocking=False):
            return ChannelResult(warnings=["semantic retrieval busy after previous timeout"],
                                 diagnostics={"status": "busy"})
        done = threading.Event()
        outputs: list[ChannelResult] = []
        deadline = started + self.timeout

        # 开线程去异步获取 embedding 结果
        def run() -> None:
            try:
                outputs.append(self._retrieve(queries, deadline))
            except Exception as exc:
                outputs.append(ChannelResult(warnings=[f"semantic retrieval failed: {exc}"],
                                             diagnostics={"status": "failed"}))
            finally:
                self._busy.release()
                done.set()

        threading.Thread(target=run, name="memory-semantic-recall", daemon=True).start()
        # 超时则直接返回 error
        if not done.wait(max(0, deadline - time.monotonic())):
            return ChannelResult(warnings=["semantic retrieval timed out; results discarded"],
                                 diagnostics={"status": "timeout", "elapsed_ms": self.timeout * 1000})
        result = outputs[0]
        result.diagnostics["elapsed_ms"] = (time.monotonic() - started) * 1000
        return result

    def _retrieve(self, queries: list[tuple[str, MemoryQuery]], deadline: float) -> ChannelResult:
        """ 请求 embedding 服务"""
        if not queries:
            return ChannelResult(diagnostics={"status": "ok", "scanned_vectors": 0})
        if not self.index.path.is_file():
            raise ValueError("semantic index missing; run memory semantic rebuild")
        client = self.index.embedding_client
        texts = list(dict.fromkeys(normalize_memory_knowledge(q.text) for _, q in queries))
        keys = {text: (client.model_fingerprint, hashlib.sha256(text.encode()).hexdigest()) for text in texts}
        vectors = {text: self._cache[key] for text, key in keys.items() if key in self._cache}
        cache_hits = len(vectors)
        for text in vectors:
            self._cache.move_to_end(keys[text])
        missing = [text for text in texts if text not in vectors]
        if missing:
            batch = client.embed(missing)
            if batch.model_fingerprint != client.model_fingerprint or len(batch.vectors) != len(missing):
                raise ValueError("query embedding response does not match requested model or count")
            if time.monotonic() >= deadline:
                raise TimeoutError("query embedding deadline exceeded")
            for text, vector in zip(missing, batch.vectors):
                vectors[text] = vector
                self._cache[keys[text]] = vector
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        result = ChannelResult(diagnostics={"status": "ok", "cache_hits": cache_hits,
                                            "embedded_texts": len(missing), "scanned_vectors": 0})
        for label, query in queries:
            hits, scanned = self.index.retrieve_vector(
                replace(query, limit=self.candidate_limit),
                vectors[normalize_memory_knowledge(query.text)],
                min_score=self.min_score, deadline=deadline,
            )
            result.hits[label] = hits
            result.diagnostics["scanned_vectors"] += scanned
        return result


def validated_candidates(results: dict[str, ChannelResult], store: Any, *, mode: str,
                         min_keyword_score: float, rrf_k: int, semantic_weight: float,
                         keyword_weight: float, warnings: list[str]) -> tuple[list, dict, list]:
    """Read each document once; remove stale channel contributions before fusion."""
    documents: dict = {}
    hashes: dict = {}
    merged: dict = {}
    rejected: list = []
    for channel, result in results.items():
        channel_hits: dict = {}
        # 对命中的记忆分数决断
        for label, hits in result.hits.items():
            for hit in hits:
                if channel == "keyword" and hit.score < min_keyword_score:
                    rejected.append({"memory_id": hit.memory_id, "channel": channel,
                                     "selected": False, "skip_reason": "below_min_score"})
                    continue
                current = channel_hits.get(hit.memory_id)
                if current is None:
                    channel_hits[hit.memory_id] = {"hit": hit, "queries": [label], "reasons": list(hit.reasons)}
                else:
                    current["queries"].append(label)
                    current["reasons"].extend(hit.reasons)
                    if hit.score > current["hit"].score:
                        current["hit"] = hit
        valid = []
        # 读取 Markdown，剔除过期索引贡献
        for memory_id, record in channel_hits.items():
            hit = record["hit"]
            try:
                if memory_id not in documents:
                    documents[memory_id] = store.get(memory_id)
                    if documents[memory_id] is not None:
                        hashes[memory_id] = store.content_hash(memory_id)
                document = documents[memory_id]
                if document is None:
                    raise ValueError("missing_markdown")
                if channel == "keyword":
                    if not hit.content_hash or hit.content_hash != hashes[memory_id]:
                        raise ValueError("stale_projection")
                elif hit.knowledge_hash != document.knowledge_hash or hit.revision != document.revision:
                    raise ValueError("stale_semantic_projection")
            except Exception as exc:
                warnings.append(f"{channel} memory {memory_id}: {exc}")
                rejected.append({"memory_id": memory_id, "channel": channel,
                                 "selected": False, "skip_reason": str(exc)})
                continue
            valid.append(record)
        valid.sort(key=lambda record: (-record["hit"].score, record["hit"].memory_id))
        # 去重并融合排名
        #根据模式处理：
        #- keyword：使用关键词通道分数。
        #- semantic：使用向量相似度。
        #- hybrid：使用 RRF
        #融合两个通道的排名。
        for rank, record in enumerate(valid, 1):
            hit = record["hit"]
            entry = merged.setdefault(hit.memory_id, {
                "hit": replace(hit, content_hash=hashes[hit.memory_id]), "score": 0.0,
                "queries": [], "reasons": [], "channels": {},
            })
            weight = keyword_weight if channel == "keyword" else semantic_weight
            entry["score"] += weight / (rrf_k + rank) if mode == "hybrid" else hit.score
            entry["queries"].extend(f"{channel}:{label}" for label in record["queries"])
            entry["reasons"].extend(record["reasons"] + [f"{channel} rank {rank}"])
            entry["channels"][channel] = {"rank": rank, "score": hit.score}
    return sorted(merged.items(), key=lambda pair: (-pair[1]["score"], pair[0])), documents, rejected
