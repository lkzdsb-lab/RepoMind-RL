"""Build provenance-preserving candidate seeds before semantic extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_runtime.memory.consolidation.models import CandidateSeed, TaskEvidenceBundle


RUNTIME_CANDIDATE_SCHEMA_VERSION = 2

_ORIGIN_KINDS = {
    "code_observation",
    "verification_failure",
    "error_pattern",
    "observer_observation",
}


class CandidateSeedBuilder:
    """Recall candidate seeds without deciding their final memory semantics."""

    def __init__(self, max_candidates: int = 24) -> None:
        self.max_candidates = max(1, int(max_candidates))

    def build(self, bundle: TaskEvidenceBundle) -> list[CandidateSeed]:
        """ 找出值得进一步分析的候选内容"""
        seeds: list[CandidateSeed] = []
        evidence_by_kind: dict[str, list[Any]] = {}
        for item in bundle.evidence:
            evidence_by_kind.setdefault(item.kind, []).append(item)

        for item in evidence_by_kind.get("user_statement", []):
            seeds.append(
                _seed(
                    origin_kind="user_statement",
                    content=item.summary,
                    evidence_refs=(item.reference,),
                    source_event_ids=item.source_event_ids,
                )
            )

        task_evidence = evidence_by_kind.get("user_statement", [])
        report_evidence = evidence_by_kind.get("final_report", [])
        if bundle.objective or bundle.outcome:
            refs = tuple(item.reference for item in task_evidence + report_evidence)
            seeds.append(
                _seed(
                    origin_kind="task_outcome",
                    content=_join(bundle.objective, bundle.outcome),
                    evidence_refs=refs,
                )
            )

        for index, raw in enumerate(bundle.runtime_candidates):
            seeds.append(_runtime_seed(raw, index))

        trajectory = evidence_by_kind.get("trajectory", [])
        passed = [
            item
            for item in evidence_by_kind.get("verification", [])
            if item.metadata.get("exit_code") == 0
        ]
        if trajectory and passed:
            related = trajectory[:1] + passed
            seeds.append(
                _seed(
                    origin_kind="successful_workflow",
                    content=f"Successful task workflow: {trajectory[0].summary}",
                    evidence_refs=tuple(item.reference for item in related),
                    files=tuple(
                        dict.fromkeys(path for item in related for path in item.files)
                    ),
                    commands=tuple(
                        dict.fromkeys(item.command for item in passed if item.command)
                    ),
                )
            )

        return _dedupe(seeds)[: self.max_candidates]


def _runtime_seed(raw: dict[str, Any], index: int) -> CandidateSeed:
    version = raw.get("schema_version")
    if version != RUNTIME_CANDIDATE_SCHEMA_VERSION:
        raise ValueError(
            "memory_candidates.json uses an unsupported candidate schema at "
            f"index {index}: expected {RUNTIME_CANDIDATE_SCHEMA_VERSION}, got {version!r}"
        )
    origin_kind = str(raw.get("origin_kind") or "").strip()
    if origin_kind not in _ORIGIN_KINDS:
        raise ValueError(
            f"memory candidate {index} has unsupported origin_kind {origin_kind!r}"
        )
    content = str(raw.get("content") or "").strip()
    if not content:
        raise ValueError(f"memory candidate {index} has empty content")
    own_ref = f"memory_candidates.json#/candidates/{index}"
    return _seed(
        origin_kind=origin_kind,
        content=content,
        evidence_refs=(own_ref, *_strings(raw.get("evidence_refs"), 20, 1000)),
        source_event_ids=_strings(raw.get("source_event_ids"), 20, 300),
        files=_strings(raw.get("files"), 30, 500),
        symbols=_strings(raw.get("symbols"), 30, 500),
        commands=_strings(raw.get("commands"), 10, 1000),
    )


def _seed(
    *,
    origin_kind: str,
    content: str,
    evidence_refs: tuple[str, ...] = (),
    source_event_ids: tuple[str, ...] = (),
    files: tuple[str, ...] = (),
    symbols: tuple[str, ...] = (),
    commands: tuple[str, ...] = (),
) -> CandidateSeed:
    clean_content = str(content or "").strip()[:6000]
    identity = json.dumps(
        {
            "origin_kind": origin_kind,
            "content": clean_content,
            "evidence_refs": list(evidence_refs),
            "source_event_ids": list(source_event_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return CandidateSeed(
        candidate_id=f"mc_{fingerprint}",
        origin_kind=origin_kind,
        content=clean_content,
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        source_event_ids=tuple(dict.fromkeys(source_event_ids)),
        files=tuple(dict.fromkeys(files)),
        symbols=tuple(dict.fromkeys(symbols)),
        commands=tuple(dict.fromkeys(commands)),
    )


def _strings(value: Any, limit: int, max_chars: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(
            str(item or "").strip()[:max_chars]
            for item in value
            if str(item or "").strip()
        )
    )[:limit]


def _dedupe(values: list[CandidateSeed]) -> list[CandidateSeed]:
    return list({item.candidate_id: item for item in values}.values())


def _join(*values: str) -> str:
    return "\n".join(value for value in values if value).strip()
