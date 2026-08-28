"""Rule-based candidate recall before semantic extraction."""

from __future__ import annotations

import hashlib
from typing import Any

from agent_runtime.memory.consolidation.models import (
    MemoryCandidate,
    TaskEvidenceBundle,
)


_TYPE_MAP = {
    "code_fact": "semantic",
    "semantic": "semantic",
    "procedure": "procedural",
    "procedural": "procedural",
    "verification_failure": "anti_pattern",
    "error_pattern": "anti_pattern",
    "anti_pattern": "anti_pattern",
    "user_constraint": "preference",
    "preference": "preference",
    "episodic": "episodic",
}


class RuleBasedCandidateBuilder:
    def __init__(self, max_candidates: int = 24) -> None:
        self.max_candidates = max(1, int(max_candidates))

    def build(self, bundle: TaskEvidenceBundle) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        all_evidence = tuple(item.evidence_id for item in bundle.evidence)
        if bundle.objective or bundle.outcome:
            candidates.append(
                _candidate(
                    "episodic",
                    _join(bundle.objective, bundle.outcome),
                    all_evidence[:12],
                    "task_summary",
                    bundle.files,
                )
            )

        evidence_by_ref = {str(item.reference): item.evidence_id for item in bundle.evidence}
        for index, raw in enumerate(bundle.runtime_candidates):
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            memory_type = _TYPE_MAP.get(str(raw.get("type") or "").strip(), "semantic")
            evidence_ids = _candidate_evidence_ids(
                raw, index, memory_type, bundle, evidence_by_ref
            )
            candidates.append(
                _candidate(
                    memory_type,
                    content,
                    evidence_ids,
                    "runtime_candidate",
                    _candidate_files(raw, bundle.files),
                )
            )

        verification = [
            item for item in bundle.evidence if item.kind == "verification"
        ]
        trajectory = next(
            (item for item in bundle.evidence if item.kind == "trajectory"), None
        )
        if trajectory and any(item.metadata.get("exit_code") == 0 for item in verification):
            candidates.append(
                _candidate(
                    "procedural",
                    f"Successful task workflow: {trajectory.summary}",
                    tuple([trajectory.evidence_id] + [
                        item.evidence_id
                        for item in verification
                        if item.metadata.get("exit_code") == 0
                    ]),
                    "successful_trajectory",
                    bundle.files,
                )
            )

        return _dedupe(candidates)[: self.max_candidates]


def _candidate(
    memory_type: str,
    content: str,
    evidence_ids: tuple[str, ...],
    source: str,
    files: tuple[str, ...],
) -> MemoryCandidate:
    clean_content = str(content or "").strip()[:6000]
    fingerprint = hashlib.sha256(
        f"{memory_type}\0{source}\0{clean_content}".encode("utf-8")
    ).hexdigest()[:20]
    return MemoryCandidate(
        candidate_id=f"mc_{fingerprint}",
        suggested_type=memory_type,
        content=clean_content,
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        source=source,
        files=tuple(files[:30]),
    )


def _candidate_evidence_ids(
    raw: dict[str, Any],
    index: int,
    memory_type: str,
    bundle: TaskEvidenceBundle,
    evidence_by_ref: dict[str, str],
) -> tuple[str, ...]:
    matching: list[str] = []
    candidate_ref = f"memory_candidates.json#/candidates/{index}"
    if candidate_ref in evidence_by_ref:
        matching.append(evidence_by_ref[candidate_ref])
    for item in bundle.evidence:
        include = (
            memory_type == "semantic" and item.kind in {"source", "diff"}
            or memory_type == "procedural"
            and (
                item.kind == "trajectory"
                or item.kind == "verification" and item.metadata.get("exit_code") == 0
            )
            or memory_type == "anti_pattern"
            and item.kind == "verification"
            and item.metadata.get("exit_code") not in (None, 0)
        )
        if include:
            matching.append(item.evidence_id)
    return tuple(dict.fromkeys(matching[:8]))


def _candidate_files(raw: dict[str, Any], fallback: tuple[str, ...]) -> tuple[str, ...]:
    content = str(raw.get("content") or "")
    matched = tuple(path for path in fallback if path and path in content)
    return matched or fallback[:10]


def _dedupe(values: list[MemoryCandidate]) -> list[MemoryCandidate]:
    return list({item.candidate_id: item for item in values}.values())


def _join(*values: str) -> str:
    return "\n".join(value for value in values if value).strip()
