"""Exact provenance resolution between candidate seeds and archive evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from agent_runtime.memory.consolidation.models import (
    CandidateSeed,
    EvidenceItem,
    EvidenceLink,
    EvidenceResolution,
    MemoryCandidate,
)


class EvidenceIndex:
    def __init__(self, evidence: Iterable[EvidenceItem]) -> None:
        self.by_id: dict[str, EvidenceItem] = {}
        self.by_reference: dict[str, list[EvidenceItem]] = defaultdict(list)
        self.by_event_id: dict[str, list[EvidenceItem]] = defaultdict(list)
        self.by_file: dict[str, list[EvidenceItem]] = defaultdict(list)
        self.by_symbol: dict[str, list[EvidenceItem]] = defaultdict(list)
        self.by_command: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in evidence:
            self.by_id[item.evidence_id] = item
            self.by_reference[item.reference].append(item)
            for event_id in item.source_event_ids:
                self.by_event_id[event_id].append(item)
            for path in item.files:
                self.by_file[_path_key(path)].append(item)
            for symbol in item.symbols:
                self.by_symbol[_text_key(symbol)].append(item)
            if item.command:
                self.by_command[_text_key(item.command)].append(item)


class CandidateEvidenceResolver:
    """Resolve only explicit, exact provenance relationships; never topical similarity."""

    def resolve_all(
        self,
        seeds: Iterable[CandidateSeed],
        evidence: Iterable[EvidenceItem],
    ) -> tuple[list[MemoryCandidate], list[EvidenceResolution]]:
        """ 精确绑定候选与证据"""
        index = EvidenceIndex(evidence)
        candidates: list[MemoryCandidate] = []
        resolutions: list[EvidenceResolution] = []
        for seed in seeds:
            candidate, resolution = self.resolve(seed, index)
            candidates.append(candidate)
            resolutions.append(resolution)
        return candidates, resolutions

    def resolve(
        self,
        seed: CandidateSeed,
        index: EvidenceIndex,
    ) -> tuple[MemoryCandidate, EvidenceResolution]:
        links: dict[str, EvidenceLink] = {}
        unresolved: list[str] = []

        for reference in seed.evidence_refs:
            matches = index.by_reference.get(reference, [])
            if not matches:
                unresolved.append(reference)
            _add_links(links, matches, "explicit_reference")
        for event_id in seed.source_event_ids:
            _add_links(
                links,
                (
                    item
                    for item in index.by_event_id.get(event_id, [])
                    if item.kind != "runtime_candidate"
                ),
                "same_source_event",
            )
        for path in seed.files:
            _add_links(
                links,
                (
                    item
                    for item in index.by_file.get(_path_key(path), [])
                    if item.kind in {"source", "diff", "verification"}
                ),
                "same_file",
            )
        for symbol in seed.symbols:
            _add_links(
                links,
                (
                    item
                    for item in index.by_symbol.get(_text_key(symbol), [])
                    if item.kind in {"source", "diff"}
                ),
                "same_symbol",
            )
        for command in seed.commands:
            _add_links(
                links,
                (
                    item
                    for item in index.by_command.get(_text_key(command), [])
                    if item.kind == "verification"
                ),
                "same_command",
            )

        ordered = tuple(links.values())
        candidate = MemoryCandidate(
            candidate_id=seed.candidate_id,
            origin_kind=seed.origin_kind,
            content=seed.content,
            evidence_ids=tuple(item.evidence_id for item in ordered),
            evidence_links=ordered,
            unresolved_references=tuple(dict.fromkeys(unresolved)),
            files=seed.files,
            symbols=seed.symbols,
        )
        return candidate, EvidenceResolution(
            candidate_id=seed.candidate_id,
            links=ordered,
            unresolved_references=tuple(dict.fromkeys(unresolved)),
        )


def _add_links(
    result: dict[str, EvidenceLink],
    evidence: Iterable[EvidenceItem],
    relation: str,
) -> None:
    for item in evidence:
        result.setdefault(
            item.evidence_id,
            EvidenceLink(evidence_id=item.evidence_id, relation=relation),
        )


def _path_key(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _text_key(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()
