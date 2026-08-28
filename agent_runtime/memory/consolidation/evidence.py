"""Deterministic collection of trustworthy facts from Task Archive artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

from agent_runtime.memory.domain.interfaces import TaskArchiveStore
from agent_runtime.memory.consolidation.models import EvidenceItem, TaskEvidenceBundle


class ArchiveEvidenceCollector:
    def __init__(self, archive: TaskArchiveStore) -> None:
        self.archive = archive

    def collect(self, task_id: str) -> TaskEvidenceBundle:
        errors = self.archive.verify(task_id)
        if errors:
            raise ValueError(f"Task Archive integrity check failed: {'; '.join(errors)}")
        manifest = self.archive.load_manifest(task_id)
        state = _object(self.archive.read_json_artifact(task_id, "final_state"))
        report = _object(self.archive.read_json_artifact(task_id, "final_report"))
        verification = _object(self.archive.read_json_artifact(task_id, "verification"))
        candidate_payload = _object(
            self.archive.read_json_artifact(task_id, "memory_candidates")
        )
        runtime_candidates = _dicts(candidate_payload.get("candidates"))
        source_payload = _object(
            self.archive.read_json_artifact(task_id, "source_snapshots")
        )
        session_context = _object(
            self.archive.read_json_artifact(task_id, "session_snapshot")
        )

        evidence: list[EvidenceItem] = []
        summary = str(report.get("summary") or state.get("error") or "").strip()
        if summary:
            evidence.append(
                _evidence("final_report", summary, "final_report.json#/summary", 0.6)
            )

        for index, result in enumerate(_dicts(verification.get("test_results"))):
            evidence.append(_verification_evidence(result, "test_results", index))
        for index, result in enumerate(
            _dicts(verification.get("verification_commands"))
        ):
            evidence.append(
                _verification_evidence(result, "verification_commands", index)
            )
        for index, result in enumerate(_dicts(verification.get("command_results"))):
            evidence.append(_verification_evidence(result, "command_results", index))

        if "diff" in manifest.files:
            diff = self.archive.read_text_artifact(task_id, "diff").strip()
            if diff:
                evidence.append(
                    _evidence(
                        "diff",
                        _bounded(diff, 2400),
                        "diff.patch",
                        0.9,
                        {"changed_files": _strings(state.get("edited_files"), 50)},
                    )
                )

        source_files: list[str] = []
        for index, item in enumerate(_dicts(source_payload.get("files"))):
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            source_files.append(path)
            if item.get("source_sha256"):
                evidence.append(
                    _evidence(
                        "source",
                        f"Archived source snapshot for {path}",
                        f"source_snapshots.json#/files/{index}",
                        0.75,
                        {
                            "file_path": path,
                            "source_sha256": item.get("source_sha256"),
                        },
                    )
                )

        actions = _action_sequence(state.get("trajectory"))
        if actions:
            evidence.append(
                _evidence(
                    "trajectory",
                    " -> ".join(actions),
                    "events.jsonl",
                    0.55,
                    {"actions": actions},
                )
            )

        for index, candidate in enumerate(runtime_candidates):
            content = str(candidate.get("content") or "").strip()
            if not content:
                continue
            evidence.append(
                _evidence(
                    "runtime_candidate",
                    content,
                    f"memory_candidates.json#/candidates/{index}",
                    0.6,
                    {
                        "candidate_type": str(candidate.get("type") or ""),
                        "source_event_id": str(candidate.get("source_event_id") or ""),
                    },
                )
            )

        files = _unique(
            source_files
            + _strings(state.get("edited_files"), 50)
            + _strings(state.get("candidate_files"), 50)
        )
        objective = " ".join(
            value
            for value in (
                str(state.get("title") or "").strip(),
                str(state.get("description") or "").strip(),
            )
            if value
        )
        return TaskEvidenceBundle(
            task_id=task_id,
            repo_id=manifest.repo_id,
            archive_path=self.archive.task_path(task_id).as_posix(),
            archive_hash=self.archive.manifest_hash(task_id),
            repo_revision=manifest.repo_revision,
            status=manifest.status,
            objective=_bounded(objective, 3000),
            outcome=_bounded(summary, 3000),
            files=tuple(files),
            evidence=tuple(_dedupe_evidence(evidence)),
            runtime_candidates=tuple(runtime_candidates),
            session_context=session_context,
        )


def _verification_evidence(result: dict[str, Any], group: str, index: int) -> EvidenceItem:
    command = str(result.get("command") or "verification").strip()
    exit_code = result.get("exit_code")
    summary = f"{command} exit_code={exit_code}"
    output = str(result.get("stderr") or result.get("stdout") or "").strip()
    if output:
        summary += f": {_bounded(output, 1200)}"
    strength = 1.0 if exit_code == 0 else 0.85 if exit_code is not None else 0.5
    return _evidence(
        "verification",
        summary,
        f"verification.json#/{group}/{index}",
        strength,
        {"exit_code": exit_code, "command": command},
    )


def _evidence(
    kind: str,
    summary: str,
    reference: str,
    strength: float,
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem:
    fingerprint = hashlib.sha256(
        f"{kind}\0{reference}\0{summary}".encode("utf-8")
    ).hexdigest()[:20]
    return EvidenceItem(
        evidence_id=f"ev_{fingerprint}",
        kind=kind,
        summary=_bounded(summary, 4000),
        reference=reference,
        strength=max(0.0, min(1.0, float(strength))),
        metadata=dict(metadata or {}),
    )


def _action_sequence(value: Any) -> list[str]:
    actions: list[str] = []
    for item in _dicts(value):
        action = str(item.get("action") or item.get("node") or "").strip()
        if action and (not actions or actions[-1] != action):
            actions.append(action)
        if len(actions) >= 30:
            break
    return actions


def _dedupe_evidence(values: list[EvidenceItem]) -> list[EvidenceItem]:
    result: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in values:
        if item.evidence_id not in seen:
            result.append(item)
            seen.add(item.evidence_id)
    return result


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique([str(item).strip() for item in value if str(item).strip()])[:limit]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _bounded(value: str, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
