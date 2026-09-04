"""Deterministic collection of provenance-rich facts from Task Archive artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

from agent_runtime.memory.consolidation.models import EvidenceItem, TaskEvidenceBundle
from agent_runtime.memory.domain.interfaces import TaskArchiveStore


class ArchiveEvidenceCollector:
    def __init__(self, archive: TaskArchiveStore) -> None:
        self.archive = archive

    def collect(self, task_id: str) -> TaskEvidenceBundle:
        """
            从归档文件中提取证据并赋分
            title: 0.8
            user_input: 0.8
            summary: 0.6
            verification: 测试正常范围 1；测试返回异常 code 0.85；测试未正确执行完 0.5
            diff: 0.9
            source_code: 0.75
        """
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
        if candidate_payload.get("schema_version") != 2:
            raise ValueError(
                "memory_candidates.json must use schema_version 2; "
                "legacy Task Archives are not supported"
            )
        runtime_candidates = _dicts(candidate_payload.get("candidates"))
        source_payload = _object(
            self.archive.read_json_artifact(task_id, "source_snapshots")
        )
        session_context = _object(
            self.archive.read_json_artifact(task_id, "session_snapshot")
        )

        evidence: list[EvidenceItem] = []
        title = str(state.get("title") or "").strip()
        description = str(state.get("description") or "").strip()
        objective = " ".join(value for value in (title, description) if value)
        if title:
            evidence.append(
                _evidence(
                    "user_statement",
                    title,
                    "final_state.json#/title",
                    0.8,
                    {"statement_position": "initial"},
                    source_event_ids=("task:current",),
                )
            )
        for index, item in enumerate(_dicts(state.get("user_inputs"))):
            statement = str(
                item.get("answer") or item.get("content") or item.get("input") or ""
            ).strip()
            if not statement:
                continue
            evidence.append(
                _evidence(
                    "user_statement",
                    statement,
                    f"final_state.json#/user_inputs/{index}",
                    0.8,
                    {"statement_position": "follow_up"},
                    source_event_ids=(f"user_input:{index}",),
                )
            )

        summary = str(report.get("summary") or state.get("error") or "").strip()
        if summary:
            evidence.append(
                _evidence("final_report", summary, "final_report.json#/summary", 0.6)
            )

        for group in ("test_results", "verification_commands", "command_results"):
            for index, result in enumerate(_dicts(verification.get(group))):
                evidence.append(_verification_evidence(result, group, index))

        if "diff" in manifest.files:
            diff = self.archive.read_text_artifact(task_id, "diff").strip()
            if diff:
                changed_files = tuple(_strings(state.get("edited_files"), 50))
                evidence.append(
                    _evidence(
                        "diff",
                        _bounded(diff, 2400),
                        "diff.patch",
                        0.9,
                        {"changed_files": list(changed_files)},
                        files=changed_files,
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
                        files=(path,),
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
            source_event_ids = tuple(
                _strings(candidate.get("source_event_ids"), 20)
            )
            files = tuple(_strings(candidate.get("files"), 30))
            symbols = tuple(_strings(candidate.get("symbols"), 30))
            commands = _strings(candidate.get("commands"), 10)
            evidence.append(
                _evidence(
                    "runtime_candidate",
                    content,
                    f"memory_candidates.json#/candidates/{index}",
                    0.6,
                    {
                        "candidate_origin": str(candidate.get("origin_kind") or ""),
                        "candidate_schema_version": candidate.get("schema_version"),
                    },
                    source_event_ids=source_event_ids,
                    files=files,
                    symbols=symbols,
                    command=commands[0] if commands else "",
                )
            )

        files = _unique(
            source_files
            + _strings(state.get("edited_files"), 50)
            + _strings(state.get("candidate_files"), 50)
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


def _verification_evidence(
    result: dict[str, Any],
    group: str,
    index: int,
) -> EvidenceItem:
    command = str(result.get("command") or "verification").strip()
    exit_code = result.get("exit_code")
    summary = f"{command} exit_code={exit_code}"
    output = str(result.get("stderr") or result.get("stdout") or "").strip()
    if output:
        summary += f": {_bounded(output, 1200)}"
    strength = 1.0 if exit_code == 0 else 0.85 if exit_code is not None else 0.5
    files = tuple(
        _unique(
            _strings(result.get("files"), 30)
            + _strings(result.get("changed_files"), 30)
        )
    )
    return _evidence(
        "verification",
        summary,
        f"verification.json#/{group}/{index}",
        strength,
        {"exit_code": exit_code, "command": command},
        files=files,
        command=command,
    )


def _evidence(
    kind: str,
    summary: str,
    reference: str,
    strength: float,
    metadata: dict[str, Any] | None = None,
    *,
    source_event_ids: tuple[str, ...] = (),
    files: tuple[str, ...] = (),
    symbols: tuple[str, ...] = (),
    command: str = "",
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
        source_event_ids=tuple(dict.fromkeys(source_event_ids)),
        files=tuple(dict.fromkeys(files)),
        symbols=tuple(dict.fromkeys(symbols)),
        command=str(command or "").strip(),
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
    return list({item.evidence_id: item for item in values}.values())


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return _unique([str(item).strip() for item in value if str(item).strip()])[:limit]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _bounded(value: str, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
