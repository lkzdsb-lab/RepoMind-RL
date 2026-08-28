"""Strict, deterministic Markdown codec for MemoryDocument."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.memory.documents.migrations import MemorySchemaMigrator
from agent_runtime.memory.domain.models import MemoryDocument
from agent_runtime.memory.io import compact_json, sha256_bytes


_FRONTMATTER_ORDER = (
    "schema_version",
    "memory_id",
    "memory_type",
    "title",
    "status",
    "scope",
    "source",
    "triggers",
    "tags",
    "confidence",
    "evidence_strength",
    "created_at",
    "updated_at",
    "evidence",
    "extensions",
)
_BODY_SECTIONS = ("knowledge", "applicability", "evidence", "invalidation")


class MarkdownMemoryCodec:
    def __init__(self, migrator: MemorySchemaMigrator | None = None) -> None:
        self.migrator = migrator or MemorySchemaMigrator()

    def encode(self, document: MemoryDocument) -> bytes:
        _validate_markdown_text(document)
        data = document.to_dict()
        knowledge = str(data.pop("knowledge"))
        applicability = str(data.pop("applicability"))
        invalidation = str(data.pop("invalidation"))
        lines = ["---"]
        lines.extend(
            f"{key}: {compact_json(data[key])}"
            for key in _FRONTMATTER_ORDER
        )
        lines.extend(
            [
                "---",
                "",
                f"# {document.title}",
                "",
                *_render_section("Knowledge", "knowledge", knowledge),
                *_render_section("Applicability", "applicability", applicability),
                *_render_section(
                    "Evidence",
                    "evidence",
                    _render_evidence(data["evidence"]),
                ),
                *_render_section(
                    "Invalidation",
                    "invalidation",
                    invalidation or "(none)",
                ),
            ]
        )
        return ("\n".join(lines).rstrip() + "\n").encode("utf-8")

    def decode(self, content: str | bytes) -> MemoryDocument:
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        metadata, body = _parse_frontmatter(normalized)
        metadata = self.migrator.migrate(metadata)
        sections = {
            name: _extract_section(body, name)
            for name in _BODY_SECTIONS
        }
        expected_evidence = _render_evidence(metadata.get("evidence") or [])
        if sections["evidence"] != expected_evidence:
            raise ValueError("Evidence section does not match structured evidence")
        title = str(metadata.get("title") or "")
        if _document_heading(body) != title:
            raise ValueError("Markdown title does not match front matter title")
        metadata.update(
            {
                "knowledge": sections["knowledge"],
                "applicability": sections["applicability"],
                "invalidation": ""
                if sections["invalidation"] == "(none)"
                else sections["invalidation"],
            }
        )
        return MemoryDocument.from_dict(metadata)

    def content_hash(self, document: MemoryDocument) -> str:
        return sha256_bytes(self.encode(document))

    def is_canonical(self, content: str | bytes) -> bool:
        raw = content.encode("utf-8") if isinstance(content, str) else content
        try:
            return self.encode(self.decode(raw)) == raw
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return False


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("Markdown memory must start with YAML front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("Markdown memory front matter is not closed") from exc
    metadata: dict[str, Any] = {}
    for line in lines[1:end]:
        key, separator, raw_value = line.partition(":")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"invalid front matter line: {line!r}")
        if key in metadata:
            raise ValueError(f"duplicate front matter field: {key}")
        try:
            metadata[key] = json.loads(raw_value.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"front matter field {key!r} is not JSON-inline YAML") from exc
    missing = [key for key in _FRONTMATTER_ORDER if key not in metadata]
    if missing:
        raise ValueError(f"missing front matter fields: {', '.join(missing)}")
    return metadata, "\n".join(lines[end + 1 :]).strip("\n")


def _render_section(title: str, key: str, content: str) -> list[str]:
    return [
        f"## {title}",
        f"<!-- repomind:{key}:start -->",
        str(content).strip(),
        f"<!-- repomind:{key}:end -->",
        "",
    ]


def _extract_section(body: str, key: str) -> str:
    start = f"<!-- repomind:{key}:start -->"
    end = f"<!-- repomind:{key}:end -->"
    if body.count(start) != 1 or body.count(end) != 1:
        raise ValueError(f"Markdown memory requires exactly one {key} section")
    before, _, remainder = body.partition(start)
    content, separator, _ = remainder.partition(end)
    if not separator or before.find(f"## {key.title()}") < 0:
        raise ValueError(f"Markdown memory {key} section is malformed")
    return content.strip()


def _document_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _render_evidence(evidence: list[dict[str, Any]]) -> str:
    if not isinstance(evidence, list) or any(
        not isinstance(item, dict) for item in evidence
    ):
        raise ValueError("structured evidence must be a list of objects")
    if not evidence:
        return "- (none)"
    return "\n".join(f"- {compact_json(item)}" for item in evidence)


def _validate_markdown_text(document: MemoryDocument) -> None:
    if "\n" in document.title or "\r" in document.title:
        raise ValueError("memory title must be a single line")
    reserved = [
        f"<!-- repomind:{key}:{boundary} -->"
        for key in _BODY_SECTIONS
        for boundary in ("start", "end")
    ]
    for field_name, value in (
        ("knowledge", document.knowledge),
        ("applicability", document.applicability),
        ("invalidation", document.invalidation),
    ):
        if any(marker in value for marker in reserved):
            raise ValueError(f"{field_name} contains a reserved RepoMind marker")
