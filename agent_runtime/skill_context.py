from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def build_selected_skill_context(
    selected_skills: list[str],
    skills: Any,
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for skill_name in selected_skills:
        spec = skills.get(skill_name) if hasattr(skills, "get") else None
        if spec is None:
            continue
        context.append(_summarize_skill_spec(skill_name, spec))
    return context


def _summarize_skill_spec(skill_name: str, spec: Any) -> dict[str, Any]:
    resources = [Path(str(item)) for item in getattr(spec, "resources", [])]
    entry_path = _entry_resource_path(resources)
    summary: dict[str, Any] = {
        "source": "registry",
        "skill_name": skill_name,
        "description": str(getattr(spec, "description", "") or "")[:400],
        "triggers": [str(item)[:80] for item in getattr(spec, "triggers", [])[:8]],
        "entrypoints": {
            str(key): str(value)[:120]
            for key, value in dict(getattr(spec, "entrypoints", {}) or {}).items()
        },
        "artifacts": _artifact_summary(entry_path, resources),
    }
    if entry_path is not None:
        summary["entry"] = _summarize_entry(entry_path)
    else:
        summary["entry"] = {"error": "entry resource not found"}
    return summary


def _entry_resource_path(resources: list[Path]) -> Path | None:
    for path in resources:
        if path.name == "SKILL.md":
            return path
    for path in resources:
        if path.suffix.lower() == ".md":
            return path
    for path in resources:
        if path.is_file():
            return path
    return None


def _artifact_summary(entry_path: Path | None, resources: list[Path]) -> dict[str, Any]:
    root = _skill_root(entry_path)
    if root is None:
        return {
            "layout": "resource_only",
            "resource_files": [path.as_posix() for path in resources[:4]],
        }
    skill_md = root / "SKILL.md"
    references_dir = root / "references"
    scripts_dir = root / "scripts"
    agents_yaml = root / "agents" / "openai.yaml"
    if skill_md.is_file() or references_dir.is_dir() or scripts_dir.is_dir() or agents_yaml.is_file():
        return {
            "layout": "standard_skill",
            "root": root.as_posix(),
            "has_skill_md": skill_md.is_file(),
            "has_openai_yaml": agents_yaml.is_file(),
            "has_scripts_dir": scripts_dir.is_dir(),
            "reference_files": _relative_files(references_dir, root, limit=6),
            "script_files": _relative_files(scripts_dir, root, limit=6),
        }
    return {
        "layout": "flat_markdown",
        "root": root.as_posix(),
        "resource_files": [path.name for path in resources[:4]],
    }


def _skill_root(entry_path: Path | None) -> Path | None:
    if entry_path is None:
        return None
    if entry_path.name == "SKILL.md":
        return entry_path.parent
    return entry_path.parent


def _relative_files(directory: Path, root: Path, limit: int) -> list[str]:
    if not directory.is_dir():
        return []
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    result: list[str] = []
    for path in files[:limit]:
        try:
            result.append(path.relative_to(root).as_posix())
        except ValueError:
            result.append(path.as_posix())
    return result


def _summarize_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.as_posix(), "error": "resource not found"}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"path": path.as_posix(), "error": str(exc)}
    body, frontmatter = _split_frontmatter(text)
    title = _first_heading(body) or frontmatter.get("name") or path.stem
    return {
        "path": path.as_posix(),
        "title": str(title)[:160],
        "description": str(frontmatter.get("description") or _first_paragraph(body))[:400],
        "sections": _headings(body, limit=6),
        "rules": _rule_snippets(body, limit=6),
    }


def _split_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    stripped = text.lstrip()
    if not stripped.startswith("---\n"):
        return text, {}
    lines = stripped.splitlines()
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        return text, {}
    data: dict[str, str] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    body = "\n".join(lines[end_index + 1 :])
    return body, data


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _first_paragraph(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _headings(text: str, limit: int) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = stripped.lstrip("#").strip()
        if heading and heading not in headings:
            headings.append(heading[:120])
        if len(headings) >= limit:
            break
    return headings


def _rule_snippets(text: str, limit: int) -> list[str]:
    snippets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(\d+\.\s+|[-*]\s+)", stripped):
            cleaned = re.sub(r"^(\d+\.\s+|[-*]\s+)", "", stripped).strip()
            if cleaned and cleaned not in snippets:
                snippets.append(cleaned[:180])
        if len(snippets) >= limit:
            break
    return snippets
