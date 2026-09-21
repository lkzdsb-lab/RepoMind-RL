"""One source of truth for findings and executable acceptance scenarios."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import EvaluationCase, ExpectedFinding

def load_baseline(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("bugs"):
        raise ValueError("unsupported or empty Bug baseline")
    ids = [item["bug_id"] for item in data["bugs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate baseline Bug IDs")
    return data

def resolve_case_baseline(case: EvaluationCase, project_root: Path) -> EvaluationCase:
    if not case.bug_baseline:
        return case
    if case.expected_findings:
        raise ValueError("use bug_baseline or inline expected_findings, not both")
    path = (project_root / case.bug_baseline).resolve()
    baseline = load_baseline(path)
    selected = [item for item in baseline["bugs"]
                if not case.bug_difficulties or item["difficulty"] in case.bug_difficulties]
    if not selected:
        raise ValueError("no baseline Bugs match requested difficulties")
    fields = set(ExpectedFinding.model_fields)
    return case.model_copy(update={"expected_findings": [
        ExpectedFinding.model_validate({key: value for key, value in item.items() if key in fields})
        for item in selected
    ]})
