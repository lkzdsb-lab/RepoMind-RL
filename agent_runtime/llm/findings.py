"""Finding candidate normalization shared by policy and observation stages."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from utils import _clamp_float, _clean_string_list
from model.finding_updates import FindingChange


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def normalize_finding_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()[:600]
        if not claim:
            continue
        locations = normalize_locations(item.get("locations"))
        severity = str(item.get("severity") or "medium").strip().lower()
        if severity not in _SEVERITY_RANK:
            severity = "medium"
        category = str(item.get("category") or "").strip().lower()[:80]
        findings.append(
            {
                "candidate_id": str(item.get("candidate_id") or "").strip()[:100],
                "claim": claim,
                "locations": locations,
                "related_tests": sorted(set(_clean_string_list(item.get("related_tests"), len(item.get("related_tests") or []), 240))),
                "confidence": _clamp_float(
                    item.get("confidence"), 0.5, "invalid finding confidence"
                ),
                "severity": severity,
                "category": category,
            }
        )
    return findings


def finding_change(old: dict[str, Any], new: dict[str, Any]) -> FindingChange:
    old, new = normalize_finding_candidates([old, new])
    if old["claim"] != new["claim"]:
        return FindingChange.REVISED
    if old["locations"] != new["locations"] or old["related_tests"] != new["related_tests"]:
        return FindingChange.EVIDENCE
    return FindingChange.UNCHANGED if old == new else FindingChange.METADATA


def new_candidate_id(item: dict[str, Any]) -> str:
    identity = json.dumps({"claim": item["claim"].casefold(), "locations": item["locations"]},
                          ensure_ascii=True, sort_keys=True)
    return f"candidate_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"


def shares_source_anchor(first: dict[str, Any], second: dict[str, Any]) -> bool:
    for left in first["locations"]:
        for right in second["locations"]:
            if left["file_path"] != right["file_path"]:
                continue
            if left == right:
                return True
            if left.get("symbol") and left.get("symbol") == right.get("symbol"):
                return True
            if left.get("start_line") and right.get("start_line"):
                if max(left["start_line"], right["start_line"]) <= min(
                    left.get("end_line", left["start_line"]), right.get("end_line", right["start_line"])
                ):
                    return True
    return False


def normalize_locations(value: Any) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return locations
    for raw_location in value:
        if not isinstance(raw_location, dict):
            continue
        file_path = str(raw_location.get("file_path") or "").strip()[:400]
        if not file_path:
            continue
        location: dict[str, Any] = {
            "file_path": file_path,
            "symbol": str(raw_location.get("symbol") or "").strip()[:300],
        }
        for key in ("start_line", "end_line"):
            try:
                line = int(raw_location.get(key) or 0)
            except (TypeError, ValueError):
                line = 0
            if line > 0:
                location[key] = line
        locations.append(location)
    return [json.loads(key) for key in sorted({json.dumps(item, sort_keys=True) for item in locations})]

