"""Finding candidate normalization shared by policy and observation stages."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from utils import _clamp_float, _clean_string_list


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def normalize_finding_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in value[:20]:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()[:600]
        if not claim:
            continue
        locations = _normalize_locations(item.get("locations"))
        severity = str(item.get("severity") or "medium").strip().lower()
        if severity not in _SEVERITY_RANK:
            severity = "medium"
        category = str(item.get("category") or "").strip().lower()[:80]
        identity = json.dumps(
            {"claim": claim.lower(), "locations": locations},
            ensure_ascii=True,
            sort_keys=True,
        )
        findings.append(
            {
                "candidate_id": (
                    f"candidate_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"
                ),
                "claim": claim,
                "locations": locations,
                "related_tests": _clean_string_list(item.get("related_tests"), 8, 240),
                "confidence": _clamp_float(
                    item.get("confidence"), 0.5, "invalid finding confidence"
                ),
                "severity": severity,
                "category": category,
            }
        )
    return findings


def merge_finding_candidates(existing: Any, incoming: Any) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in normalize_finding_candidates(existing) + normalize_finding_candidates(incoming):
        candidate_id = item["candidate_id"]
        if candidate_id not in merged:
            merged[candidate_id] = item
            order.append(candidate_id)
            continue
        current = merged[candidate_id]
        current["confidence"] = max(current["confidence"], item["confidence"])
        if _SEVERITY_RANK[item["severity"]] > _SEVERITY_RANK[current["severity"]]:
            current["severity"] = item["severity"]
        current["related_tests"] = _merge_unique(
            current.get("related_tests"), item.get("related_tests"), limit=8
        )
        if not current.get("category") and item.get("category"):
            current["category"] = item["category"]
    return [merged[candidate_id] for candidate_id in order][-20:]


def _normalize_locations(value: Any) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return locations
    for raw_location in value[:8]:
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
    return locations


def _merge_unique(first: Any, second: Any, *, limit: int) -> list[str]:
    merged: list[str] = []
    for value in list(first or []) + list(second or []):
        text = str(value).strip()
        if text and text not in merged:
            merged.append(text)
    return merged[:limit]
