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
        findings.append(
            {
                "candidate_id": str(item.get("candidate_id") or "").strip()[:100],
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
    stored = normalize_finding_candidates(existing)
    for index, item in enumerate(stored + normalize_finding_candidates(incoming)):
        requested_id = item["candidate_id"]
        current = merged.get(requested_id)
        if current is not None and _shares_location(current, item):
            candidate_id = requested_id
        else:
            # Text-identical findings at the same source anchor are safe to reuse.
            # A file/function match alone is not enough to merge different bugs.
            candidate_id = next((key for key, value in merged.items()
                                 if value["claim"].casefold() == item["claim"].casefold()
                                 and _shares_location(value, item)), "")
            if not candidate_id:
                candidate_id = requested_id if index < len(stored) and requested_id else _new_candidate_id(item)
            if candidate_id in merged and not _shares_location(merged[candidate_id], item):
                candidate_id = _new_candidate_id(item)
        item["candidate_id"] = candidate_id
        if candidate_id not in merged:
            merged[candidate_id] = item
            order.append(candidate_id)
            continue
        current = merged[candidate_id]
        current["claim"] = item["claim"]
        for location in item["locations"]:
            if location not in current["locations"]:
                current["locations"].append(location)
        current["locations"] = current["locations"][:8]
        current["confidence"] = max(current["confidence"], item["confidence"])
        if _SEVERITY_RANK[item["severity"]] > _SEVERITY_RANK[current["severity"]]:
            current["severity"] = item["severity"]
        current["related_tests"] = _merge_unique(
            current.get("related_tests"), item.get("related_tests"), limit=8
        )
        if not current.get("category") and item.get("category"):
            current["category"] = item["category"]
    return [merged[candidate_id] for candidate_id in order][-20:]


def _new_candidate_id(item: dict[str, Any]) -> str:
    identity = json.dumps({"claim": item["claim"].casefold(), "locations": item["locations"]},
                          ensure_ascii=True, sort_keys=True)
    return f"candidate_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"


def _shares_location(first: dict[str, Any], second: dict[str, Any]) -> bool:
    for left in first["locations"]:
        for right in second["locations"]:
            if left["file_path"] != right["file_path"]:
                continue
            if left.get("symbol") and left.get("symbol") == right.get("symbol"):
                return True
            if left.get("start_line") and right.get("start_line"):
                if max(left["start_line"], right["start_line"]) <= min(
                    left.get("end_line", left["start_line"]), right.get("end_line", right["start_line"])
                ):
                    return True
    return False


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
