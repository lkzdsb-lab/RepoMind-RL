"""Range-based session file cache with lazy invalidation and SLRU metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


NORMAL = "normal"
PROTECTED = "protected"


def normalize_snapshot(value: Any, *, file_path: str = "") -> dict[str, Any]:
    snapshot = dict(value) if isinstance(value, dict) else {}
    path = str(snapshot.get("file_path") or file_path).strip()
    spans = _normalize_spans(snapshot.get("spans"))
    legacy_content = snapshot.get("content")
    if not spans and isinstance(legacy_content, str) and legacy_content:
        start = max(1, _as_int(snapshot.get("start_line"), 1))
        end = max(start, _as_int(snapshot.get("end_line"), 0))
        if end <= start:
            end = start + max(0, len(legacy_content.splitlines()) - 1)
        spans = [make_span(start, end, legacy_content, access_seq=0)]
    snapshot.pop("content", None)
    snapshot["file_path"] = path
    snapshot["spans"] = spans
    snapshot["dirty_ranges"] = merge_ranges(snapshot.get("dirty_ranges"))
    snapshot["access_seq"] = max(
        _as_int(snapshot.get("access_seq"), 0),
        max((_as_int(span.get("access_seq"), 0) for span in spans), default=0),
    )
    snapshot["size_bytes"] = snapshot_size_bytes(snapshot)
    return snapshot


def make_span(
    start_line: int,
    end_line: int,
    content: str,
    *,
    access_seq: int,
    segment: str = NORMAL,
) -> dict[str, Any]:
    start = max(1, int(start_line))
    end = max(start, int(end_line))
    text = str(content or "")
    return {
        "start_line": start,
        "end_line": end,
        "content": text,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "segment": segment if segment in {NORMAL, PROTECTED} else NORMAL,
        "access_seq": max(0, int(access_seq)),
        "size_bytes": len(text.encode("utf-8")),
    }


def next_access_seq(cache: Any, current: int = 0) -> int:
    maximum = max(0, int(current or 0))
    if isinstance(cache, dict):
        for raw_snapshot in cache.values():
            if not isinstance(raw_snapshot, dict):
                continue
            maximum = max(maximum, _as_int(raw_snapshot.get("access_seq"), 0))
            for span in raw_snapshot.get("spans", []) or []:
                if isinstance(span, dict):
                    maximum = max(maximum, _as_int(span.get("access_seq"), 0))
    return maximum + 1


def cache_read_result(
    state: dict[str, Any],
    output: dict[str, Any],
    *,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_path = str(output.get("file_path") or "").strip()
    if not file_path or output.get("error"):
        return {}
    cache = dict(state.get("read_file_cache") or {})
    revision = str(output.get("file_revision") or "").strip()
    previous = normalize_snapshot(cache.get(file_path), file_path=file_path)
    if str(previous.get("file_revision") or "") != revision:
        previous["spans"] = []
        previous["dirty_ranges"] = []

    access_seq = next_access_seq(cache, _as_int(state.get("file_cache_access_seq"), 0))
    last_read = {
        key: output.get(key)
        for key in ("file_path", "file_revision", "start_line", "end_line", "requested_end_line", "truncated", "source")
    }
    returned_content = str(output.get("content") or "")
    last_read["complete_end_line"] = _as_int(output.get("end_line"), 0)
    if output.get("truncated") and not returned_content.endswith(("\n", "\r")):
        last_read["complete_end_line"] -= 1
    if output.get("source") == "cache" and previous.get("file_revision") == revision:
        previous["access_seq"] = access_seq
        for span in previous.get("spans", []):
            if ranges_overlap(
                _as_int(output.get("start_line"), 1), _as_int(output.get("end_line"), 1),
                span["start_line"], span["end_line"],
            ):
                span["access_seq"] = access_seq
        cache[file_path] = previous
        return {
            "read_file_cache": cache,
            "read_file_order": touch_file_order(state.get("read_file_order"), file_path),
            "file_cache_access_seq": access_seq,
            "last_read": last_read,
        }
    start = max(1, _as_int(output.get("start_line"), 1))
    end = max(start, _as_int(output.get("end_line"), start))
    content = str(output.get("content") or "")
    if output.get("truncated") and content and not content.endswith(("\n", "\r")):
        complete = content.splitlines(keepends=True)
        content = "".join(line for line in complete if line.endswith(("\n", "\r")))
        end = start + len(content.splitlines()) - 1
    spans = [
        span
        for span in previous.get("spans", [])
        if end < start or not ranges_overlap(start, end, span["start_line"], span["end_line"])
    ]
    if content:
        spans.append(make_span(start, end, content, access_seq=access_seq))
    spans.sort(key=lambda span: (span["start_line"], span["end_line"]))

    snapshot = {
        **previous,
        **dict(extra_fields or {}),
        "file_path": file_path,
        "file_revision": revision,
        "total_lines": max(0, _as_int(output.get("total_lines"), 0)),
        "truncated": bool(output.get("truncated", False)),
        "start_line": output.get("start_line"),
        "end_line": output.get("end_line"),
        "line_range_requested": bool(output.get("line_range_requested", False)),
        "is_empty": _as_int(output.get("total_lines"), 0) == 0 and not content,
        "spans": spans,
        "dirty_ranges": subtract_range(previous.get("dirty_ranges"), start, end) if end >= start else previous.get("dirty_ranges", []),
        "access_seq": access_seq,
        "session_cache_reused": False,
    }
    snapshot["size_bytes"] = snapshot_size_bytes(snapshot)
    cache[file_path] = snapshot
    order = touch_file_order(state.get("read_file_order"), file_path)
    return {
        "read_file_cache": cache,
        "read_file_order": order,
        "file_cache_access_seq": access_seq,
        "last_read": last_read,
    }


def cached_read_coverage(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    """Return reusable coverage only for the current on-disk file revision."""
    path = str(args.get("file_path") or "").strip()
    cache = state.get("read_file_cache")
    snapshot = cache.get(path) if isinstance(cache, dict) else None
    if not isinstance(snapshot, dict):
        return None
    revision = source_revision(state, path)
    if not revision or revision != snapshot.get("file_revision"):
        return None
    total = _as_int(snapshot.get("total_lines"), 0)
    start = _as_int(args.get("start_line"), 1) if args.get("start_line") is not None else 1
    end = _as_int(args.get("end_line"), total) if args.get("end_line") is not None else total
    if start < 1 or end < start or start > total:
        return None
    end = min(end, total)
    covered = merge_ranges(snapshot.get("spans"))
    if not range_is_cached(snapshot, start, end):
        return None
    return {
        "file_path": path,
        "file_revision": revision,
        "requested_range": [start, end],
        "covered_ranges": covered[:8],
        "has_full_content": total > 0 and range_is_cached(snapshot, 1, total),
    }


def source_revision(state: dict[str, Any], path: str) -> str | None:
    """Hash current source in bounded chunks, also when the cache has evicted it."""
    root_value = str(state.get("repo_path") or "").strip()
    if not root_value:
        return None
    try:
        root = Path(root_value).resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            return None
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def range_is_cached(snapshot: dict[str, Any], start: int, end: int) -> bool:
    """One clean span must contain the entire range used by source projection or read reuse."""
    if start < 1 or end < start:
        return False
    if any(
        ranges_overlap(start, end, left, right)
        for left, right in merge_ranges(snapshot.get("dirty_ranges"))
    ):
        return False
    return any(
        isinstance(span, dict)
        and _as_int(span.get("start_line"), 0) <= start
        and _as_int(span.get("end_line"), 0) >= end
        for span in snapshot.get("spans", []) or []
    )


def content_for_range(snapshot: dict[str, Any], start: int, end: int) -> str:
    if not range_is_cached(snapshot, start, end):
        return ""
    for span in reversed(snapshot.get("spans", [])):
        left = int(span.get("start_line") or 1)
        right = int(span.get("end_line") or left)
        if left <= start and right >= end:
            return "".join(str(span.get("content") or "").splitlines(keepends=True)[start - left:end - left + 1])
    return ""


def cached_read_result(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    coverage = cached_read_coverage(state, args)
    if coverage is None:
        return None
    path = coverage["file_path"]
    snapshot = state["read_file_cache"][path]
    start, end = coverage["requested_range"]
    content = content_for_range(snapshot, start, end)
    limit = max(1, int(args.get("max_chars", 8000)))
    truncated = len(content) > limit
    content = content[:limit]
    return {
        "file_path": path, "file_revision": coverage["file_revision"],
        "content": content, "start_line": start,
        "end_line": start + max(0, len(content.splitlines()) - 1),
        "requested_end_line": args.get("end_line") or snapshot["total_lines"],
        "total_lines": snapshot["total_lines"], "truncated": truncated,
        "line_range_requested": args.get("start_line") is not None or args.get("end_line") is not None,
        "source": "cache",
    }


def touch_cache_files(state: dict[str, Any], file_paths: Iterable[str]) -> dict[str, Any]:
    """ 文件从 normal 晋升为 protected"""
    cache = dict(state.get("read_file_cache") or {})
    paths = [str(path).strip() for path in file_paths if str(path).strip() in cache]
    if not paths:
        return {}
    access_seq = _as_int(state.get("file_cache_access_seq"), 0)
    changed = False
    for path in paths:
        snapshot = normalize_snapshot(cache.get(path), file_path=path)
        spans = snapshot.get("spans", [])
        if not spans:
            continue
        access_seq = next_access_seq(cache, access_seq)
        for span in spans:
            span["segment"] = PROTECTED
            span["access_seq"] = access_seq
        snapshot["access_seq"] = access_seq
        snapshot["size_bytes"] = snapshot_size_bytes(snapshot)
        cache[path] = snapshot
        changed = True
    if not changed:
        return {}
    return {
        "read_file_cache": cache,
        "file_cache_access_seq": access_seq,
    }


def cache_after_patch(state: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    if not output.get("applied"):
        return {}
    file_changes = output.get("file_changes")
    if not isinstance(file_changes, list):
        return {}
    cache = dict(state.get("read_file_cache") or {})
    order = list(state.get("read_file_order") or [])
    access_seq = _as_int(state.get("file_cache_access_seq"), 0)
    changed = False
    for raw_change in file_changes:
        if not isinstance(raw_change, dict):
            continue
        path = str(raw_change.get("file_path") or "").strip()
        if not path or path not in cache:
            continue
        snapshot = normalize_snapshot(cache.get(path), file_path=path)
        old_spans = list(snapshot.get("spans", []))
        ranges = _normalize_change_ranges(raw_change.get("ranges"))
        old_revision = str(raw_change.get("old_revision") or "")
        new_revision = str(raw_change.get("new_revision") or "")
        if old_revision and str(snapshot.get("file_revision") or "") != old_revision:
            access_seq = next_access_seq(cache, access_seq)
            snapshot.update(
                {
                    "file_revision": new_revision,
                    "total_lines": max(
                        0, _as_int(raw_change.get("new_total_lines"), 0)
                    ),
                    "spans": [],
                    "dirty_ranges": [
                        [1, max(1, _as_int(raw_change.get("new_total_lines"), 1))]
                    ],
                    "session_cache_reused": False,
                    "access_seq": access_seq,
                }
            )
            snapshot["size_bytes"] = 0
            cache[path] = snapshot
            changed = True
            continue
        if not ranges and old_revision != new_revision:
            ranges = [
                {
                    "old_start_line": 1,
                    "old_end_line": max(
                        1, _as_int(raw_change.get("old_total_lines"), 1)
                    ),
                    "new_start_line": 1,
                    "new_end_line": max(
                        1, _as_int(raw_change.get("new_total_lines"), 1)
                    ),
                    "line_delta": _as_int(raw_change.get("new_total_lines"), 0)
                    - _as_int(raw_change.get("old_total_lines"), 0),
                }
            ]
        access_seq = next_access_seq(cache, access_seq)
        existing_dirty = merge_ranges(snapshot.get("dirty_ranges"))
        closure_regions = old_spans + [
            {"start_line": start, "end_line": end}
            for start, end in existing_dirty
        ]
        dirty_old = overlap_closures(ranges, closure_regions)
        surviving: list[dict[str, Any]] = []
        for span in old_spans:
            if any(
                ranges_overlap(
                    span["start_line"], span["end_line"], dirty[0], dirty[1]
                )
                for dirty in dirty_old
            ):
                continue
            shifted = dict(span)
            delta = sum(
                change["line_delta"]
                for change in ranges
                if change["old_end_line"] < span["start_line"]
            )
            shifted["start_line"] = max(1, span["start_line"] + delta)
            shifted["end_line"] = max(shifted["start_line"], span["end_line"] + delta)
            shifted["segment"] = PROTECTED
            shifted["access_seq"] = access_seq
            surviving.append(shifted)

        dirty_new = [
            _old_envelope_to_new(start, end, ranges) for start, end in dirty_old
        ]
        shifted_existing_dirty = [
            _old_envelope_to_new(start, end, ranges)
            for start, end in existing_dirty
        ]
        snapshot.update(
            {
                "file_revision": new_revision,
                "total_lines": max(0, _as_int(raw_change.get("new_total_lines"), 0)),
                "spans": surviving,
                "dirty_ranges": merge_ranges(
                    shifted_existing_dirty + dirty_new
                ),
                "access_seq": access_seq,
                "session_cache_reused": False,
                "imports_excerpt": "",
                "focus_excerpt": "",
                "focus_ranges": [],
            }
        )
        snapshot["size_bytes"] = snapshot_size_bytes(snapshot)
        cache[path] = snapshot
        changed = True
    if not changed:
        return {}
    return {
        "read_file_cache": cache,
        "read_file_order": order,
        "file_cache_access_seq": access_seq,
    }


def prune_cache(
    cache_value: Any,
    order_value: Any,
    *,
    max_files: int,
    max_spans: int,
    max_bytes: int,
    protected_ratio: float = 0.7,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    cache = {
        str(path): normalize_snapshot(snapshot, file_path=str(path))
        for path, snapshot in (cache_value.items() if isinstance(cache_value, dict) else [])
        if str(path).strip() and isinstance(snapshot, dict)
    }
    order = [str(path) for path in order_value or [] if str(path) in cache]
    for path in cache:
        if path not in order:
            order.append(path)

    span_refs = _span_refs(cache)
    protected = sorted(
        (ref for ref in span_refs if ref[2].get("segment") == PROTECTED),
        key=lambda ref: _as_int(ref[2].get("access_seq"), 0),
    )
    protected_span_limit = max(1, int(max_spans * protected_ratio))
    protected_byte_limit = max(1, int(max_bytes * protected_ratio))
    protected_bytes = sum(_as_int(ref[2].get("size_bytes"), 0) for ref in protected)
    while protected and (
        len(protected) > protected_span_limit or protected_bytes > protected_byte_limit
    ):
        _, _, span = protected.pop(0)
        span["segment"] = NORMAL
        protected_bytes -= _as_int(span.get("size_bytes"), 0)

    def totals() -> tuple[int, int]:
        refs = _span_refs(cache)
        return len(refs), sum(_as_int(ref[2].get("size_bytes"), 0) for ref in refs)

    span_count, total_bytes = totals()
    while span_count > max_spans or total_bytes > max_bytes:
        refs = _span_refs(cache)
        normal = [ref for ref in refs if ref[2].get("segment") == NORMAL]
        candidates = normal or refs
        if not candidates:
            break
        path, index, _ = min(
            candidates, key=lambda ref: _as_int(ref[2].get("access_seq"), 0)
        )
        cache[path]["spans"].pop(index)
        span_count, total_bytes = totals()

    while len(cache) > max_files:
        candidates = [path for path in order if path in cache]
        if not candidates:
            break
        victim = min(
            candidates,
            key=lambda path: _as_int(cache[path].get("access_seq"), 0),
        )
        del cache[victim]
        order.remove(victim)

    for path, snapshot in cache.items():
        snapshot["size_bytes"] = snapshot_size_bytes(snapshot)
    order = [path for path in order if path in cache]
    return cache, order


def snapshot_size_bytes(snapshot: dict[str, Any]) -> int:
    return sum(
        len(str(span.get("content") or "").encode("utf-8"))
        for span in snapshot.get("spans", [])
        if isinstance(span, dict)
    )


def merge_ranges(value: Any) -> list[list[int]]:
    """ 合并 span 的区间"""
    ranges: list[list[int]] = []
    for item in value or []:
        if isinstance(item, dict):
            start = _as_int(item.get("start_line"), 0)
            end = _as_int(item.get("end_line"), start)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = _as_int(item[0], 0)
            end = _as_int(item[1], start)
        else:
            continue
        if start <= 0:
            continue
        ranges.append([start, max(start, end)])
    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def subtract_range(value: Any, start: int, end: int) -> list[list[int]]:
    remaining: list[list[int]] = []
    for dirty_start, dirty_end in merge_ranges(value):
        if not ranges_overlap(start, end, dirty_start, dirty_end):
            remaining.append([dirty_start, dirty_end])
            continue
        if dirty_start < start:
            remaining.append([dirty_start, start - 1])
        if dirty_end > end:
            remaining.append([end + 1, dirty_end])
    return remaining


def overlap_closures(
    changes: list[dict[str, int]], spans: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    envelopes = [
        [change["old_start_line"], change["old_end_line"]] for change in changes
    ]
    expanded = True
    while expanded:
        expanded = False
        for envelope in envelopes:
            for span in spans:
                if not ranges_overlap(
                    envelope[0], envelope[1], span["start_line"], span["end_line"]
                ):
                    continue
                new_start = min(envelope[0], span["start_line"])
                new_end = max(envelope[1], span["end_line"])
                if [new_start, new_end] != envelope:
                    envelope[:] = [new_start, new_end]
                    expanded = True
        merged = merge_ranges(envelopes)
        if merged != envelopes:
            envelopes = merged
            expanded = True
    return [(start, end) for start, end in envelopes]


def touch_file_order(value: Any, file_path: str) -> list[str]:
    order = [str(path).strip() for path in value or [] if str(path).strip()]
    if file_path in order:
        order.remove(file_path)
    order.append(file_path)
    return order


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a <= end_b and end_a >= start_b


def _normalize_spans(value: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        start = max(1, _as_int(item.get("start_line"), 1))
        end = max(start, _as_int(item.get("end_line"), start))
        spans.append(
            make_span(
                start,
                end,
                content,
                access_seq=_as_int(item.get("access_seq"), 0),
                segment=str(item.get("segment") or NORMAL),
            )
        )
    spans.sort(key=lambda span: (span["start_line"], span["end_line"]))
    return spans


def _normalize_change_ranges(value: Any) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        old_start = max(1, _as_int(item.get("old_start_line"), 1))
        old_end = max(old_start, _as_int(item.get("old_end_line"), old_start))
        new_start = max(1, _as_int(item.get("new_start_line"), 1))
        new_end = max(new_start, _as_int(item.get("new_end_line"), new_start))
        ranges.append(
            {
                "old_start_line": old_start,
                "old_end_line": old_end,
                "new_start_line": new_start,
                "new_end_line": new_end,
                "line_delta": _as_int(
                    item.get("line_delta"), (new_end - new_start) - (old_end - old_start)
                ),
            }
        )
    return sorted(ranges, key=lambda item: item["old_start_line"])


def _old_envelope_to_new(
    start: int, end: int, changes: list[dict[str, int]]
) -> list[int]:
    delta_before = sum(
        change["line_delta"]
        for change in changes
        if change["old_end_line"] < start
    )
    delta_through = sum(
        change["line_delta"]
        for change in changes
        if change["old_start_line"] <= end
    )
    new_start = max(1, start + delta_before)
    new_end = max(new_start, end + delta_through)
    for change in changes:
        if ranges_overlap(
            start,
            end,
            change["old_start_line"],
            change["old_end_line"],
        ):
            new_start = min(new_start, change["new_start_line"])
            new_end = max(new_end, change["new_end_line"])
    return [new_start, new_end]


def _span_refs(cache: dict[str, dict[str, Any]]) -> list[tuple[str, int, dict[str, Any]]]:
    refs: list[tuple[str, int, dict[str, Any]]] = []
    for path, snapshot in cache.items():
        refs.extend(
            (path, index, span)
            for index, span in enumerate(snapshot.get("spans", []))
            if isinstance(span, dict)
        )
    return refs


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
