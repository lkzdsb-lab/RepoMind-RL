"""Offline benchmark comparison for policy replay files.

This module compares two fixed task runs, typically heuristic policy vs RL
policy, using replay JSONL files.  It measures harness behavior quality; it
does not claim real bug-fix success unless the replay was collected from a
trusted benchmark task set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_runtime.rl.evaluator import (
    _behavior_metrics,
    _is_verification_pass,
    _load_replay,
    _stale_finish_count,
)

CORE_METRICS = (
    "task_success_rate",
    "avg_steps_to_success",
    "verification_pass_rate",
    "duplicate_read_ratio",
    "stale_finish_count",
)


def summarize_replay(
    replay_path: str | Path,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    transitions = _load_replay(replay_path)
    grouped = _episodes(transitions)
    ordered_task_ids = task_ids or sorted(grouped)

    successful_tasks = 0
    steps_to_success: list[int] = []
    for task_id in ordered_task_ids:
        success_step = _success_step(grouped.get(task_id, []))
        if success_step is not None:
            successful_tasks += 1
            steps_to_success.append(success_step)

    task_count = len(ordered_task_ids)
    behavior = _behavior_metrics(_filter_to_tasks(transitions, ordered_task_ids))
    return {
        "task_count": task_count,
        "successful_tasks": successful_tasks,
        "task_success_rate": round(successful_tasks / task_count, 4) if task_count else 0.0,
        "avg_steps_to_success": round(
            sum(steps_to_success) / len(steps_to_success), 4
        ) if steps_to_success else 0.0,
        "verification_pass_rate": behavior["verification_pass_rate"],
        "duplicate_read_ratio": behavior["duplicate_read_ratio"],
        "stale_finish_count": _stale_finish_count(_filter_to_tasks(transitions, ordered_task_ids)),
        "steps_per_episode_avg": behavior["steps_per_episode_avg"],
        "search_hit_rate": behavior["search_hit_rate"],
        "transition_count": len(_filter_to_tasks(transitions, ordered_task_ids)),
    }


def compare_replays(
    baseline_replay: str | Path,
    candidate_replay: str | Path,
    *,
    tasks_path: str | Path | None = None,
) -> dict[str, Any]:
    task_ids = _load_task_ids(tasks_path) if tasks_path else None
    baseline = summarize_replay(baseline_replay, task_ids=task_ids)
    candidate = summarize_replay(candidate_replay, task_ids=task_ids)
    if task_ids is None:
        task_ids = sorted(
            set(_episodes(_load_replay(baseline_replay)))
            | set(_episodes(_load_replay(candidate_replay)))
        )
        baseline = summarize_replay(baseline_replay, task_ids=task_ids)
        candidate = summarize_replay(candidate_replay, task_ids=task_ids)

    delta = {
        key: round(candidate.get(key, 0.0) - baseline.get(key, 0.0), 4)
        for key in set(baseline) | set(candidate)
        if isinstance(candidate.get(key, 0.0), (int, float))
        and isinstance(baseline.get(key, 0.0), (int, float))
    }
    return {
        "task_ids": task_ids,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
    }


def _episodes(transitions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for transition in transitions:
        grouped[str(transition.get("task_id", "unknown"))].append(transition)
    return dict(grouped)


def _filter_to_tasks(
    transitions: list[dict[str, Any]],
    task_ids: list[str],
) -> list[dict[str, Any]]:
    allowed = set(task_ids)
    return [t for t in transitions if str(t.get("task_id", "unknown")) in allowed]


def _success_step(transitions: list[dict[str, Any]]) -> int | None:
    had_verification_pass = False
    for index, transition in enumerate(transitions, start=1):
        if _is_verification_pass(transition):
            had_verification_pass = True
        if transition.get("action") != "finish":
            continue
        if had_verification_pass and not _finish_is_stale(transition):
            return index
    return None


def _finish_is_stale(transition: dict[str, Any]) -> bool:
    reasons = transition.get("reward_reasons") or []
    return any("stale_verification" in str(reason) for reason in reasons)


def _load_task_ids(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tasks file must be a JSON list")
    task_ids: list[str] = []
    for item in data:
        if isinstance(item, str):
            task_id = item
        elif isinstance(item, dict):
            task_id = str(item.get("task_id") or item.get("id") or "").strip()
        else:
            task_id = ""
        if task_id:
            task_ids.append(task_id)
    return task_ids


def _print_text(result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    candidate = result["candidate"]
    delta = result["delta"]
    print("=" * 72)
    print("  RL Policy Benchmark Comparison")
    print("=" * 72)
    print(f"  Tasks: {len(result['task_ids'])}")
    print("-" * 72)
    print(f"{'Metric':<28s} {'Baseline':>12s} {'Candidate':>12s} {'Delta':>12s}")
    print("-" * 72)
    for key in CORE_METRICS:
        print(
            f"{key:<28s} "
            f"{_format_metric(baseline.get(key, 0.0)):>12s} "
            f"{_format_metric(candidate.get(key, 0.0)):>12s} "
            f"{_format_metric(delta.get(key, 0.0), signed=True):>12s}"
        )
    print("-" * 72)
    print("Note: lower is better for avg_steps_to_success, duplicate_read_ratio, and stale_finish_count.")
    print("=" * 72)


def _format_metric(value: Any, *, signed: bool = False) -> str:
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    if isinstance(value, float):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare heuristic and RL policy replay files on fixed benchmark tasks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="Compare baseline and candidate replay files.")
    compare.add_argument("--baseline-replay", required=True, help="Baseline replay JSONL.")
    compare.add_argument("--candidate-replay", required=True, help="Candidate replay JSONL.")
    compare.add_argument("--tasks", help="Optional JSON task list to fix benchmark task IDs.")
    compare.add_argument("--format", choices=("text", "json"), default="text")

    args = parser.parse_args(argv)
    if args.command == "compare":
        result = compare_replays(
            args.baseline_replay,
            args.candidate_replay,
            tasks_path=args.tasks,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_text(result)


if __name__ == "__main__":
    main()
