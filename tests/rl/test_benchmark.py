"""Tests for offline policy benchmark comparison."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import sys

import agent_runtime.rl.benchmark as benchmark


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _transition(
    task_id: str,
    action: str,
    *,
    reward: float = 0.0,
    file_path: str = "",
    exit_code: int | None = None,
    candidate_count: int | None = None,
    stale_finish: bool = False,
) -> dict:
    action_args = {}
    summary = {}
    if file_path:
        action_args["file_path"] = file_path
    if exit_code is not None:
        summary["exit_code"] = exit_code
    if candidate_count is not None:
        summary["candidate_count"] = candidate_count
    reward_reasons = ["finish_with_stale_verification=-1.2"] if stale_finish else []
    return {
        "task_id": task_id,
        "action": action,
        "reward": reward,
        "action_args": action_args,
        "tool_output_summary": summary,
        "reward_reasons": reward_reasons,
    }


def test_summarize_replay_counts_success_steps_and_behavior_metrics(tmp_path):
    replay = _write_jsonl(
        tmp_path / "rl.jsonl",
        [
            _transition("task-a", "search_code_context", candidate_count=2),
            _transition("task-a", "read_file", file_path="a.py"),
            _transition("task-a", "run_tests", exit_code=0),
            _transition("task-a", "finish"),
            _transition("task-b", "search_code_context", candidate_count=1),
            _transition("task-b", "read_file", file_path="b.py"),
            _transition("task-b", "read_file", file_path="b.py"),
            _transition("task-b", "run_tests", exit_code=1),
            _transition("task-b", "finish", stale_finish=True),
        ],
    )

    result = benchmark.summarize_replay(str(replay), task_ids=["task-a", "task-b"])

    assert result["task_count"] == 2
    assert result["successful_tasks"] == 1
    assert result["task_success_rate"] == 0.5
    assert result["avg_steps_to_success"] == 4.0
    assert result["stale_finish_count"] == 1
    assert result["search_hit_rate"] == 1.0
    assert result["duplicate_read_ratio"] == 0.3333
    assert result["verification_pass_rate"] == 0.5


def test_compare_reports_candidate_vs_baseline_deltas(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "heuristic.jsonl",
        [
            _transition("task-a", "search_code_context", candidate_count=1),
            _transition("task-a", "read_file", file_path="a.py"),
            _transition("task-a", "read_file", file_path="a.py"),
            _transition("task-a", "run_tests", exit_code=0),
            _transition("task-a", "finish"),
            _transition("task-b", "search_code_context", candidate_count=0),
            _transition("task-b", "finish", stale_finish=True),
        ],
    )
    candidate = _write_jsonl(
        tmp_path / "rl.jsonl",
        [
            _transition("task-a", "search_code_context", candidate_count=1),
            _transition("task-a", "read_file", file_path="a.py"),
            _transition("task-a", "run_tests", exit_code=0),
            _transition("task-a", "finish"),
            _transition("task-b", "search_code_context", candidate_count=1),
            _transition("task-b", "read_file", file_path="b.py"),
            _transition("task-b", "run_tests", exit_code=0),
            _transition("task-b", "finish"),
        ],
    )
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"task_id": "task-a"}, {"task_id": "task-b"}]),
        encoding="utf-8",
    )

    comparison = benchmark.compare_replays(str(baseline), str(candidate), tasks_path=str(tasks))

    assert comparison["baseline"]["task_success_rate"] == 0.5
    assert comparison["candidate"]["task_success_rate"] == 1.0
    assert comparison["delta"]["task_success_rate"] == 0.5
    assert comparison["delta"]["stale_finish_count"] == -1
    assert comparison["delta"]["duplicate_read_ratio"] < 0
    assert comparison["task_ids"] == ["task-a", "task-b"]


def test_cli_json_output_contains_core_metrics(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "heuristic.jsonl",
        [
            _transition("task-a", "run_tests", exit_code=0),
            _transition("task-a", "finish"),
        ],
    )
    candidate = _write_jsonl(
        tmp_path / "rl.jsonl",
        [
            _transition("task-a", "run_tests", exit_code=0),
            _transition("task-a", "finish"),
        ],
    )

    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = captured
    try:
        benchmark.main(
            [
                "compare",
                "--baseline-replay",
                str(baseline),
                "--candidate-replay",
                str(candidate),
                "--format",
                "json",
            ]
        )
    finally:
        sys.stdout = old_stdout

    data = json.loads(captured.getvalue())
    assert data["baseline"]["task_success_rate"] == 1.0
    assert data["candidate"]["verification_pass_rate"] == 1.0
