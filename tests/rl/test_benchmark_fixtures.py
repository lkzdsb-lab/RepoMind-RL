"""Regression tests for committed benchmark smoke replay fixtures."""

from __future__ import annotations

from pathlib import Path

import agent_runtime.rl.benchmark as benchmark


FIXTURE_DIR = Path("benchmarks/rl_policy_smoke")


def test_smoke_fixture_replays_show_candidate_policy_improvement():
    comparison = benchmark.compare_replays(
        FIXTURE_DIR / "heuristic_replay.jsonl",
        FIXTURE_DIR / "rl_replay.jsonl",
        tasks_path=FIXTURE_DIR / "tasks.json",
    )

    assert comparison["baseline"]["task_success_rate"] == 0.6667
    assert comparison["candidate"]["task_success_rate"] == 1.0
    assert comparison["delta"]["task_success_rate"] == 0.3333
    assert comparison["candidate"]["avg_steps_to_success"] < comparison["baseline"]["avg_steps_to_success"]
    assert comparison["candidate"]["duplicate_read_ratio"] < comparison["baseline"]["duplicate_read_ratio"]
    assert comparison["candidate"]["stale_finish_count"] < comparison["baseline"]["stale_finish_count"]
