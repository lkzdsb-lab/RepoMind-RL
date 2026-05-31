"""Offline evaluator for RL replay data and Q-table.

Usage::

    python -m agent_runtime.rl.evaluator \\
        --replay .repomind/rl/replay.jsonl \\
        --q-table .repomind/rl/q_table.json \\
        --format text
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_runtime.rl.action_space import ACTION_SPACE_VERSION
from agent_runtime.rl.reward import REWARD_VERSION
from agent_runtime.rl.state_encoder import ENCODER_VERSION

_EXPECTED_VERSIONS = {
    "encoder_version": ENCODER_VERSION,
    "action_space_version": ACTION_SPACE_VERSION,
    "reward_version": REWARD_VERSION,
}


def _load_replay(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"[evaluator] replay file not found: {p}", file=sys.stderr)
        return []
    transitions: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                transitions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return transitions


def _load_q_table(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"[evaluator] q-table file not found: {p}", file=sys.stderr)
        return {"metadata": {}, "q_values": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[evaluator] q-table unreadable: {p} error={exc}", file=sys.stderr)
        return {"metadata": {}, "q_values": {}}

    if isinstance(data, dict) and "metadata" in data:
        return data
    # Legacy format — wrap so caller can still inspect
    return {"metadata": {}, "q_values": data}


def _episodes(transitions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in transitions:
        by_task[t.get("task_id", "unknown")].append(t)
    return dict(by_task)


def _is_verification_pass(transition: dict[str, Any]) -> bool:
    """Return True if *transition* represents a successful verification."""
    action = transition.get("action", "")
    if action == "run_tests" and transition.get("reward", 0) > 0.5:
        return True
    # Also recognise run_shell_command with purpose=verification and exit_code=0
    if action == "run_shell_command":
        action_args = transition.get("action_args", {})
        if isinstance(action_args, dict) and action_args.get("purpose") == "verification":
            if _transition_exit_code(transition) == 0:
                return True
            # Older replay records did not persist tool output. Keep a reward
            # fallback so historical verification runs remain inspectable.
            if transition.get("reward", 0) > 0.5:
                return True
    return False


def _transition_exit_code(transition: dict[str, Any]) -> int | None:
    for key in ("tool_output", "output", "tool_result", "action_args"):
        value = transition.get(key)
        if isinstance(value, dict) and "exit_code" in value:
            try:
                return int(value["exit_code"])
            except (TypeError, ValueError):
                return None
    return None


def _finish_after_tests_ratio(transitions: list[dict[str, Any]]) -> float:
    """Fraction of finish transitions that follow a successful verification within
    the same episode.  Recognises both run_tests pass and run_shell_command
    (verification) pass."""
    by_task = _episodes(transitions)
    finish_count = 0
    finish_after = 0
    for _tid, items in by_task.items():
        had_verification_pass = False
        for t in items:
            if _is_verification_pass(t):
                had_verification_pass = True
            if t.get("action") == "finish":
                finish_count += 1
                if had_verification_pass:
                    finish_after += 1
    if finish_count == 0:
        return 0.0
    return finish_after / finish_count


def _stale_finish_count(transitions: list[dict[str, Any]]) -> int:
    count = 0
    for t in transitions:
        if t.get("action") != "finish":
            continue
        reasons = t.get("reward_reasons", [])
        if any("stale_verification" in r for r in reasons):
            count += 1
    return count


def _replay_version_coverage(
    transitions: list[dict[str, Any]],
) -> dict[str, float]:
    """For each version field, what fraction of transitions have a non-empty
    value matching the expected version."""
    expected = {
        "encoder_version": ENCODER_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "reward_version": REWARD_VERSION,
    }
    coverage: dict[str, float] = {}
    total = len(transitions)
    if total == 0:
        return {"encoder_version": 0.0, "action_space_version": 0.0, "reward_version": 0.0}
    for field, expected_value in expected.items():
        matched = sum(
            1 for t in transitions if t.get(field) == expected_value
        )
        coverage[field] = round(matched / total, 4)
    return coverage


def _top_q_values(
    q_values: dict[str, dict[str, float]], top_n: int = 10
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for state, actions in q_values.items():
        for action, value in actions.items():
            entries.append({"state": state, "action": action, "value": value})
    entries.sort(key=lambda e: e["value"], reverse=True)
    return entries[:top_n]


def run(replay_path: str, q_table_path: str, fmt: str) -> None:
    transitions = _load_replay(replay_path)
    q_data = _load_q_table(q_table_path)
    q_values: dict[str, dict[str, float]] = q_data.get("q_values", {})
    metadata: dict[str, str] = q_data.get("metadata", {})

    episodes = _episodes(transitions)
    rewards = [t.get("reward", 0.0) for t in transitions]
    actions = [t.get("action", "unknown") for t in transitions]
    action_counter = Counter(actions)
    action_rewards: dict[str, list[float]] = defaultdict(list)
    for t in transitions:
        action_rewards[t.get("action", "unknown")].append(t.get("reward", 0.0))

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    stale_finishes = _stale_finish_count(transitions)
    finish_ratio = _finish_after_tests_ratio(transitions)
    replay_cov = _replay_version_coverage(transitions)

    per_action_avg: dict[str, float] = {}
    for action, rs in sorted(action_rewards.items()):
        per_action_avg[action] = sum(rs) / len(rs) if rs else 0.0

    total_q_states = len(q_values)
    all_q_actions: list[float] = []
    for actions_dict in q_values.values():
        all_q_actions.extend(actions_dict.values())
    total_q_entries = len(all_q_actions)
    max_q = max(all_q_actions) if all_q_actions else 0.0
    min_q = min(all_q_actions) if all_q_actions else 0.0
    top_q = _top_q_values(q_values, top_n=10)

    metadata_matches = all(
        metadata.get(k) == v for k, v in _EXPECTED_VERSIONS.items()
    )

    if fmt == "json":
        output: dict[str, Any] = {
            "transitions": len(transitions),
            "episodes": len(episodes),
            "avg_reward": round(avg_reward, 4),
            "action_distribution": dict(action_counter.most_common()),
            "per_action_avg_reward": {k: round(v, 4) for k, v in per_action_avg.items()},
            "stale_finish_count": stale_finishes,
            "finish_after_tests_ratio": round(finish_ratio, 4),
            "expected_versions": dict(_EXPECTED_VERSIONS),
            "q_table_metadata": metadata,
            "metadata_matches_expected": metadata_matches,
            "replay_version_coverage": replay_cov,
            "q_table_states": total_q_states,
            "q_table_entries": total_q_entries,
            "q_table_max": round(max_q, 4),
            "q_table_min": round(min_q, 4),
            "top_q_values": top_q,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  RL Evaluation Report")
        lines.append("=" * 60)
        lines.append(f"  Transitions            : {len(transitions)}")
        lines.append(f"  Episodes (tasks)       : {len(episodes)}")
        lines.append(f"  Avg reward             : {avg_reward:+.4f}")
        lines.append("-" * 60)
        lines.append("  Action distribution:")
        for action, count in action_counter.most_common():
            pct = count / len(transitions) * 100 if transitions else 0.0
            avg_r = per_action_avg.get(action, 0.0)
            lines.append(
                f"    {action:<30s} {count:>5d} ({pct:5.1f}%)  avg={avg_r:+.4f}"
            )
        lines.append("-" * 60)
        lines.append(f"  Stale finish count     : {stale_finishes}")
        lines.append(f"  Finish-after-tests ratio: {finish_ratio:.2%}")
        lines.append("-" * 60)
        lines.append("  Version info:")
        lines.append("    Expected versions:")
        for k, v in _EXPECTED_VERSIONS.items():
            lines.append(f"      {k}: {v}")
        lines.append("    Q-table metadata:")
        if metadata:
            for k, v in metadata.items():
                lines.append(f"      {k}: {v}")
        else:
            lines.append("      (none / legacy)")
        lines.append(f"    Metadata matches expected: {metadata_matches}")
        lines.append("    Replay version coverage:")
        for k, v in replay_cov.items():
            lines.append(f"      {k}: {v:.1%}")
        lines.append("-" * 60)
        lines.append("  Q-table:")
        lines.append(f"    states               : {total_q_states}")
        lines.append(f"    entries              : {total_q_entries}")
        lines.append(f"    max Q                : {max_q:+.4f}")
        lines.append(f"    min Q                : {min_q:+.4f}")
        if top_q:
            lines.append("    Top Q values:")
            for entry in top_q[:10]:
                lines.append(
                    f"      {entry['state'][:40]:<40s} "
                    f"{entry['action']:<25s} {entry['value']:+.4f}"
                )
        lines.append("=" * 60)
        print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline evaluator for RL replay and Q-table data."
    )
    parser.add_argument(
        "--replay",
        default=".repomind/rl/replay.jsonl",
        help="Path to replay JSONL file (default: .repomind/rl/replay.jsonl)",
    )
    parser.add_argument(
        "--q-table",
        default=".repomind/rl/q_table.json",
        help="Path to Q-table JSON file (default: .repomind/rl/q_table.json)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args(argv)
    run(args.replay, args.q_table, args.format)


if __name__ == "__main__":
    main()
