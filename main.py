from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_runtime.executor import DebugAgent, DebugAgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RepoMind-RL first-version debug agent harness."
    )
    parser.add_argument("issue", help="Bug/issue title or short description.")
    parser.add_argument(
        "--description",
        default="",
        help="Detailed issue description.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Target repository path. Defaults to current directory.",
    )
    parser.add_argument(
        "--verify",
        default="pytest",
        help="Verification command, for example `pytest` or `go test ./...`.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=8,
        help="Maximum agent action loops.",
    )
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Optional directory containing runtime registry manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo).resolve()
    config = DebugAgentConfig(
        repo_path=repo_path.as_posix(),
        verify_command=args.verify,
        max_loops=args.max_loops,
        manifest_dir=args.manifest_dir,
    )
    result = DebugAgent(config).run(
        title=args.issue,
        description=args.description,
    )

    state = result.state
    summary = {
        "task_id": state.get("task_id"),
        "status": state.get("status"),
        "candidate_files": state.get("candidate_files", []),
        "patch_summary": state.get("patch_summary"),
        "trace_path": result.trace_path,
        "tool_calls": len(state.get("tool_calls", [])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
