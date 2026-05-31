"""Q-table inspection and legacy-wrapping tools.

Usage::

    # Inspect a Q-table file
    python -m agent_runtime.rl.qtable_tools inspect --q-table .repomind/rl/q_table.json

    # Wrap a legacy Q-table into the envelope format
    python -m agent_runtime.rl.qtable_tools wrap-legacy \\
        --input old.json --output wrapped.json --trust-legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent_runtime.rl.action_space import ACTION_SPACE_VERSION
from agent_runtime.rl.reward import REWARD_VERSION
from agent_runtime.rl.state_encoder import ENCODER_VERSION

_CURRENT_VERSIONS = {
    "encoder_version": ENCODER_VERSION,
    "action_space_version": ACTION_SPACE_VERSION,
    "reward_version": REWARD_VERSION,
}


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
def _inspect(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"[inspect] file not found: {p}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[inspect] unreadable JSON: {p} error={exc}", file=sys.stderr)
        sys.exit(1)

    is_envelope = isinstance(data, dict) and "metadata" in data

    if is_envelope:
        metadata = data.get("metadata", {})
        q_values = data.get("q_values", {})
    else:
        metadata = {}
        q_values = data if isinstance(data, dict) else {}

    state_count = len(q_values)
    action_entries = sum(len(actions) for actions in q_values.values())
    all_q = [
        float(v)
        for actions in q_values.values()
        for v in actions.values()
    ]

    result = {
        "path": str(p),
        "is_envelope": is_envelope,
        "state_count": state_count,
        "action_entries": action_entries,
        "metadata": metadata,
        "q_max": round(max(all_q), 4) if all_q else 0.0,
        "q_min": round(min(all_q), 4) if all_q else 0.0,
    }
    return result


def inspect_command(args: argparse.Namespace) -> None:
    result = _inspect(args.q_table)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Path           : {result['path']}")
        print(f"Envelope       : {result['is_envelope']}")
        print(f"States         : {result['state_count']}")
        print(f"Action entries : {result['action_entries']}")
        print(f"Q max          : {result['q_max']:+.4f}")
        print(f"Q min          : {result['q_min']:+.4f}")
        print("Metadata:")
        meta = result["metadata"]
        if meta:
            for k, v in meta.items():
                print(f"  {k}: {v}")
        else:
            print("  (none / legacy)")


# ---------------------------------------------------------------------------
# wrap-legacy
# ---------------------------------------------------------------------------
def wrap_legacy_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[wrap-legacy] input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[wrap-legacy] unreadable JSON: {input_path} error={exc}", file=sys.stderr)
        sys.exit(1)

    # Already envelope? Refuse unless --force is set.
    if isinstance(data, dict) and "metadata" in data:
        if not args.force:
            print(
                "[wrap-legacy] input already has an envelope (contains 'metadata' key). "
                "Use --force to overwrite the metadata.",
                file=sys.stderr,
            )
            sys.exit(1)
        q_values = data.get("q_values", {})
    elif isinstance(data, dict):
        q_values = data
    else:
        print("[wrap-legacy] input must be a JSON object.", file=sys.stderr)
        sys.exit(1)

    # --trust-legacy gate
    if not args.trust_legacy:
        print(
            "[wrap-legacy] refusing to wrap legacy Q-table without --trust-legacy. "
            "Legacy Q-tables are unversioned and may not be compatible with the "
            "current RL policy.  Re-run with --trust-legacy if you have manually "
            "verified that this data is safe to use.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "metadata": {
            **_CURRENT_VERSIONS,
            "migrated_from": "legacy/unversioned",
            "migration_mode": "wrap_only_no_semantic_conversion",
        },
        "q_values": q_values,
    }
    output_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[wrap-legacy] wrapped {len(q_values)} states into envelope "
        f"→ {output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Q-table inspection and legacy-wrapping tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect a Q-table file.")
    p_inspect.add_argument("--q-table", required=True, help="Path to Q-table JSON.")
    p_inspect.add_argument("--json", action="store_true", help="Output as JSON.")

    # wrap-legacy
    p_wrap = sub.add_parser("wrap-legacy", help="Wrap a legacy Q-table into an envelope.")
    p_wrap.add_argument("--input", required=True, help="Path to legacy Q-table JSON.")
    p_wrap.add_argument("--output", required=True, help="Path for the wrapped output.")
    p_wrap.add_argument(
        "--trust-legacy",
        action="store_true",
        help="Acknowledge that the legacy data is safe to use.",
    )
    p_wrap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite metadata if input already has an envelope.",
    )

    args = parser.parse_args(argv)

    if args.command == "inspect":
        inspect_command(args)
    elif args.command == "wrap-legacy":
        wrap_legacy_command(args)


if __name__ == "__main__":
    main()
