from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_runtime.executor import DebugAgent, DebugAgentConfig
from agent_runtime.logging_config import configure_logging
from loguru import logger


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
    parser.add_argument(
        "--memory-redis-url",
        default=None,
        help="Optional Redis URL for mid-term memory.",
    )
    parser.add_argument(
        "--disable-context-compression",
        action="store_true",
        help="Disable runtime context compression.",
    )
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=32000,
        help="Estimated token budget used before context compression.",
    )
    parser.add_argument(
        "--context-compression-threshold",
        type=float,
        default=0.75,
        help="Compress when estimated context tokens exceed this budget ratio.",
    )
    parser.add_argument(
        "--context-llm-provider",
        default="disabled",
        help="LLM provider for context compression, for example openai_compatible.",
    )
    parser.add_argument(
        "--context-llm-model",
        default="",
        help="Model name for LLM context compression.",
    )
    parser.add_argument(
        "--context-llm-api-base",
        default="",
        help="Chat completions API base URL for LLM context compression.",
    )
    parser.add_argument(
        "--context-llm-api-key-env",
        default="LLM_API_KEY",
        help="Environment variable containing the LLM API key.",
    )
    parser.add_argument(
        "--code-context-index-path",
        default=".repomind/codebase_context/index.json",
        help="Path inside the target repo for the codebase context index.",
    )
    parser.add_argument(
        "--rl-enabled",
        action="store_true",
        help="Use the epsilon-greedy Q-learning policy and record replay transitions.",
    )
    parser.add_argument(
        "--rl-q-table-path",
        default=".repomind/rl/q_table.json",
        help="Path inside the target repo for the RL Q-table.",
    )
    parser.add_argument(
        "--rl-replay-path",
        default=".repomind/rl/replay.jsonl",
        help="Path inside the target repo for the RL replay buffer.",
    )
    parser.add_argument(
        "--rl-epsilon",
        type=float,
        default=0.15,
        help="Exploration rate for the RL policy.",
    )
    parser.add_argument(
        "--rl-learning-rate",
        type=float,
        default=0.2,
        help="Q-learning update rate.",
    )
    parser.add_argument(
        "--rl-discount",
        type=float,
        default=0.9,
        help="Q-learning discount factor.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level: TRACE, DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )
    parser.add_argument(
        "--log-file",
        default=".repomind/logs/agent.log",
        help="Path inside the target repo for runtime logs. Empty disables file logs.",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Write logs as JSON lines.",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable stderr logging.",
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
        memory_redis_url=args.memory_redis_url,
        context_compression_enabled=not args.disable_context_compression,
        context_max_tokens=args.context_max_tokens,
        context_compression_threshold=args.context_compression_threshold,
        context_llm_provider=args.context_llm_provider,
        context_llm_model=args.context_llm_model,
        context_llm_api_base=args.context_llm_api_base,
        context_llm_api_key_env=args.context_llm_api_key_env,
        code_context_index_path=args.code_context_index_path,
        rl_enabled=args.rl_enabled,
        rl_q_table_path=args.rl_q_table_path,
        rl_replay_path=args.rl_replay_path,
        rl_epsilon=args.rl_epsilon,
        rl_learning_rate=args.rl_learning_rate,
        rl_discount=args.rl_discount,
        log_level=args.log_level,
        log_file=args.log_file,
        log_json=args.log_json,
        log_to_console=not args.no_console_log,
    )
    log_file = args.log_file or None
    if log_file:
        path = Path(log_file)
        if not path.is_absolute():
            path = repo_path / path
        log_file = path.as_posix()
    configure_logging(
        level=args.log_level,
        log_file=log_file,
        json_logs=args.log_json,
        console=not args.no_console_log,
        force=True,
    )
    logger.info(
        "starting agent cli repo_path={} issue={} rl_enabled={} manifest_dir={}",
        repo_path.as_posix(),
        args.issue,
        args.rl_enabled,
        args.manifest_dir,
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
        "memory_context_present": bool(state.get("memory_context")),
        "compressed_context_present": bool(state.get("compressed_context")),
        "context_compression_method": state.get("context_digest", {}).get("compression_method"),
        "code_context_present": bool(state.get("code_context")),
        "rl_enabled": bool(state.get("rl_enabled")),
        "rl_transitions": len(state.get("rl_transitions", [])),
        "rl_last_reward": state.get("rl_last_reward", {}),
        "promoted_memories": len(state.get("promoted_memories", [])),
        "consolidated_skills": len(state.get("consolidated_skills", [])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
