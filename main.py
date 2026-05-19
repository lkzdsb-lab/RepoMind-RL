from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_runtime.executor import DebugAgent, DebugAgentConfig
from agent_runtime.logging_config import configure_logging
from config import LLMConfig
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RepoMind-RL first-version debug agent harness."
    )
    parser.add_argument("issue", help="Bug/issue title or short description.")
    parser.add_argument("--description", default="", help="Detailed issue description.")
    parser.add_argument("--repo", default=".", help="Target repository path.")
    parser.add_argument("--verify", default="pytest", help="Verification command.")
    parser.add_argument("--max-loops", type=int, default=8, help="Maximum agent action loops.")
    parser.add_argument("--manifest-dir", default=None, help="Runtime registry manifest directory.")
    parser.add_argument("--memory-redis-url", default=None, help="Optional Redis URL for mid-term memory.")
    parser.add_argument(
        "--disable-context-compression",
        action="store_true",
        help="Disable runtime context compression.",
    )
    parser.add_argument("--context-max-tokens", type=int, default=32000, help="Context token budget.")
    parser.add_argument(
        "--context-compression-threshold",
        type=float,
        default=0.75,
        help="Compression threshold ratio.",
    )
    parser.add_argument("--llm-provider", default="disabled", help="Default LLM provider.")
    parser.add_argument("--llm-model", default="", help="Default LLM model.")
    parser.add_argument("--llm-api-base", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument(
        "--llm-api-key-env",
        default="LLM_API_KEY",
        help="Environment variable containing the LLM API key.",
    )
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM timeout in seconds.")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument(
        "--llm-max-output-chars",
        type=int,
        default=12000,
        help="Maximum characters kept from an LLM response.",
    )
    parser.add_argument(
        "--planner-mode",
        default="heuristic",
        choices=["heuristic", "llm"],
        help="Planner implementation.",
    )
    parser.add_argument(
        "--action-policy-mode",
        default="heuristic",
        choices=["heuristic", "rl", "llm"],
        help="Action selection policy.",
    )
    parser.add_argument(
        "--task-analyzer-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Task analyzer implementation.",
    )
    parser.add_argument(
        "--observer-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Observation synthesis implementation.",
    )
    parser.add_argument("--plan-llm-provider", default="", help="Planner-specific LLM provider.")
    parser.add_argument("--plan-llm-model", default="", help="Planner-specific LLM model.")
    parser.add_argument("--plan-llm-api-base", default="", help="Planner-specific LLM API base.")
    parser.add_argument(
        "--plan-llm-api-key-env",
        default="",
        help="Planner-specific API key environment variable.",
    )
    parser.add_argument("--action-llm-provider", default="", help="Action-policy-specific LLM provider.")
    parser.add_argument("--action-llm-model", default="", help="Action-policy-specific LLM model.")
    parser.add_argument("--action-llm-api-base", default="", help="Action-policy-specific LLM API base.")
    parser.add_argument(
        "--action-llm-api-key-env",
        default="",
        help="Action-policy-specific API key environment variable.",
    )
    parser.add_argument(
        "--task-analysis-llm-provider",
        default="",
        help="Task-analysis-specific LLM provider.",
    )
    parser.add_argument("--task-analysis-llm-model", default="", help="Task-analysis-specific LLM model.")
    parser.add_argument(
        "--task-analysis-llm-api-base",
        default="",
        help="Task-analysis-specific LLM API base.",
    )
    parser.add_argument(
        "--task-analysis-llm-api-key-env",
        default="",
        help="Task-analysis-specific API key environment variable.",
    )
    parser.add_argument("--observer-llm-provider", default="", help="Observer-specific LLM provider.")
    parser.add_argument("--observer-llm-model", default="", help="Observer-specific LLM model.")
    parser.add_argument("--observer-llm-api-base", default="", help="Observer-specific LLM API base.")
    parser.add_argument(
        "--observer-llm-api-key-env",
        default="",
        help="Observer-specific API key environment variable.",
    )
    parser.add_argument(
        "--code-context-index-path",
        default=".repomind/codebase_context/index.json",
        help="Path inside the target repo for the codebase context index.",
    )
    parser.add_argument("--rl-enabled", action="store_true", help="Enable Q-learning policy.")
    parser.add_argument("--rl-q-table-path", default=".repomind/rl/q_table.json", help="RL Q-table path.")
    parser.add_argument("--rl-replay-path", default=".repomind/rl/replay.jsonl", help="RL replay path.")
    parser.add_argument("--rl-epsilon", type=float, default=0.15, help="RL exploration rate.")
    parser.add_argument("--rl-learning-rate", type=float, default=0.2, help="RL learning rate.")
    parser.add_argument("--rl-discount", type=float, default=0.9, help="RL discount factor.")
    parser.add_argument("--log-level", default="INFO", help="Runtime log level.")
    parser.add_argument(
        "--log-file",
        default=".repomind/logs/agent.log",
        help="Path inside the target repo for runtime logs.",
    )
    parser.add_argument("--log-json", action="store_true", help="Write JSON logs.")
    parser.add_argument("--no-console-log", action="store_true", help="Disable stderr logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo).resolve()

    base_llm_config = LLMConfig(
        provider=args.llm_provider,
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        temperature=args.llm_temperature,
        max_output_chars=args.llm_max_output_chars,
    )
    config = DebugAgentConfig(
        repo_path=repo_path.as_posix(),
        verify_command=args.verify,
        max_loops=args.max_loops,
        manifest_dir=args.manifest_dir,
        memory_redis_url=args.memory_redis_url,
        context_compression_enabled=not args.disable_context_compression,
        context_max_tokens=args.context_max_tokens,
        context_compression_threshold=args.context_compression_threshold,
        llm_config=base_llm_config,
        plan_llm_config=_merge_llm_config(base_llm_config, _specific_llm_config(args, "plan")),
        action_llm_config=_merge_llm_config(base_llm_config, _specific_llm_config(args, "action")),
        task_analysis_llm_config=_merge_llm_config(
            base_llm_config,
            _specific_llm_config(args, "task_analysis"),
        ),
        observer_llm_config=_merge_llm_config(
            base_llm_config,
            _specific_llm_config(args, "observer"),
        ),
        planner_mode=args.planner_mode,
        action_policy_mode=args.action_policy_mode,
        task_analyzer_mode=args.task_analyzer_mode,
        observer_mode=args.observer_mode,
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
        "starting agent cli repo_path={} issue={} planner_mode={} action_policy_mode={} task_analyzer_mode={} observer_mode={} rl_enabled={}",
        repo_path.as_posix(),
        args.issue,
        args.planner_mode,
        args.action_policy_mode,
        args.task_analyzer_mode,
        args.observer_mode,
        args.rl_enabled,
    )
    result = DebugAgent(config).run(title=args.issue, description=args.description)

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
        "planner_mode": config.planner_mode,
        "action_policy_mode": config.action_policy_mode,
        "task_analyzer_mode": config.task_analyzer_mode,
        "observer_mode": config.observer_mode,
        "task_analysis": state.get("task_analysis", {}),
        "llm_observations": len(state.get("llm_observations", [])),
        "rl_enabled": bool(state.get("rl_enabled")),
        "rl_transitions": len(state.get("rl_transitions", [])),
        "rl_last_reward": state.get("rl_last_reward", {}),
        "promoted_memories": len(state.get("promoted_memories", [])),
        "consolidated_skills": len(state.get("consolidated_skills", [])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _specific_llm_config(args: argparse.Namespace, prefix: str) -> LLMConfig:
    return LLMConfig(
        provider=getattr(args, f"{prefix}_llm_provider"),
        model=getattr(args, f"{prefix}_llm_model"),
        api_base=getattr(args, f"{prefix}_llm_api_base"),
        api_key_env=getattr(args, f"{prefix}_llm_api_key_env"),
        timeout=args.llm_timeout,
        temperature=args.llm_temperature,
        max_output_chars=args.llm_max_output_chars,
    )


def _merge_llm_config(base: LLMConfig, override: LLMConfig) -> LLMConfig:
    return LLMConfig(
        provider=override.provider or base.provider,
        model=override.model or base.model,
        api_base=override.api_base or base.api_base,
        api_key_env=override.api_key_env or base.api_key_env,
        timeout=override.timeout or base.timeout,
        temperature=override.temperature if override.temperature != 0.0 else base.temperature,
        max_output_chars=override.max_output_chars or base.max_output_chars,
    )


if __name__ == "__main__":
    main()
