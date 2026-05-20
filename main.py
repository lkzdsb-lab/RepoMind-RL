from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_runtime.executor import DebugAgent
from agent_runtime.logging_config import configure_logging
from config import (
    DEFAULT_CONFIG_PATH,
    DebugAgentConfig,
    LLMConfig,
    debug_agent_config_from_dict,
    ensure_default_config_file,
    load_config_payload,
    load_env_file,
    normalize_project_runtime_paths,
    validate_debug_agent_config,
)
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RepoMind-RL first-version debug agent harness."
    )
    parser.add_argument("issue", nargs="?", help="Bug/issue title or short description.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="JSON config file. The default config.json is loaded automatically when present.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Do not load config.json.",
    )
    parser.add_argument("--description", default="", help="Detailed issue description.")
    parser.add_argument("--repo", default=".", help="Target repository path.")
    parser.add_argument("--verify", default="pytest", help="Verification command.")
    parser.add_argument("--max-loops", type=int, default=8, help="Maximum agent action loops.")
    parser.add_argument("--env-file", default=".env", help="Local env file containing secrets.")
    parser.add_argument(
        "--env-override",
        action="store_true",
        help="Let env-file values override existing process environment variables.",
    )
    parser.add_argument("--manifest-dir", default=None, help="Runtime registry manifest directory.")
    parser.add_argument("--memory-redis-url", default=None, help="Optional Redis URL for mid-term memory.")
    parser.add_argument(
        "--disable-context-compression",
        action="store_true",
        help="Disable runtime context compression.",
    )
    parser.add_argument(
        "--context-compressor-mode",
        default="rule_based",
        choices=["disabled", "rule_based", "llm"],
        help="Context compressor implementation.",
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
    parser.add_argument(
        "--memory-query-planner-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Memory query planner implementation.",
    )
    parser.add_argument(
        "--memory-reranker-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Memory reranker implementation.",
    )
    parser.add_argument(
        "--code-context-query-planner-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Codebase context query planner implementation.",
    )
    parser.add_argument(
        "--code-context-reranker-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Codebase context reranker implementation.",
    )
    parser.add_argument(
        "--skill-selector-mode",
        default="disabled",
        choices=["disabled", "llm"],
        help="Skill selector implementation.",
    )
    parser.add_argument(
        "--final-reporter-mode",
        default="rule_based",
        choices=["rule_based", "llm"],
        help="Final user-facing report implementation.",
    )
    parser.add_argument("--plan-llm-provider", default="", help="Planner-specific LLM provider.")
    parser.add_argument("--plan-llm-model", default="", help="Planner-specific LLM model.")
    parser.add_argument("--plan-llm-api-base", default="", help="Planner-specific LLM API base.")
    parser.add_argument(
        "--plan-llm-api-key-env",
        default="",
        help="Planner-specific API key environment variable.",
    )
    parser.add_argument(
        "--context-compressor-llm-provider",
        default="",
        help="Context-compressor-specific LLM provider.",
    )
    parser.add_argument(
        "--context-compressor-llm-model",
        default="",
        help="Context-compressor-specific LLM model.",
    )
    parser.add_argument(
        "--context-compressor-llm-api-base",
        default="",
        help="Context-compressor-specific LLM API base.",
    )
    parser.add_argument(
        "--context-compressor-llm-api-key-env",
        default="",
        help="Context-compressor-specific API key environment variable.",
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
    parser.add_argument("--memory-query-llm-provider", default="", help="Memory-query-specific LLM provider.")
    parser.add_argument("--memory-query-llm-model", default="", help="Memory-query-specific LLM model.")
    parser.add_argument("--memory-query-llm-api-base", default="", help="Memory-query-specific LLM API base.")
    parser.add_argument(
        "--memory-query-llm-api-key-env",
        default="",
        help="Memory-query-specific API key environment variable.",
    )
    parser.add_argument("--memory-rerank-llm-provider", default="", help="Memory-rerank-specific LLM provider.")
    parser.add_argument("--memory-rerank-llm-model", default="", help="Memory-rerank-specific LLM model.")
    parser.add_argument("--memory-rerank-llm-api-base", default="", help="Memory-rerank-specific LLM API base.")
    parser.add_argument(
        "--memory-rerank-llm-api-key-env",
        default="",
        help="Memory-rerank-specific API key environment variable.",
    )
    parser.add_argument(
        "--code-context-query-llm-provider",
        default="",
        help="Code-context-query-specific LLM provider.",
    )
    parser.add_argument(
        "--code-context-query-llm-model",
        default="",
        help="Code-context-query-specific LLM model.",
    )
    parser.add_argument(
        "--code-context-query-llm-api-base",
        default="",
        help="Code-context-query-specific LLM API base.",
    )
    parser.add_argument(
        "--code-context-query-llm-api-key-env",
        default="",
        help="Code-context-query-specific API key environment variable.",
    )
    parser.add_argument(
        "--code-context-rerank-llm-provider",
        default="",
        help="Code-context-rerank-specific LLM provider.",
    )
    parser.add_argument(
        "--code-context-rerank-llm-model",
        default="",
        help="Code-context-rerank-specific LLM model.",
    )
    parser.add_argument(
        "--code-context-rerank-llm-api-base",
        default="",
        help="Code-context-rerank-specific LLM API base.",
    )
    parser.add_argument(
        "--code-context-rerank-llm-api-key-env",
        default="",
        help="Code-context-rerank-specific API key environment variable.",
    )
    parser.add_argument(
        "--skill-selector-llm-provider",
        default="",
        help="Skill-selector-specific LLM provider.",
    )
    parser.add_argument(
        "--skill-selector-llm-model",
        default="",
        help="Skill-selector-specific LLM model.",
    )
    parser.add_argument(
        "--skill-selector-llm-api-base",
        default="",
        help="Skill-selector-specific LLM API base.",
    )
    parser.add_argument(
        "--skill-selector-llm-api-key-env",
        default="",
        help="Skill-selector-specific API key environment variable.",
    )
    parser.add_argument(
        "--final-reporter-llm-provider",
        default="",
        help="Final-reporter-specific LLM provider.",
    )
    parser.add_argument(
        "--final-reporter-llm-model",
        default="",
        help="Final-reporter-specific LLM model.",
    )
    parser.add_argument(
        "--final-reporter-llm-api-base",
        default="",
        help="Final-reporter-specific LLM API base.",
    )
    parser.add_argument(
        "--final-reporter-llm-api-key-env",
        default="",
        help="Final-reporter-specific API key environment variable.",
    )
    parser.add_argument("--memory-query-limit", type=int, default=5, help="Per-query memory retrieval limit.")
    parser.add_argument("--memory-selected-limit", type=int, default=12, help="Maximum selected memories.")
    parser.add_argument(
        "--memory-rerank-candidate-limit",
        type=int,
        default=24,
        help="Maximum memory candidates sent to the reranker.",
    )
    parser.add_argument(
        "--code-context-index-path",
        default=".repomind/codebase_context/index.json",
        help="Path inside the target repo for the codebase context index.",
    )
    parser.add_argument(
        "--code-context-query-limit",
        type=int,
        default=10,
        help="Per-query code context retrieval limit.",
    )
    parser.add_argument(
        "--code-context-selected-limit",
        type=int,
        default=12,
        help="Maximum selected code context candidates.",
    )
    parser.add_argument(
        "--code-context-rerank-candidate-limit",
        type=int,
        default=40,
        help="Maximum code context candidates sent to the reranker.",
    )
    parser.add_argument(
        "--skill-selected-limit",
        type=int,
        default=5,
        help="Maximum skills selected for a run.",
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
    args = parser.parse_args()
    args._provided_dests = _provided_optional_dests(parser, sys.argv[1:])
    args._parser = parser
    return args


def main() -> None:
    # 命令行没指示直接从配置里拿
    args = parse_args()
    provided = set(getattr(args, "_provided_dests", set()))
    config_path_provided = "config" in provided
    try:
        if not args.no_config:
            ensure_default_config_file(args.config)
        config_payload = (
            {}
            if args.no_config
            else load_config_payload(args.config, require_exists=config_path_provided)
        )
    except Exception as exc:
        args._parser.error(str(exc))

    # 将配置的参数加载到 agent
    config = debug_agent_config_from_dict(config_payload)
    _apply_cli_overrides(config, args, provided)
    try:
        validate_debug_agent_config(config)
    except Exception as exc:
        args._parser.error(str(exc))

    task_config = config_payload.get("task", {})
    if not isinstance(task_config, dict):
        task_config = {}
    issue = args.issue or str(task_config.get("title") or task_config.get("issue") or "").strip()
    if not issue:
        args._parser.error("issue is required unless config.json defines task.title.")
    description = (
        args.description
        if "description" in provided
        else str(task_config.get("description") or "")
    )

    if not config.repo_path:
        config.repo_path = "."
    normalize_project_runtime_paths(config)
    repo_path = Path(config.repo_path)

    env_file = _resolve_config_path(args.config, config.env_file)
    try:
        load_env_file(env_file, override=config.env_override)
    except Exception as exc:
        args._parser.error(str(exc))

    log_file = config.log_file or None
    if log_file:
        path = Path(log_file)
        if not path.is_absolute():
            path = repo_path / path
        log_file = path.as_posix()
    configure_logging(
        level=config.log_level,
        log_file=log_file,
        json_logs=config.log_json,
        console=config.log_to_console,
        force=True,
    )

    logger.info(
        "starting agent cli repo_path={} issue={} config_path={} config_loaded={} planner_mode={} context_compressor_mode={} action_policy_mode={} task_analyzer_mode={} observer_mode={} memory_query_mode={} memory_reranker_mode={} code_context_query_mode={} code_context_reranker_mode={} skill_selector_mode={} final_reporter_mode={} rl_enabled={}",
        repo_path.as_posix(),
        issue,
        None if args.no_config else args.config,
        bool(config_payload),
        config.planner_mode,
        config.context_compressor_mode,
        config.action_policy_mode,
        config.task_analyzer_mode,
        config.observer_mode,
        config.memory_query_planner_mode,
        config.memory_reranker_mode,
        config.code_context_query_planner_mode,
        config.code_context_reranker_mode,
        config.skill_selector_mode,
        config.final_reporter_mode,
        config.rl_enabled,
    )
    result = DebugAgent(config).run(title=issue, description=description)

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
        "context_compressor_mode": config.context_compressor_mode,
        "action_policy_mode": config.action_policy_mode,
        "task_analyzer_mode": config.task_analyzer_mode,
        "observer_mode": config.observer_mode,
        "memory_query_planner_mode": config.memory_query_planner_mode,
        "memory_reranker_mode": config.memory_reranker_mode,
        "code_context_query_planner_mode": config.code_context_query_planner_mode,
        "code_context_reranker_mode": config.code_context_reranker_mode,
        "skill_selector_mode": config.skill_selector_mode,
        "final_reporter_mode": config.final_reporter_mode,
        "verification_required": state.get("verification_required", True),
        "verification_reason": state.get("verification_reason", ""),
        "final_report": state.get("final_report", {}),
        "memory_query_plan": state.get("memory_query_plan", {}),
        "code_context_query_plan": state.get("code_context_query_plan", {}),
        "code_context_rerank": state.get("code_context_rerank", {}),
        "skill_selection": state.get("skill_selection", {}),
        "selected_skills": state.get("selected_skills", []),
        "selected_memories_present": bool(state.get("selected_memories")),
        "task_analysis": state.get("task_analysis", {}),
        "llm_observations": len(state.get("llm_observations", [])),
        "rl_enabled": bool(state.get("rl_enabled")),
        "rl_transitions": len(state.get("rl_transitions", [])),
        "rl_last_reward": state.get("rl_last_reward", {}),
        "promoted_memories": len(state.get("promoted_memories", [])),
        "consolidated_skills": len(state.get("consolidated_skills", [])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _apply_cli_overrides(
    config: DebugAgentConfig,
    args: argparse.Namespace,
    provided: set[str],
) -> None:
    field_overrides = {
        "repo": "repo_path",
        "verify": "verify_command",
        "max_loops": "max_loops",
        "env_file": "env_file",
        "env_override": "env_override",
        "manifest_dir": "manifest_dir",
        "memory_redis_url": "memory_redis_url",
        "context_max_tokens": "context_max_tokens",
        "context_compressor_mode": "context_compressor_mode",
        "context_compression_threshold": "context_compression_threshold",
        "planner_mode": "planner_mode",
        "action_policy_mode": "action_policy_mode",
        "task_analyzer_mode": "task_analyzer_mode",
        "observer_mode": "observer_mode",
        "memory_query_planner_mode": "memory_query_planner_mode",
        "memory_reranker_mode": "memory_reranker_mode",
        "code_context_query_planner_mode": "code_context_query_planner_mode",
        "code_context_reranker_mode": "code_context_reranker_mode",
        "skill_selector_mode": "skill_selector_mode",
        "final_reporter_mode": "final_reporter_mode",
        "memory_query_limit": "memory_query_limit",
        "memory_selected_limit": "memory_selected_limit",
        "memory_rerank_candidate_limit": "memory_rerank_candidate_limit",
        "code_context_index_path": "code_context_index_path",
        "code_context_query_limit": "code_context_query_limit",
        "code_context_selected_limit": "code_context_selected_limit",
        "code_context_rerank_candidate_limit": "code_context_rerank_candidate_limit",
        "skill_selected_limit": "skill_selected_limit",
        "rl_q_table_path": "rl_q_table_path",
        "rl_replay_path": "rl_replay_path",
        "rl_epsilon": "rl_epsilon",
        "rl_learning_rate": "rl_learning_rate",
        "rl_discount": "rl_discount",
        "log_level": "log_level",
        "log_file": "log_file",
    }
    for dest, field_name in field_overrides.items():
        if dest in provided:
            setattr(config, field_name, getattr(args, dest))

    if "disable_context_compression" in provided:
        config.context_compression_enabled = False
    if "rl_enabled" in provided:
        config.rl_enabled = True
    if "log_json" in provided:
        config.log_json = True
    if "no_console_log" in provided:
        config.log_to_console = False

    _apply_llm_cli_overrides(config.llm_config, args, provided, "llm")
    for prefix, field_name in {
        "plan": "plan_llm_config",
        "context_compressor": "context_compressor_llm_config",
        "action": "action_llm_config",
        "task_analysis": "task_analysis_llm_config",
        "observer": "observer_llm_config",
        "memory_query": "memory_query_llm_config",
        "memory_rerank": "memory_rerank_llm_config",
        "code_context_query": "code_context_query_llm_config",
        "code_context_rerank": "code_context_rerank_llm_config",
        "skill_selector": "skill_selector_llm_config",
        "final_reporter": "final_reporter_llm_config",
    }.items():
        _apply_llm_cli_overrides(getattr(config, field_name), args, provided, prefix)


def _apply_llm_cli_overrides(
    llm_config: LLMConfig,
    args: argparse.Namespace,
    provided: set[str],
    prefix: str,
) -> None:
    fields_by_dest = {
        _llm_cli_dest(prefix, "provider"): "provider",
        _llm_cli_dest(prefix, "model"): "model",
        _llm_cli_dest(prefix, "api_base"): "api_base",
        _llm_cli_dest(prefix, "api_key_env"): "api_key_env",
    }
    if prefix == "llm":
        fields_by_dest.update(
            {
                "llm_timeout": "timeout",
                "llm_temperature": "temperature",
                "llm_max_output_chars": "max_output_chars",
            }
        )
    for dest, field_name in fields_by_dest.items():
        if dest in provided:
            setattr(llm_config, field_name, getattr(args, dest))


def _llm_cli_dest(prefix: str, field_name: str) -> str:
    if prefix == "llm":
        return f"llm_{field_name}"
    return f"{prefix}_llm_{field_name}"


def _provided_optional_dests(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> set[str]:
    option_actions = parser._option_string_actions
    provided: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            provided.add(action.dest)
    return provided


def _resolve_config_path(config_path: str | None, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(config_path or DEFAULT_CONFIG_PATH)
    if not base.is_absolute():
        base = Path.cwd() / base
    return base.parent / path


if __name__ == "__main__":
    main()
