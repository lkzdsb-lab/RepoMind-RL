"""
    file name: config.py
    Author: kunze.li
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class FileConfig(object):
    """
        file 相关配置
    """
    DEBUG = False
    TESTING = False
    MAX_READ_AMOUNT = 200


@dataclass
class LLMConfig:
    provider: str = "disabled"
    model: str = ""
    api_base: str = ""
    api_key_env: str = "LLM_API_KEY"
    timeout: int = 60
    temperature: float = 0.0
    max_output_chars: int = 12000


@dataclass
class DebugAgentConfig:
    """
        agent 相关配置
    """
    # 仓库路径
    repo_path: str = ""
    review_only: bool = False
    verify_command: str = "pytest"
    max_loops: int = 8
    env_file: str | None = ".env"
    env_override: bool = False

    # 执行流程持久化路径
    trace_dir: str = ".repomind/traces"

    # 日志配置
    log_level: str = "INFO"
    log_file: str = ".repomind/logs/agent.log"
    log_json: bool = False
    log_to_console: bool = True

    # 记忆层持久化配置
    memory_path: str = ".repomind/memory.jsonl"
    mid_memory_path: str = ".repomind/memory_mid.jsonl"
    long_memory_path: str = ".repomind/memory_long.jsonl"
    skill_memory_dir: str = ".repomind/skills"
    memory_redis_url: str | None = None
    semantic_promotion_threshold: float = 0.7
    procedural_promotion_threshold: float = 1.2
    skill_consolidation_threshold: float = 1.5

    # context 压缩
    context_compression_enabled: bool = True
    context_compressor_mode: str = "rule_based"
    context_max_tokens: int = 32000
    context_compression_threshold: float = 0.75
    context_recent_items: int = 8

    # llm 配置
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    context_compressor_llm_config: LLMConfig = field(default_factory=LLMConfig)
    plan_llm_config: LLMConfig = field(default_factory=LLMConfig)
    action_llm_config: LLMConfig = field(default_factory=LLMConfig)
    task_analysis_llm_config: LLMConfig = field(default_factory=LLMConfig)
    observer_llm_config: LLMConfig = field(default_factory=LLMConfig)
    memory_query_llm_config: LLMConfig = field(default_factory=LLMConfig)
    memory_rerank_llm_config: LLMConfig = field(default_factory=LLMConfig)
    code_context_query_llm_config: LLMConfig = field(default_factory=LLMConfig)
    code_context_rerank_llm_config: LLMConfig = field(default_factory=LLMConfig)
    skill_selector_llm_config: LLMConfig = field(default_factory=LLMConfig)
    final_reporter_llm_config: LLMConfig = field(default_factory=LLMConfig)
    planner_mode: str = "heuristic"
    action_policy_mode: str = "heuristic"
    task_analyzer_mode: str = "disabled"
    observer_mode: str = "disabled"
    memory_query_planner_mode: str = "disabled"
    memory_reranker_mode: str = "disabled"
    code_context_query_planner_mode: str = "disabled"
    code_context_reranker_mode: str = "disabled"
    skill_selector_mode: str = "disabled"
    final_reporter_mode: str = "rule_based"
    memory_query_limit: int = 5
    memory_selected_limit: int = 12
    memory_rerank_candidate_limit: int = 24

    # 代码索引库
    code_context_index_path: str = ".repomind/codebase_context/index.json"
    code_context_query_limit: int = 10
    code_context_selected_limit: int = 12
    code_context_rerank_candidate_limit: int = 40
    skill_selected_limit: int = 5

    # 强化学习配置
    rl_enabled: bool = False
    rl_q_table_path: str = ".repomind/rl/q_table.json"
    rl_replay_path: str = ".repomind/rl/replay.jsonl"
    rl_epsilon: float = 0.15
    rl_learning_rate: float = 0.2
    rl_discount: float = 0.9
    rl_replay_max_size: int = 10000
    rl_train_batch_size: int = 32
    manifest_dir: str | None = None


DEFAULT_CONFIG_PATH = "config.json"


def load_config_payload(
    path: str | Path | None = DEFAULT_CONFIG_PATH,
    *,
    require_exists: bool = False,
) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        if require_exists:
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        return {}
    if not config_path.is_file():
        raise IsADirectoryError(f"Config path is not a file: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def load_debug_agent_config(
    path: str | Path | None = DEFAULT_CONFIG_PATH,
    *,
    require_exists: bool = False,
) -> DebugAgentConfig:
    config = DebugAgentConfig()
    apply_debug_agent_config(config, load_config_payload(path, require_exists=require_exists))
    return config


def load_env_file(
    path: str | Path | None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """
        从 .env 文件获取配置
    """
    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}
    if not env_path.is_file():
        raise IsADirectoryError(f"Env path is not a file: {env_path}")

    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if not key:
            raise ValueError(f"Invalid env key at {env_path}:{line_number}")
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def debug_agent_config_from_dict(data: dict[str, Any] | None) -> DebugAgentConfig:
    config = DebugAgentConfig()
    apply_debug_agent_config(config, data or {})
    return config


def apply_debug_agent_config(config: DebugAgentConfig, data: dict[str, Any]) -> DebugAgentConfig:
    if not isinstance(data, dict):
        raise ValueError("Debug agent config payload must be a JSON object.")

    _apply_section(
        config,
        data,
        {
            "repo": "repo_path",
            "verify": "verify_command",
        },
    )
    _apply_section(
        config,
        data.get("logging"),
        {
            "level": "log_level",
            "file": "log_file",
            "json": "log_json",
            "to_console": "log_to_console",
        },
    )
    _apply_section(
        config,
        data.get("memory"),
        {
            "path": "memory_path",
            "mid_path": "mid_memory_path",
            "long_path": "long_memory_path",
            "skill_dir": "skill_memory_dir",
            "redis_url": "memory_redis_url",
            "semantic_threshold": "semantic_promotion_threshold",
            "procedural_threshold": "procedural_promotion_threshold",
            "skill_consolidation_threshold": "skill_consolidation_threshold",
            "query_limit": "memory_query_limit",
            "selected_limit": "memory_selected_limit",
            "rerank_candidate_limit": "memory_rerank_candidate_limit",
        },
    )
    _apply_section(
        config,
        data.get("context"),
        {
            "enabled": "context_compression_enabled",
            "compression_enabled": "context_compression_enabled",
            "compressor_mode": "context_compressor_mode",
            "max_tokens": "context_max_tokens",
            "compression_threshold": "context_compression_threshold",
            "recent_items": "context_recent_items",
        },
    )
    _apply_section(
        config,
        data.get("modes"),
        {
            "planner": "planner_mode",
            "context_compressor": "context_compressor_mode",
            "action_policy": "action_policy_mode",
            "task_analyzer": "task_analyzer_mode",
            "observer": "observer_mode",
            "memory_query_planner": "memory_query_planner_mode",
            "memory_reranker": "memory_reranker_mode",
            "code_context_query_planner": "code_context_query_planner_mode",
            "code_context_reranker": "code_context_reranker_mode",
            "skill_selector": "skill_selector_mode",
            "final_reporter": "final_reporter_mode",
        },
    )
    _apply_section(
        config,
        data.get("code_context"),
        {
            "index_path": "code_context_index_path",
            "query_limit": "code_context_query_limit",
            "selected_limit": "code_context_selected_limit",
            "rerank_candidate_limit": "code_context_rerank_candidate_limit",
        },
    )
    _apply_section(
        config,
        data.get("skill"),
        {
            "selected_limit": "skill_selected_limit",
        },
    )
    _apply_section(
        config,
        data.get("rl"),
        {
            "enabled": "rl_enabled",
            "q_table_path": "rl_q_table_path",
            "replay_path": "rl_replay_path",
            "epsilon": "rl_epsilon",
            "learning_rate": "rl_learning_rate",
            "discount": "rl_discount",
            "replay_max_size": "rl_replay_max_size",
            "train_batch_size": "rl_train_batch_size",
        },
    )
    _apply_llm_section(config, data.get("llm"))
    return config


def _apply_section(
    config: DebugAgentConfig,
    section: Any,
    aliases: dict[str, str] | None = None,
) -> None:
    if not isinstance(section, dict):
        return
    aliases = aliases or {}
    config_fields = {item.name for item in fields(DebugAgentConfig)}
    for raw_key, value in section.items():
        key = _normalize_key(raw_key)
        target = aliases.get(key, key)
        if target not in config_fields:
            continue
        if target.endswith("_llm_config") and isinstance(value, dict):
            setattr(config, target, _llm_config_from_dict(value, getattr(config, target)))
            continue
        if target == "llm_config" and isinstance(value, dict):
            setattr(config, target, _llm_config_from_dict(value, config.llm_config))
            continue
        setattr(config, target, value)


def _apply_llm_section(config: DebugAgentConfig, section: Any) -> None:
    if not isinstance(section, dict):
        return

    base_values = {
        _normalize_key(key): value
        for key, value in section.items()
        if _normalize_key(key) in _llm_field_names()
    }
    if base_values:
        config.llm_config = _llm_config_from_dict(base_values, config.llm_config)

    component_fields = {
        "plan": "plan_llm_config",
        "planner": "plan_llm_config",
        "context_compressor": "context_compressor_llm_config",
        "context": "context_compressor_llm_config",
        "action": "action_llm_config",
        "action_policy": "action_llm_config",
        "task_analysis": "task_analysis_llm_config",
        "observer": "observer_llm_config",
        "memory_query": "memory_query_llm_config",
        "memory_rerank": "memory_rerank_llm_config",
        "code_context_query": "code_context_query_llm_config",
        "code_context_rerank": "code_context_rerank_llm_config",
        "skill_selector": "skill_selector_llm_config",
        "final_reporter": "final_reporter_llm_config",
    }
    for raw_key, value in section.items():
        target = component_fields.get(_normalize_key(raw_key))
        if target and isinstance(value, dict):
            setattr(config, target, _llm_config_from_dict(value, getattr(config, target)))


def validate_debug_agent_config(config: DebugAgentConfig) -> None:
    _validate_choice("planner_mode", config.planner_mode, {"heuristic", "llm"})
    _validate_choice(
        "context_compressor_mode",
        config.context_compressor_mode,
        {"disabled", "rule_based", "llm"},
    )
    _validate_choice("action_policy_mode", config.action_policy_mode, {"heuristic", "rl", "llm"})
    _validate_choice("final_reporter_mode", config.final_reporter_mode, {"rule_based", "llm"})
    for field_name in (
        "task_analyzer_mode",
        "observer_mode",
        "memory_query_planner_mode",
        "memory_reranker_mode",
        "code_context_query_planner_mode",
        "code_context_reranker_mode",
        "skill_selector_mode",
    ):
        _validate_choice(field_name, getattr(config, field_name), {"disabled", "llm"})

    for field_name in (
        "llm_config",
        "context_compressor_llm_config",
        "plan_llm_config",
        "action_llm_config",
        "task_analysis_llm_config",
        "observer_llm_config",
        "memory_query_llm_config",
        "memory_rerank_llm_config",
        "code_context_query_llm_config",
        "code_context_rerank_llm_config",
        "skill_selector_llm_config",
        "final_reporter_llm_config",
    ):
        _validate_llm_config(field_name, getattr(config, field_name))

    for field_name in (
        "max_loops",
        "context_max_tokens",
        "context_recent_items",
        "memory_query_limit",
        "memory_selected_limit",
        "memory_rerank_candidate_limit",
        "code_context_query_limit",
        "code_context_selected_limit",
        "code_context_rerank_candidate_limit",
        "skill_selected_limit",
        "rl_replay_max_size",
        "rl_train_batch_size",
    ):
        if int(getattr(config, field_name)) <= 0:
            raise ValueError(f"{field_name} must be greater than 0")

    _require_llm_config(
        "modes.planner",
        config.planner_mode == "llm",
        resolve_llm_config(config.llm_config, config.plan_llm_config),
    )
    _require_llm_config(
        "modes.context_compressor",
        config.context_compressor_mode == "llm",
        resolve_llm_config(config.llm_config, config.context_compressor_llm_config),
    )
    _require_llm_config(
        "modes.action_policy",
        config.action_policy_mode == "llm",
        resolve_llm_config(config.llm_config, config.action_llm_config),
    )
    _require_llm_config(
        "modes.task_analyzer",
        config.task_analyzer_mode == "llm",
        resolve_llm_config(config.llm_config, config.task_analysis_llm_config),
    )
    _require_llm_config(
        "modes.observer",
        config.observer_mode == "llm",
        resolve_llm_config(config.llm_config, config.observer_llm_config),
    )
    _require_llm_config(
        "modes.memory_query_planner",
        config.memory_query_planner_mode == "llm",
        resolve_llm_config(config.llm_config, config.memory_query_llm_config),
    )
    _require_llm_config(
        "modes.memory_reranker",
        config.memory_reranker_mode == "llm",
        resolve_llm_config(config.llm_config, config.memory_rerank_llm_config),
    )
    _require_llm_config(
        "modes.code_context_query_planner",
        config.code_context_query_planner_mode == "llm",
        resolve_llm_config(config.llm_config, config.code_context_query_llm_config),
    )
    _require_llm_config(
        "modes.code_context_reranker",
        config.code_context_reranker_mode == "llm",
        resolve_llm_config(config.llm_config, config.code_context_rerank_llm_config),
    )
    _require_llm_config(
        "modes.skill_selector",
        config.skill_selector_mode == "llm",
        resolve_llm_config(config.llm_config, config.skill_selector_llm_config),
    )
    _require_llm_config(
        "modes.final_reporter",
        config.final_reporter_mode == "llm",
        resolve_llm_config(config.llm_config, config.final_reporter_llm_config),
    )


def normalize_project_runtime_paths(config: DebugAgentConfig) -> DebugAgentConfig:
    """
    Resolve runtime artifacts into the target repo so different debug targets do
    not share memory, traces, logs, code indexes, or RL data.
    """
    repo_path = Path(config.repo_path or ".").resolve()
    config.repo_path = repo_path.as_posix()
    for field_name in (
        "trace_dir",
        "log_file",
        "memory_path",
        "mid_memory_path",
        "long_memory_path",
        "skill_memory_dir",
        "code_context_index_path",
        "rl_q_table_path",
        "rl_replay_path",
    ):
        value = getattr(config, field_name)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = repo_path / path
        setattr(config, field_name, path.as_posix())
    return config


def resolve_llm_config(base: LLMConfig, override: LLMConfig) -> LLMConfig:
    default = LLMConfig()

    def resolve_str(field: str) -> str:
        value = getattr(override, field)
        default_value = getattr(default, field)
        return value if value and value != default_value else getattr(base, field)

    return LLMConfig(
        provider=resolve_str("provider"),
        model=resolve_str("model"),
        api_base=resolve_str("api_base"),
        api_key_env=resolve_str("api_key_env"),
        timeout=override.timeout if override.timeout != default.timeout else base.timeout,
        temperature=(
            override.temperature
            if override.temperature != default.temperature
            else base.temperature
        ),
        max_output_chars=(
            override.max_output_chars
            if override.max_output_chars != default.max_output_chars
            else base.max_output_chars
        ),
    )


def _llm_config_from_dict(data: dict[str, Any], base: LLMConfig | None = None) -> LLMConfig:
    current = base or LLMConfig()
    values = {
        item.name: getattr(current, item.name)
        for item in fields(LLMConfig)
    }
    for raw_key, value in data.items():
        key = _normalize_key(raw_key)
        if key in values:
            values[key] = value
    return LLMConfig(**values)


def _llm_field_names() -> set[str]:
    return {item.name for item in fields(LLMConfig)}


def _validate_choice(field_name: str, value: Any, choices: set[str]) -> None:
    parsed = str(value).strip().lower()
    if parsed not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}, got {value!r}")


def _validate_llm_config(field_name: str, value: LLMConfig) -> None:
    provider = str(value.provider).strip().lower()
    allowed = {"", "disabled", "none", "openai", "openai_compatible", "openai-compatible", "enable"}
    if provider not in allowed:
        raise ValueError(
            f"{field_name}.provider must be one of {sorted(allowed)}, got {value.provider!r}"
        )
    if int(value.timeout) <= 0:
        raise ValueError(f"{field_name}.timeout must be greater than 0")
    if int(value.max_output_chars) <= 0:
        raise ValueError(f"{field_name}.max_output_chars must be greater than 0")


def _require_llm_config(owner: str, enabled: bool, value: LLMConfig) -> None:
    if not enabled:
        return
    provider = str(value.provider).strip().lower()
    if provider in {"", "disabled", "none"}:
        raise ValueError(f"{owner}=llm requires an enabled llm provider")
    if not str(value.model).strip():
        raise ValueError(f"{owner}=llm requires llm.model or a module-specific model")


def _normalize_key(value: Any) -> str:
    return str(value).strip().replace("-", "_")


def _parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    key, separator, value = text.partition("=")
    if not separator:
        return None
    key = key.strip()
    value = value.strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]
    return key, value


class SearchQueryConfig(object):
    """
        search query 相关配置
    """
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "when",
        "with",
        "bug",
        "fix",
        "issue",
        "problem",
        "error",
        "failed",
        "failure",
        "一个",
        "这个",
        "偶尔",
        "不会",
        "定位",
        "修复",
        "问题",
        "项目",
        "失败",
        "错误",
        "异常",
    }
    CHINESE_STOP_FRAGMENTS = ("不会", "不能", "无法", "失败", "问题", "修复", "定位", "异常")

class ManifestConfig(object):
    """
        manifest loader
    """
    SUPPORTED_SUFFIXES = {".json", ".toml"}

"""
    memory loader
"""
class MemoryConfig(object):
    pass

class CompressionConfig(object):
    """
    压缩层配置
    """
    IGNORED_DIRS = {
        ".git",
        ".repomind",
        ".venv",
        "__pycache__",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "target",
        "coverage",
    }
    INDEXED_EXTENSIONS = {
        ".go": "go",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".sql": "sql",
        ".proto": "proto",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
    }
    CALL_EXCLUDE = {
        "if",
        "for",
        "switch",
        "select",
        "return",
        "func",
        "range",
        "go",
        "defer",
        "make",
        "new",
        "append",
        "len",
        "cap",
        "copy",
        "delete",
        "panic",
        "recover",
    }

class CodeBaseConig(object):
    """ 代码索引配置 """
    DEFAULT_INDEX_PATH = ".repomind/codebase_context/index.json"
