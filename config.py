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
    MAX_LIST_FILE_LIMIT = 200
    MAX_READ_FILE_LIMIT = 8000
    MAX_FILE_BYTES_LIMIT = 500_000
    LIST_IGNORED_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".repomind",
        "node_modules",
        "vendor",
        "__pycache__",
        "dist",
        "build",
        "coverage",
        ".next",
        ".turbo",
    }


@dataclass
class LLMConfig:
    provider: str = "disabled"
    model: str = ""
    api_base: str = ""
    api_key_env: str = "LLM_API_KEY"
    timeout: int = 60
    temperature: float = 0.0
    max_output_chars: int = 12000
    response_format_mode: str = "auto"
    structured_fallback: bool = True
    max_retries: int = 2
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    provider: str = "disabled"
    model: str = ""
    api_base: str = ""
    api_key_env: str = "LLM_API_KEY"
    timeout: int = 60
    dimensions: int = 0
    batch_size: int = 32


@dataclass
class DebugAgentConfig:
    """
        agent 相关配置
    """
    # 仓库路径
    repo_path: str = ""
    max_loops: int = 8
    execution_enabled: bool = False
    execution_env: dict[str, str] = field(default_factory=dict)
    execution_timeout: int = 300
    execution_model_calls: int = 16
    execution_tool_calls: int = 32
    execution_retries: int = 8
    execution_startup_timeout: int = 30
    execution_operation_timeout: int = 30
    execution_cleanup_timeout: int = 5
    execution_resources: dict = field(default_factory=lambda: {
        "memory_limit_mb": 2048, "cpu_limit_percent": 50, "max_processes": 64, "mode": "strict"})
    env_file: str | None = ".env"
    env_override: bool = False

    # 执行流程持久化路径
    trace_dir: str = ".repomind/traces"

    # 日志配置
    log_level: str = "INFO"
    log_file: str = ".repomind/logs/agent.log"
    log_json: bool = False
    log_to_console: bool = True

    # 在线记忆只维护当前会话；长期晋升由未来的离线 worker 负责。
    session_memory_path: str = ".repomind/session_memory.db"
    session_memory_max_turns: int = 6
    session_memory_max_chars: int = 12000
    session_file_cache_max_files: int = 50
    session_file_cache_max_spans: int = 200
    session_file_cache_max_bytes: int = 10 * 1024 * 1024

    # 不可变任务归档是长期记忆离线重建的事实来源。
    task_archive_enabled: bool = True
    task_archive_path: str = ".repomind/archive/tasks"
    task_archive_max_text_chars: int = 200_000
    task_archive_max_source_file_bytes: int = 200_000
    task_archive_max_source_total_bytes: int = 2_000_000

    # Markdown 是长期记忆的权威表达；Catalog 和向量索引是可重建投影。
    long_term_memory_path: str = ".repomind/memory"
    memory_catalog_path: str = ".repomind/memory/catalog.sqlite3"
    memory_semantic_index_path: str = ".repomind/memory/semantic.sqlite3"
    memory_catalog_busy_timeout_ms: int = 5000
    memory_keyword_candidate_multiplier: int = 8
    long_term_memory_retrieval_enabled: bool = True
    long_term_memory_retrieval_mode: str = "keyword"
    long_term_memory_semantic_candidates: int = 24
    long_term_memory_semantic_min_score: float = 0.5
    long_term_memory_semantic_timeout: float = 5.0
    long_term_memory_query_cache_size: int = 256
    long_term_memory_rrf_k: int = 60
    long_term_memory_keyword_weight: float = 1.0
    long_term_memory_semantic_weight: float = 1.0
    long_term_memory_scope_matched_weight: float = 1.1
    long_term_memory_scope_unknown_weight: float = 0.5
    long_term_memory_scope_unknown_limit: int = 2
    long_term_memory_retrieval_limit: int = 8
    long_term_memory_retrieval_max_chars: int = 12000
    long_term_memory_retrieval_min_score: float = 0.1
    long_term_memory_retrieval_include_draft: bool = True
    long_term_memory_retrieval_max_refreshes: int = 3
    long_term_memory_retrieval_max_queries: int = 4
    long_term_memory_retrieval_type_limits: dict[str, int] = field(
        default_factory=lambda: {
            "preference": 2,
            "semantic": 4,
            "procedural": 2,
            "anti_pattern": 2,
            "episodic": 2,
        }
    )
    consolidation_run_path: str = ".repomind/consolidation/runs"
    consolidation_pipeline_version: str = "consolidation-v3"
    memory_extractor_mode: str = "rule_based"
    memory_consolidation_max_candidates: int = 24
    memory_semantic_merge_mode: str = "disabled"
    memory_semantic_top_k: int = 5
    memory_semantic_min_similarity: float = 0.78
    memory_semantic_relation_min_confidence: float = 0.85
    memory_embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # context 压缩
    context_compression_enabled: bool = True
    context_compressor_mode: str = "rule_based"
    context_max_tokens: int = 32000
    context_compression_threshold: float = 0.75
    context_compression_target: float = 0.55
    context_recent_items: int = 8
    context_min_new_tokens: int = 1200

    # llm 配置
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    context_compressor_llm_config: LLMConfig = field(default_factory=LLMConfig)
    plan_llm_config: LLMConfig = field(default_factory=LLMConfig)
    action_llm_config: LLMConfig = field(default_factory=LLMConfig)
    task_analysis_llm_config: LLMConfig = field(default_factory=LLMConfig)
    observer_llm_config: LLMConfig = field(default_factory=LLMConfig)
    memory_extractor_llm_config: LLMConfig = field(default_factory=LLMConfig)
    memory_relation_llm_config: LLMConfig = field(default_factory=LLMConfig)
    code_context_query_llm_config: LLMConfig = field(default_factory=LLMConfig)
    code_context_rerank_llm_config: LLMConfig = field(default_factory=LLMConfig)
    skill_selector_llm_config: LLMConfig = field(default_factory=LLMConfig)
    final_reporter_llm_config: LLMConfig = field(default_factory=LLMConfig)
    completion_judge_llm_config: LLMConfig = field(default_factory=LLMConfig)
    planner_mode: str = "heuristic"
    action_policy_mode: str = "llm"
    task_analyzer_mode: str = "disabled"
    observer_mode: str = "disabled"
    observer_use_delta: bool = True
    observer_full_state_on_severe: bool = True
    observer_write_threshold: float = 0.35
    observer_store_limit: int = 12
    code_context_query_planner_mode: str = "disabled"
    code_context_reranker_mode: str = "disabled"
    skill_selector_mode: str = "disabled"
    final_reporter_mode: str = "rule_based"
    completion_judge_mode: str = "auto"

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

    # 受限代码编辑能力。默认关闭，避免现有 debug 流程意外写文件。
    editing_enabled: bool = False
    editing_max_files: int = 5
    editing_max_changed_lines: int = 300
    editing_max_file_bytes: int = 200000
    editing_require_read_before_write: bool = True
    editing_confidence_threshold: float = 0.75
    editing_allow_create: bool = False

    # 人工审批。开启后每个 agent action 执行前都会暂停等待用户批准。
    require_step_approval: bool = False


DEFAULT_CONFIG_PATH = "config.json"
CONFIG_ENV_VAR = "REPOMIND_CONFIG"


@dataclass(frozen=True)
class ResolvedConfigPath:
    """An absolute configuration path and the source that selected it."""

    path: Path
    source: str


class ConfigLocator:
    """Locate Agent configuration without depending on the target repo cwd."""

    def __init__(
        self,
        *,
        runtime_root: str | Path | None = None,
        invocation_dir: str | Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root or Path(__file__).resolve().parent).resolve()
        self.invocation_dir = Path(invocation_dir or Path.cwd()).resolve()
        self.environ = os.environ if environ is None else environ

    def resolve(self, explicit_path: str | Path | None = None) -> ResolvedConfigPath:
        """Resolve an existing config; explicit relative paths belong to the caller cwd."""
        if explicit_path is not None:
            return self._require_file(
                self._absolute(explicit_path, self.invocation_dir),
                source="explicit",
            )

        configured = str(self.environ.get(CONFIG_ENV_VAR, "")).strip()
        if configured:
            return self._require_file(
                self._absolute(configured, self.invocation_dir),
                source="environment",
            )

        candidates = [
            (self.runtime_root / DEFAULT_CONFIG_PATH, "runtime"),
            (self._user_config_dir() / DEFAULT_CONFIG_PATH, "user"),
        ]
        for candidate, source in candidates:
            if candidate.is_file():
                return ResolvedConfigPath(candidate.resolve(), source)

        searched = ", ".join(str(path.resolve()) for path, _ in candidates)
        raise FileNotFoundError(
            "No RepoMind config file was found. "
            f"Set {CONFIG_ENV_VAR}, pass --config, or create one at: {searched}"
        )

    def _absolute(self, value: str | Path, base: Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    def _require_file(self, path: Path, *, source: str) -> ResolvedConfigPath:
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Config path is not a file: {path}")
        return ResolvedConfigPath(path, source)

    def _user_config_dir(self) -> Path:
        local_app_data = str(self.environ.get("LOCALAPPDATA", "")).strip()
        if local_app_data:
            return Path(local_app_data).expanduser() / "RepoMind"
        xdg_config_home = str(self.environ.get("XDG_CONFIG_HOME", "")).strip()
        if xdg_config_home:
            return Path(xdg_config_home).expanduser() / "repomind"
        return Path.home() / ".config" / "repomind"


def locate_config_file(
    path: str | Path | None = None,
    *,
    invocation_dir: str | Path | None = None,
) -> Path:
    """Public shared resolver used by the CLI and isolated evaluation processes."""
    return ConfigLocator(invocation_dir=invocation_dir).resolve(path).path


def default_config_payload() -> dict[str, Any]:
    config = DebugAgentConfig()
    llm = LLMConfig()
    return {
        "$schema": "./config.schema.json",
        "task": {
            "title": "",
            "description": "",
        },
        "repo_path": ".",
        "max_loops": config.max_loops,
        "trace_dir": config.trace_dir,
        "execution_enabled": config.execution_enabled,
        "execution_env": config.execution_env,
        "execution_timeout": config.execution_timeout,
        "execution_model_calls": config.execution_model_calls,
        "execution_tool_calls": config.execution_tool_calls,
        "execution_retries": config.execution_retries,
        "execution_startup_timeout": config.execution_startup_timeout,
        "execution_operation_timeout": config.execution_operation_timeout,
        "execution_cleanup_timeout": config.execution_cleanup_timeout,
        "execution_resources": config.execution_resources,
        "env_file": config.env_file,
        "env_override": config.env_override,
        "manifest_dir": config.manifest_dir,
        "logging": {
            "level": config.log_level,
            "file": config.log_file,
            "json": config.log_json,
            "to_console": config.log_to_console,
        },
        "llm": {
            "provider": llm.provider,
            "model": llm.model,
            "api_base": llm.api_base,
            "api_key_env": llm.api_key_env,
            "timeout": llm.timeout,
            "temperature": llm.temperature,
            "max_output_chars": llm.max_output_chars,
            "response_format_mode": llm.response_format_mode,
            "structured_fallback": llm.structured_fallback,
            "max_retries": llm.max_retries,
            "max_completion_tokens": llm.max_completion_tokens,
            "reasoning_effort": llm.reasoning_effort,
            "extra_body": dict(llm.extra_body),
            "context_compressor": {},
            "plan": {},
            "action": {},
            "task_analysis": {},
            "observer": {},
            "memory_extractor": {},
            "memory_relation": {},
            "code_context_query": {},
            "code_context_rerank": {},
            "skill_selector": {},
            "final_reporter": {},
            "completion_judge": {},
        },
        "modes": {
            "planner": config.planner_mode,
            "context_compressor": config.context_compressor_mode,
            "action_policy": config.action_policy_mode,
            "task_analyzer": config.task_analyzer_mode,
            "observer": config.observer_mode,
            "code_context_query_planner": config.code_context_query_planner_mode,
            "code_context_reranker": config.code_context_reranker_mode,
            "skill_selector": config.skill_selector_mode,
            "final_reporter": config.final_reporter_mode,
            "completion_judge": config.completion_judge_mode,
        },
        "memory": {
            "path": config.session_memory_path,
            "max_turns": config.session_memory_max_turns,
            "max_chars": config.session_memory_max_chars,
            "file_cache_max_files": config.session_file_cache_max_files,
            "file_cache_max_spans": config.session_file_cache_max_spans,
            "file_cache_max_bytes": config.session_file_cache_max_bytes,
        },
        "archive": {
            "enabled": config.task_archive_enabled,
            "path": config.task_archive_path,
            "max_text_chars": config.task_archive_max_text_chars,
            "max_source_file_bytes": config.task_archive_max_source_file_bytes,
            "max_source_total_bytes": config.task_archive_max_source_total_bytes,
        },
        "long_term_memory": {
            "document_path": config.long_term_memory_path,
            "catalog_path": config.memory_catalog_path,
            "semantic_index_path": config.memory_semantic_index_path,
            "catalog_busy_timeout_ms": config.memory_catalog_busy_timeout_ms,
            "keyword_candidate_multiplier": config.memory_keyword_candidate_multiplier,
            "retrieval_enabled": config.long_term_memory_retrieval_enabled,
            "retrieval_mode": config.long_term_memory_retrieval_mode,
            "semantic_candidates": config.long_term_memory_semantic_candidates,
            "semantic_min_score": config.long_term_memory_semantic_min_score,
            "semantic_timeout": config.long_term_memory_semantic_timeout,
            "query_cache_size": config.long_term_memory_query_cache_size,
            "rrf_k": config.long_term_memory_rrf_k,
            "keyword_weight": config.long_term_memory_keyword_weight,
            "semantic_weight": config.long_term_memory_semantic_weight,
            "scope_matched_weight": config.long_term_memory_scope_matched_weight,
            "scope_unknown_weight": config.long_term_memory_scope_unknown_weight,
            "scope_unknown_limit": config.long_term_memory_scope_unknown_limit,
            "retrieval_limit": config.long_term_memory_retrieval_limit,
            "retrieval_max_chars": config.long_term_memory_retrieval_max_chars,
            "retrieval_min_score": config.long_term_memory_retrieval_min_score,
            "retrieval_include_draft": config.long_term_memory_retrieval_include_draft,
            "retrieval_max_refreshes": config.long_term_memory_retrieval_max_refreshes,
            "retrieval_max_queries": config.long_term_memory_retrieval_max_queries,
            "retrieval_type_limits": dict(config.long_term_memory_retrieval_type_limits),
            "consolidation_path": config.consolidation_run_path,
            "pipeline_version": config.consolidation_pipeline_version,
            "extractor_mode": config.memory_extractor_mode,
            "max_candidates": config.memory_consolidation_max_candidates,
            "semantic_merge_mode": config.memory_semantic_merge_mode,
            "semantic_top_k": config.memory_semantic_top_k,
            "semantic_min_similarity": config.memory_semantic_min_similarity,
            "semantic_relation_min_confidence": config.memory_semantic_relation_min_confidence,
            "embedding": {
                field.name: getattr(config.memory_embedding_config, field.name)
                for field in fields(EmbeddingConfig)
            },
        },
        "context": {
            "enabled": config.context_compression_enabled,
            "compressor_mode": config.context_compressor_mode,
            "max_tokens": config.context_max_tokens,
            "compression_threshold": config.context_compression_threshold,
            "compression_target": config.context_compression_target,
            "recent_items": config.context_recent_items,
            "min_new_tokens": config.context_min_new_tokens,
        },
        "observer": {
            "use_delta": config.observer_use_delta,
            "full_state_on_severe": config.observer_full_state_on_severe,
            "write_threshold": config.observer_write_threshold,
            "store_limit": config.observer_store_limit,
        },
        "code_context": {
            "index_path": config.code_context_index_path,
            "query_limit": config.code_context_query_limit,
            "selected_limit": config.code_context_selected_limit,
            "rerank_candidate_limit": config.code_context_rerank_candidate_limit,
        },
        "skill": {
            "selected_limit": config.skill_selected_limit,
        },
        "rl": {
            "enabled": config.rl_enabled,
            "q_table_path": config.rl_q_table_path,
            "replay_path": config.rl_replay_path,
            "epsilon": config.rl_epsilon,
            "learning_rate": config.rl_learning_rate,
            "discount": config.rl_discount,
            "replay_max_size": config.rl_replay_max_size,
            "train_batch_size": config.rl_train_batch_size,
        },
        "editing": {
            "enabled": config.editing_enabled,
            "max_files": config.editing_max_files,
            "max_changed_lines": config.editing_max_changed_lines,
            "max_file_bytes": config.editing_max_file_bytes,
            "require_read_before_write": config.editing_require_read_before_write,
            "confidence_threshold": config.editing_confidence_threshold,
            "allow_create": config.editing_allow_create,
        },
        "approval": {
            "require_step_approval": config.require_step_approval,
        },
    }


def ensure_default_config_file(path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    config_path = Path(path)
    if config_path.exists():
        return config_path
    if config_path.parent != Path("."):
        config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(default_config_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


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
            "step_approval": "require_step_approval",
            "require_step_approval": "require_step_approval",
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
            "path": "session_memory_path",
            "max_turns": "session_memory_max_turns",
            "max_chars": "session_memory_max_chars",
            "file_cache_max_files": "session_file_cache_max_files",
            "file_cache_max_spans": "session_file_cache_max_spans",
            "file_cache_max_bytes": "session_file_cache_max_bytes",
        },
    )
    _apply_section(
        config,
        data.get("archive"),
        {
            "enabled": "task_archive_enabled",
            "path": "task_archive_path",
            "max_text_chars": "task_archive_max_text_chars",
            "max_source_file_bytes": "task_archive_max_source_file_bytes",
            "max_source_total_bytes": "task_archive_max_source_total_bytes",
        },
    )
    _apply_section(
        config,
        data.get("long_term_memory"),
        {
            "document_path": "long_term_memory_path",
            "catalog_path": "memory_catalog_path",
            "semantic_index_path": "memory_semantic_index_path",
            "catalog_busy_timeout_ms": "memory_catalog_busy_timeout_ms",
            "keyword_candidate_multiplier": "memory_keyword_candidate_multiplier",
            "retrieval_enabled": "long_term_memory_retrieval_enabled",
            "retrieval_mode": "long_term_memory_retrieval_mode",
            "semantic_candidates": "long_term_memory_semantic_candidates",
            "semantic_min_score": "long_term_memory_semantic_min_score",
            "semantic_timeout": "long_term_memory_semantic_timeout",
            "query_cache_size": "long_term_memory_query_cache_size",
            "rrf_k": "long_term_memory_rrf_k",
            "keyword_weight": "long_term_memory_keyword_weight",
            "semantic_weight": "long_term_memory_semantic_weight",
            "scope_matched_weight": "long_term_memory_scope_matched_weight",
            "scope_unknown_weight": "long_term_memory_scope_unknown_weight",
            "scope_unknown_limit": "long_term_memory_scope_unknown_limit",
            "retrieval_limit": "long_term_memory_retrieval_limit",
            "retrieval_max_chars": "long_term_memory_retrieval_max_chars",
            "retrieval_min_score": "long_term_memory_retrieval_min_score",
            "retrieval_include_draft": "long_term_memory_retrieval_include_draft",
            "retrieval_max_refreshes": "long_term_memory_retrieval_max_refreshes",
            "retrieval_max_queries": "long_term_memory_retrieval_max_queries",
            "retrieval_type_limits": "long_term_memory_retrieval_type_limits",
            "consolidation_path": "consolidation_run_path",
            "pipeline_version": "consolidation_pipeline_version",
            "extractor_mode": "memory_extractor_mode",
            "max_candidates": "memory_consolidation_max_candidates",
            "semantic_merge_mode": "memory_semantic_merge_mode",
            "semantic_top_k": "memory_semantic_top_k",
            "semantic_min_similarity": "memory_semantic_min_similarity",
            "semantic_relation_min_confidence": "memory_semantic_relation_min_confidence",
            "embedding": "memory_embedding_config",
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
            "compression_target": "context_compression_target",
            "recent_items": "context_recent_items",
            "min_new_tokens": "context_min_new_tokens",
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
            "code_context_query_planner": "code_context_query_planner_mode",
            "code_context_reranker": "code_context_reranker_mode",
            "skill_selector": "skill_selector_mode",
            "final_reporter": "final_reporter_mode",
            "completion_judge": "completion_judge_mode",
        },
    )
    _apply_section(
        config,
        data.get("observer"),
        {
            "use_delta": "observer_use_delta",
            "full_state_on_severe": "observer_full_state_on_severe",
            "write_threshold": "observer_write_threshold",
            "store_limit": "observer_store_limit",
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
    _apply_section(
        config,
        data.get("editing"),
        {
            "enabled": "editing_enabled",
            "max_files": "editing_max_files",
            "max_changed_lines": "editing_max_changed_lines",
            "max_file_bytes": "editing_max_file_bytes",
            "require_read_before_write": "editing_require_read_before_write",
            "confidence_threshold": "editing_confidence_threshold",
            "allow_create": "editing_allow_create",
        },
    )
    _apply_section(
        config,
        data.get("approval"),
        {
            "require_step_approval": "require_step_approval",
            "step_approval": "require_step_approval",
            "each_step": "require_step_approval",
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
        if target == "memory_embedding_config" and isinstance(value, dict):
            setattr(config, target, _embedding_config_from_dict(value, getattr(config, target)))
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
        "memory_extractor": "memory_extractor_llm_config",
        "memory_relation": "memory_relation_llm_config",
        "code_context_query": "code_context_query_llm_config",
        "code_context_rerank": "code_context_rerank_llm_config",
        "skill_selector": "skill_selector_llm_config",
        "final_reporter": "final_reporter_llm_config",
        "completion_judge": "completion_judge_llm_config",
    }
    for raw_key, value in section.items():
        target = component_fields.get(_normalize_key(raw_key))
        if target and isinstance(value, dict):
            setattr(config, target, _llm_config_from_dict(value, getattr(config, target)))


def validate_debug_agent_config(config: DebugAgentConfig) -> None:
    if not 0 < config.context_compression_target < config.context_compression_threshold <= 1:
        raise ValueError("context requires 0 < compression_target < compression_threshold <= 1")
    from agent_runtime.execution.contracts import Limits, ResourceLimits
    Limits(timeout=config.execution_timeout, model_calls=config.execution_model_calls,
           tool_calls=config.execution_tool_calls, retries=config.execution_retries,
           startup_timeout=config.execution_startup_timeout, operation_timeout=config.execution_operation_timeout,
           cleanup_timeout=config.execution_cleanup_timeout)
    ResourceLimits.model_validate(config.execution_resources)
    if not isinstance(config.execution_enabled, bool):
        raise ValueError("execution_enabled must be a boolean")
    if not isinstance(config.execution_env, dict) or any(
        not isinstance(key, str) or not key or "=" in key or "\0" in key
        or not isinstance(value, str) or "\0" in value
        for key, value in config.execution_env.items()
    ):
        raise ValueError("execution_env must map valid environment names to strings")
    _validate_choice("planner_mode", config.planner_mode, {"heuristic", "llm"})
    _validate_choice(
        "memory_extractor_mode", config.memory_extractor_mode, {"rule_based", "llm"}
    )
    _validate_choice(
        "memory_semantic_merge_mode",
        config.memory_semantic_merge_mode,
        {"disabled", "observe", "apply"},
    )
    _validate_choice(
        "long_term_memory_retrieval_mode",
        config.long_term_memory_retrieval_mode,
        {"keyword", "semantic", "hybrid"},
    )
    _validate_choice(
        "context_compressor_mode",
        config.context_compressor_mode,
        {"disabled", "rule_based", "llm"},
    )
    _validate_choice("action_policy_mode", config.action_policy_mode, {"rl", "llm"})
    _validate_choice("final_reporter_mode", config.final_reporter_mode, {"rule_based", "llm"})
    _validate_choice("completion_judge_mode", config.completion_judge_mode, {"auto", "rule_based", "llm"})
    for field_name in (
        "task_analyzer_mode",
        "observer_mode",
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
        "memory_extractor_llm_config",
        "memory_relation_llm_config",
        "code_context_query_llm_config",
        "code_context_rerank_llm_config",
        "skill_selector_llm_config",
        "final_reporter_llm_config",
        "completion_judge_llm_config",
    ):
        _validate_llm_config(field_name, getattr(config, field_name))
        if field_name != "llm_config":
            _validate_llm_config(field_name, resolve_llm_config(config.llm_config, getattr(config, field_name)))

    for field_name in (
        "max_loops",
        "context_max_tokens",
        "context_recent_items",
        "context_min_new_tokens",
        "observer_store_limit",
        "session_memory_max_turns",
        "session_memory_max_chars",
        "session_file_cache_max_files",
        "session_file_cache_max_spans",
        "session_file_cache_max_bytes",
        "memory_consolidation_max_candidates",
        "memory_semantic_top_k",
        "memory_catalog_busy_timeout_ms",
        "memory_keyword_candidate_multiplier",
        "long_term_memory_retrieval_limit",
        "long_term_memory_retrieval_max_chars",
        "long_term_memory_retrieval_max_refreshes",
        "long_term_memory_retrieval_max_queries",
        "task_archive_max_text_chars",
        "task_archive_max_source_file_bytes",
        "task_archive_max_source_total_bytes",
        "code_context_query_limit",
        "code_context_selected_limit",
        "code_context_rerank_candidate_limit",
        "skill_selected_limit",
        "rl_replay_max_size",
        "rl_train_batch_size",
        "editing_max_files",
        "editing_max_changed_lines",
        "editing_max_file_bytes",
    ):
        if int(getattr(config, field_name)) <= 0:
            raise ValueError(f"{field_name} must be greater than 0")
    if not 0.0 <= float(config.editing_confidence_threshold) <= 1.0:
        raise ValueError("editing_confidence_threshold must be between 0 and 1")
    if not 0.0 <= float(config.observer_write_threshold) <= 1.0:
        raise ValueError("observer_write_threshold must be between 0 and 1")
    if int(config.memory_consolidation_max_candidates) > 100:
        raise ValueError("memory_consolidation_max_candidates must not exceed 100")
    if int(config.memory_semantic_top_k) > 20:
        raise ValueError("memory_semantic_top_k must not exceed 20")
    for field_name in (
        "memory_semantic_min_similarity",
        "memory_semantic_relation_min_confidence",
    ):
        if not 0.0 <= float(getattr(config, field_name)) <= 1.0:
            raise ValueError(f"{field_name} must be between 0 and 1")
    _validate_embedding_config(config.memory_embedding_config)
    if int(config.memory_keyword_candidate_multiplier) > 100:
        raise ValueError("memory_keyword_candidate_multiplier must not exceed 100")
    validate_memory_retrieval_config(config)

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
        "long_term_memory.extractor_mode",
        config.memory_extractor_mode == "llm",
        resolve_llm_config(config.llm_config, config.memory_extractor_llm_config),
    )
    semantic_enabled = config.memory_semantic_merge_mode in {"observe", "apply"}
    if semantic_enabled and str(config.memory_embedding_config.provider).lower() in {
        "", "disabled", "none"
    }:
        raise ValueError("semantic memory merge requires an enabled embedding provider")
    _require_llm_config(
        "long_term_memory.semantic_merge_mode",
        semantic_enabled,
        resolve_llm_config(config.llm_config, config.memory_relation_llm_config),
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
    _require_llm_config(
        "modes.completion_judge",
        config.completion_judge_mode == "llm",
        resolve_llm_config(config.llm_config, config.completion_judge_llm_config),
    )


def validate_memory_retrieval_config(config: DebugAgentConfig) -> None:
    """Validate retrieval without requiring unrelated agent LLM capabilities."""
    _validate_choice("long_term_memory_retrieval_mode", config.long_term_memory_retrieval_mode,
                     {"keyword", "semantic", "hybrid"})
    _validate_embedding_config(config.memory_embedding_config)
    for name, low, high in (
        ("semantic_candidates", 1, 100), ("semantic_min_score", 0, 1),
        ("semantic_timeout", 0.01, 60), ("query_cache_size", 1, 10000),
        ("rrf_k", 1, 1000), ("keyword_weight", 0.001, 100), ("semantic_weight", 0.001, 100),
        ("scope_matched_weight", 1, 2), ("scope_unknown_weight", 0.01, 1),
        ("scope_unknown_limit", 0, 100),
        ("retrieval_limit", 1, 100), ("retrieval_max_queries", 1, 8),
        ("retrieval_max_refreshes", 1, 10), ("retrieval_min_score", 0, 1),
        ("retrieval_max_chars", 1000, 10000000),
    ):
        value = float(getattr(config, "long_term_memory_" + name))
        if not low <= value <= high:
            raise ValueError(f"long_term_memory_{name} must be between {low} and {high}")
        if name in {"semantic_candidates", "query_cache_size", "rrf_k", "scope_unknown_limit", "retrieval_limit",
                    "retrieval_max_queries", "retrieval_max_refreshes", "retrieval_max_chars"} and not value.is_integer():
            raise ValueError(f"long_term_memory_{name} must be an integer")
    limits = config.long_term_memory_retrieval_type_limits
    if not isinstance(limits, dict):
        raise ValueError("long_term_memory_retrieval_type_limits must be an object")
    for memory_type, limit in limits.items():
        if memory_type not in {"preference", "semantic", "procedural", "anti_pattern", "episodic"}:
            raise ValueError(f"unsupported retrieval memory type limit: {memory_type}")
        if int(limit) < 0:
            raise ValueError(f"retrieval type limit for {memory_type} must not be negative")


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
        "session_memory_path",
        "task_archive_path",
        "long_term_memory_path",
        "memory_catalog_path",
        "memory_semantic_index_path",
        "consolidation_run_path",
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
        response_format_mode=resolve_str("response_format_mode"),
        structured_fallback=(
            override.structured_fallback
            if override.structured_fallback != default.structured_fallback
            else base.structured_fallback
        ),
        max_retries=override.max_retries if override.max_retries != default.max_retries else base.max_retries,
        max_completion_tokens=override.max_completion_tokens if override.max_completion_tokens is not None else base.max_completion_tokens,
        reasoning_effort=override.reasoning_effort if override.reasoning_effort is not None else base.reasoning_effort,
        extra_body={**base.extra_body, **override.extra_body},
    )


def _embedding_config_from_dict(
    data: dict[str, Any], base: EmbeddingConfig | None = None
) -> EmbeddingConfig:
    current = base or EmbeddingConfig()
    values = {
        field.name: data.get(field.name, getattr(current, field.name))
        for field in fields(EmbeddingConfig)
    }
    return EmbeddingConfig(**values)


def _validate_embedding_config(config: EmbeddingConfig) -> None:
    provider = str(config.provider or "").strip().lower()
    if provider not in {"", "disabled", "none", "openai", "openai_compatible", "openai-compatible"}:
        raise ValueError(f"unsupported embedding provider: {config.provider}")
    if provider not in {"", "disabled", "none"} and not str(config.model or "").strip():
        raise ValueError("embedding model is required when embedding is enabled")
    if int(config.dimensions) < 0:
        raise ValueError("embedding dimensions must not be negative")
    if int(config.batch_size) <= 0:
        raise ValueError("embedding batch_size must be greater than 0")


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


def validate_llm_generation(value: LLMConfig) -> None:
    if value.max_completion_tokens is not None and (
        type(value.max_completion_tokens) is not int or value.max_completion_tokens <= 0
    ):
        raise ValueError("max_completion_tokens must be a positive integer or null")
    if value.reasoning_effort is not None and (
        not isinstance(value.reasoning_effort, str) or not value.reasoning_effort.strip()
    ):
        raise ValueError("reasoning_effort must be a nonempty string or null")
    if not isinstance(value.extra_body, dict):
        raise ValueError("extra_body must be an object")
    reserved = {"model", "messages", "temperature", "response_format", "stream", "stream_options",
                "max_tokens", "max_completion_tokens", "reasoning_effort", "n", "tools", "tool_choice"}
    if reserved.intersection(value.extra_body):
        raise ValueError(f"extra_body cannot override request fields: {sorted(reserved.intersection(value.extra_body))}")
    budget = value.extra_body.get("thinking_budget")
    if "thinking_budget" in value.extra_body and (type(budget) is not int or budget < 0):
        raise ValueError("thinking_budget must be a nonnegative integer")
    if value.model.lower().startswith("qwen3.8") and value.reasoning_effort is not None and budget is not None:
        raise ValueError("Qwen3.8 does not support reasoning_effort and thinking_budget together")


def _validate_llm_config(field_name: str, value: LLMConfig) -> None:
    validate_llm_generation(value)
    _validate_choice(
        f"{field_name}.response_format_mode",
        value.response_format_mode,
        {"auto", "native", "json"},
    )
    if type(value.max_retries) is not int or value.max_retries < 0:
        raise ValueError(f"{field_name}.max_retries must be a nonnegative integer")
    if not isinstance(value.structured_fallback, bool):
        raise ValueError(f"{field_name}.structured_fallback must be a boolean")
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
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".rb": "ruby",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".swift": "swift",
        ".scala": "scala",
        ".sh": "shell",
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
