"""
    file name: config.py
    Author: kunze.li
"""
from __future__ import annotations

from dataclasses import dataclass


class FileConfig(object):
    """
        file 相关配置
    """
    DEBUG = False
    TESTING = False
    MAX_READ_AMOUNT = 200


class GraphConfig(object):
    """
        graph 相关配置
    """
    DEBUG = False
    TESTING = False
    MAX_LOOP_COUNT = 10


@dataclass
class DebugAgentConfig:
    """
        agent 相关配置
    """
    # 仓库路径
    repo_path: str
    verify_command: str = "pytest"
    max_loops: int = 8

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
    context_max_tokens: int = 32000
    context_compression_threshold: float = 0.75
    context_recent_items: int = 8

    # llm 配置
    context_llm_provider: str = "disabled"
    context_llm_model: str = ""
    context_llm_api_base: str = ""
    context_llm_api_key_env: str = "LLM_API_KEY"
    context_llm_timeout: int = 60
    context_llm_temperature: float = 0.0
    context_llm_max_output_chars: int = 12000

    # 代码索引库
    code_context_index_path: str = ".repomind/codebase_context/index.json"

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