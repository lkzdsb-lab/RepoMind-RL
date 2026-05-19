# Runtime Config Reference

`config.json` is loaded automatically on startup. CLI flags are still supported,
but only explicitly provided flags override the file.

## Runtime Behavior

Use `review_only` when you want the agent to read and analyze code without
running the configured verification command.

```json
{
  "review_only": true
}
```

The same switch is available on the CLI:

```bash
python3 main.py "review the payment flow" --repo /path/to/repo --review-only
```

When `review_only` is enabled, the agent skips `run_tests`; `verify_command`
can be omitted and the default `pytest` will not be executed.

## LLM Rules

`llm` at the root is only the default LLM config. It does not mean every LLM
module is enabled.

Each module is enabled by `modes`. If a mode is `disabled`, that module will not
call the LLM even when the root `llm` provider/model/key are configured.

Valid provider values:

- `disabled`: no LLM client.
- `none`: alias for disabled.
- `openai`: official OpenAI SDK client.
- `openai_compatible`: OpenAI-compatible endpoint.
- `openai-compatible`: alias for `openai_compatible`.
- `enable`: legacy alias for OpenAI-compatible endpoint.

The actual key is read from `.env` through `api_key_env`; do not put key values
in `config.json`.

```json
{
  "env_file": ".env",
  "llm": {
    "provider": "openai_compatible",
    "model": "qwen-plus",
    "api_base": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "api_key_env": "LLM_API_KEY"
  }
}
```

## Mode Values

| Config key | Valid values | Meaning |
| --- | --- | --- |
| `modes.planner` | `heuristic`, `llm` | Initial plan generation. |
| `modes.context_compressor` | `disabled`, `rule_based`, `llm` | Context compression implementation. |
| `modes.action_policy` | `heuristic`, `rl`, `llm` | Next-action selection. |
| `modes.task_analyzer` | `disabled`, `llm` | Task understanding. |
| `modes.observer` | `disabled`, `llm` | Tool result observation synthesis. |
| `modes.memory_query_planner` | `disabled`, `llm` | Multi-query memory retrieval planning. |
| `modes.memory_reranker` | `disabled`, `llm` | Memory candidate reranking. |
| `modes.code_context_query_planner` | `disabled`, `llm` | Multi-query codebase-context search planning. |
| `modes.code_context_reranker` | `disabled`, `llm` | Codebase-context candidate reranking. |
| `modes.skill_selector` | `disabled`, `llm` | Registered skill selection. |
| `modes.final_reporter` | `rule_based`, `llm` | Final user-facing run summary. |

## Per-Module LLM Overrides

Per-module LLM config inherits from root `llm`. Override only the fields that
need to differ.

```json
{
  "llm": {
    "provider": "openai_compatible",
    "model": "fast-model",
    "api_base": "https://host/v1",
    "api_key_env": "LLM_API_KEY",
    "action": {
      "model": "strong-action-model"
    },
    "memory_rerank": {
      "model": "cheap-rerank-model",
      "temperature": 0.0
    },
    "code_context_rerank": {
      "model": "strong-code-model"
    }
  }
}
```

Supported override blocks:

- `llm.context_compressor`
- `llm.plan` or `llm.planner`
- `llm.action` or `llm.action_policy`
- `llm.task_analysis`
- `llm.observer`
- `llm.memory_query`
- `llm.memory_rerank`
- `llm.code_context_query`
- `llm.code_context_rerank`
- `llm.skill_selector`
- `llm.final_reporter`

## Example: Enable Only Code Context LLM

This enables only code context query planning and reranking. Other LLM modules
stay disabled.

```json
{
  "llm": {
    "provider": "openai_compatible",
    "model": "qwen-plus",
    "api_base": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "api_key_env": "LLM_API_KEY",
    "code_context_query": {
      "model": "qwen-turbo"
    },
    "code_context_rerank": {
      "model": "qwen-plus"
    }
  },
  "modes": {
    "planner": "heuristic",
    "context_compressor": "rule_based",
    "action_policy": "heuristic",
    "task_analyzer": "disabled",
    "observer": "disabled",
    "memory_query_planner": "disabled",
    "memory_reranker": "disabled",
    "code_context_query_planner": "llm",
    "code_context_reranker": "llm",
    "skill_selector": "disabled"
  }
}
```

## Example: Enable Skill Selection Only

```json
{
  "llm": {
    "provider": "openai_compatible",
    "model": "qwen-plus",
    "api_base": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "api_key_env": "LLM_API_KEY"
  },
  "modes": {
    "skill_selector": "llm"
  }
}
```

## Files

- `config.schema.json`: machine-readable schema for editor validation.
- `config.example.json`: full template.
- `config.json`: local runtime config, ignored by git.
- `.env`: local secrets, ignored by git.

## Project Isolation

Runtime artifacts are resolved under the target `repo_path`. If you run:

```bash
python3 main.py "debug task" --repo /path/to/project-a
python3 main.py "debug task" --repo /path/to/project-b
```

the agent writes separate state:

```text
/path/to/project-a/.repomind/
  logs/agent.log
  traces/*.json
  memory_mid.jsonl
  memory_long.jsonl
  skills/
  codebase_context/index.json
  rl/q_table.json
  rl/replay.jsonl

/path/to/project-b/.repomind/
  ...
```

Relative runtime paths such as `.repomind/traces` and `.repomind/logs/agent.log`
are interpreted relative to the target repo, not relative to the Lee-Agent
source checkout.
