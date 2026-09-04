# Runtime Config Reference

`config.json` is loaded automatically on startup. If the configured file does
not exist, the CLI creates a default template first and then loads it. CLI flags
are still supported, but only explicitly provided flags override the file.

## LLM Rules

`llm` at the root is only the default LLM config. It does not mean every LLM
module is enabled.

Verification commands are not configured in `config.json`. The agent derives
verification capabilities from the project, lets the LLM choose an allowed
command, and validates the command through the runtime guard before execution.

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
| `modes.code_context_query_planner` | `disabled`, `llm` | Multi-query codebase-context search planning. |
| `modes.code_context_reranker` | `disabled`, `llm` | Codebase-context candidate reranking. |
| `modes.skill_selector` | `disabled`, `llm` | Registered skill selection. |
| `modes.final_reporter` | `rule_based`, `llm` | Final user-facing run summary. |
| `modes.completion_judge` | `auto`, `rule_based`, `llm` | Finish-time judgement; `auto` uses LLM when configured and can pause with user questions when information is missing. |

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
- `llm.code_context_query`
- `llm.code_context_rerank`
- `llm.skill_selector`
- `llm.final_reporter`
- `llm.completion_judge`

## Session Memory

Online memory is session-scoped and stored in SQLite. It is prepared before task
analysis and committed only after the final report is available.

```json
{
  "memory": {
    "path": ".repomind/session_memory.db",
    "max_turns": 6,
    "max_chars": 12000
  }
}
```

- `memory.path` is resolved relative to the target repository.
- `memory.max_turns` limits active recent-turn and short-lived memory context.
- `memory.max_chars` bounds the rendered session context passed downstream.
- `memory.file_cache_max_files` limits persisted file containers per session.
- `memory.file_cache_max_spans` limits cached source ranges across those files.
- `memory.file_cache_max_bytes` limits total cached source bytes; range SLRU evicts normal spans before protected spans.
- Long-term promotion and skill consolidation do not run in the online agent.

## Task Archive

Completed and failed terminal tasks are written to an immutable, replayable
archive before session memory is committed. This archive is the evidence source
for later offline long-term-memory consolidation.

```json
{
  "archive": {
    "enabled": true,
    "path": ".repomind/archive/tasks",
    "max_text_chars": 200000,
    "max_source_file_bytes": 200000,
    "max_source_total_bytes": 2000000
  }
}
```

- `archive.path` is resolved relative to the target repository.
- Text and source limits bound archive growth while preserving valid artifacts.
- Sensitive credential files and recognized secrets are omitted or redacted.
- Archive failure is recorded in AgentState and does not change task success.
- Existing valid task archives are immutable and reused idempotently.

The complete Phase 0/1 contract is documented in
[`long-term-memory-phase-0-1.md`](long-term-memory-phase-0-1.md).

## Long-term Markdown Memory

Canonical long-term memory documents are stored separately from online session
memory and task archives:

```json
{
  "long_term_memory": {
    "document_path": ".repomind/memory",
    "catalog_path": ".repomind/memory/catalog.sqlite3",
    "catalog_busy_timeout_ms": 5000,
    "keyword_candidate_multiplier": 8,
    "retrieval_enabled": true,
    "retrieval_mode": "keyword",
    "retrieval_limit": 8,
    "retrieval_max_chars": 12000,
    "retrieval_min_score": 0.1,
    "retrieval_include_draft": true,
    "retrieval_max_refreshes": 3,
    "retrieval_max_queries": 4,
    "retrieval_type_limits": {
      "preference": 2,
      "semantic": 4,
      "procedural": 2,
      "anti_pattern": 2,
      "episodic": 2
    },
    "consolidation_path": ".repomind/consolidation/runs",
    "pipeline_version": "consolidation-v2",
    "extractor_mode": "rule_based",
    "max_candidates": 24
  }
}
```

Document, Catalog, and consolidation paths are resolved relative to the target
repository. The SQLite Catalog is a rebuildable structured and FTS5 projection;
Markdown remains authoritative. `catalog_busy_timeout_ms` controls writer lock
waiting and `keyword_candidate_multiplier` bounds the FTS candidate pool before
deterministic application ranking.

Online retrieval is read-only. `retrieval_limit` bounds the number of selected
documents, while `retrieval_max_chars` bounds the rendered prompt context.
`retrieval_min_score`, status and scope checks, and per-type limits prevent a
single memory class from dominating the context. The agent performs at most
`retrieval_max_refreshes` distinct retrievals per task and reuses the result
while the query fingerprint is unchanged. Set `retrieval_enabled` to `false` to
disable injection without disabling consolidation or Catalog maintenance.

The default rule-based extractor is fully offline. Set `extractor_mode` to `llm`
and configure `llm.memory_extractor` (or the shared `llm` settings) to use semantic
extraction; deterministic evidence and promotion policy still owns the final
status and confidence.

The Phase 2 format and storage guarantees are documented in
[`long-term-memory-phase-2.md`](long-term-memory-phase-2.md). The offline archive
to document pipeline is documented in
[`long-term-memory-phase-3.md`](long-term-memory-phase-3.md). Catalog consistency,
rebuild, and keyword retrieval are documented in
[`long-term-memory-phase-4.md`](long-term-memory-phase-4.md). Agent integration,
prompt audiences, refresh rules, and usage tracing are documented in
[`long-term-memory-phase-5.md`](long-term-memory-phase-5.md).

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

## Guarded Editing

Repository writes are disabled by default. Enable them only when the run is
allowed to modify the target repo:

```json
{
  "editing": {
    "enabled": true,
    "max_files": 5,
    "max_changed_lines": 300,
    "max_file_bytes": 200000,
    "require_read_before_write": true,
    "confidence_threshold": 0.75,
    "allow_create": false
  }
}
```

When editing is enabled and `modes.action_policy` is `llm`, the LLM can select
`apply_code_patch`, but the tool is guarded by runtime state. It can only apply
exact replacements to files read during the same run, unless creation is
explicitly enabled.

Before any code-changing action, the LLM must call `EnterPlanMode` and record a
detailed Debug/Refactor Technical Plan. While in Plan Mode, code-changing tools
are not exposed by the action space and the executor rejects bypass attempts.
The LLM can call `ExitPlanMode` only after evaluating the plan as feasible; if
uncertainty remains, the run pauses with `awaiting_user_input`.

After a patch is applied, `verification_stale` becomes `true`. The run cannot
finish, write memory, or proceed to final diff summarization until a verification
command has run through `run_shell_command` with `purpose="verification"` or
through the legacy `run_tests` tool. The generic primitives available to the LLM
are:

- `search_text`: regex or fixed-string repository search backed by `rg`/`grep`.
- `run_shell_command`: guarded command execution for diagnostics, search, build,
  and verification.
- `apply_code_patch`: guarded exact-replacement edits.
- `EnterPlanMode` / `ExitPlanMode`: planning gate around code-changing work.

If the LLM reports uncertainty or confidence below the threshold, the run pauses
with `awaiting_user_input` instead of writing files.

## Human Approval

Set `approval.require_step_approval` to force an approval gate before each
agent action is executed:

```json
{
  "approval": {
    "require_step_approval": true
  }
}
```

When this is enabled, the agent pauses after selecting the next action and
shows the action name plus compact arguments. Reply `approve`, `yes`, or
`同意` to execute that action. Any other reply is treated as user feedback,
written into the conversation context, and the agent replans before selecting
the next action.

Equivalent CLI flags:

```bash
lee-agent --repo /path/to/repo --action-policy-mode llm --enable-editing --require-step-approval
```

## Files

- `config.schema.json`: machine-readable schema for editor validation.
- `config.example.json`: full template.
- `config.json`: local runtime config, ignored by git.
- `.env`: local secrets, ignored by git.

## Project Isolation

Runtime artifacts are resolved under the target `repo_path`. If you run:

```bash
lee-agent --repo /path/to/project-a
lee-agent --repo /path/to/project-b
```

the agent writes separate state:

```text
/path/to/project-a/.repomind/
  logs/agent.log
  traces/*.json
  session_memory.db
  codebase_context/index.json
  rl/q_table.json
  rl/replay.jsonl

/path/to/project-b/.repomind/
  ...
```

Relative runtime paths such as `.repomind/traces` and `.repomind/logs/agent.log`
are interpreted relative to the target repo, not relative to the Lee-Agent
source checkout.
