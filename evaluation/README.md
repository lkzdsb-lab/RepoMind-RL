# Conversation Evaluation

Run the fixed Go debugging conversation against an isolated workspace:

```powershell
.\.venv\Scripts\python.exe -m evaluation --case evaluation/cases/go_web_debug.json --config config.json
```

The evaluator records every conversation turn, computes repository changes
without relying on Git, and runs the configured verification command itself.
Artifacts are written under `.repomind/evaluation/<case_id>/<run_id>/`.
An explicit relative `--config` path is resolved from the invocation directory.
When `--config` is omitted, both the CLI and evaluator use the same stable lookup:
`REPOMIND_CONFIG`, then the RepoMind runtime root, then the user config directory.
The resolved absolute path is validated before a run directory is created and is
passed unchanged to the isolated worker, so the target workspace never controls
which Agent configuration is loaded.

The Go case now uses the parent `../agent test` fixture and the centralized Bug
baseline at `evaluation/baselines/go_web_debug.json`. The easy/medium/hard cases
select subsets. Local HTTP fixture acceptance (without an Agent/LLM), explicit
race exclusion and D-drive cache locations are documented in
`docs/debug-benchmark-v2.md`.

## Long-term memory Recall@K

Use `--case evaluation/cases/memory_recall.json` to compare keyword, semantic and
hybrid retrieval on an isolated memory corpus. Configure an embedding provider
and export its API key environment variable for semantic/hybrid runs. The case
contains fixed gold memory IDs and query-time state, not generated LLM labels.
See `docs/long-term-memory-hybrid-retrieval.md` for configuration and limitations.
Memory evaluation retains its workspace for inspection. For this case type,
`--timeout` overrides the per-retrieval semantic deadline, not an agent timeout.

`evaluation/cases/memory_scope.json` provides a keyword-only scope regression
fixture without an embedding dependency. `forbidden_memory_ids` checks all
injected IDs, while `expected_scope_status` checks scope decisions independently
of recall. See `docs/long-term-memory-scope-retrieval.md` for the three-state policy.
