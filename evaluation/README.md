# Conversation Evaluation

Run the fixed Go debugging conversation against an isolated workspace:

```powershell
.\.venv\Scripts\python.exe -m evaluation --case evaluation/cases/go_web_debug.json --config config.json
```

The evaluator records every conversation turn, computes repository changes
without relying on Git, and runs the configured verification command itself.
Artifacts are written under `.repomind/evaluation/<case_id>/<run_id>/`.

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
