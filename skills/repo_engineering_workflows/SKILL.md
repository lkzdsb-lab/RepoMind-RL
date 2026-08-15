---
name: repo_engineering_workflows
description: Use this skill for repository code work: code review, code editing, structured codebase search, test failure triage, and Go backend debugging. Trigger when the task asks to review correctness, find bugs or risks, inspect implementation quality, modify or patch code, implement a change, use search_code_context, locate routes/symbols/functions/models/tests, diagnose failing pytest or go test runs, or debug Go backend flow through route, handler, service, repository, and model layers. Read only the relevant reference file for the current task instead of loading every workflow.
---

# Repo Engineering Workflows

Use this skill as the single entrypoint for normal repository engineering tasks.

## Workflow Selection

- For code review, read [references/code_review.md](references/code_review.md).
- For code edits and patch planning, read [references/code_edit.md](references/code_edit.md).
- For structured codebase index usage, read [references/codebase_context.md](references/codebase_context.md).
- For failing test diagnosis, read [references/test_failure_triage.md](references/test_failure_triage.md).
- For Go backend flow debugging, read [references/go_backend_debug.md](references/go_backend_debug.md).

## Rules

1. Load only the workflow file that matches the current task.
2. Do not inline all workflow details into the active context.
3. Reuse the repository's current tools and guardrails rather than inventing a parallel process.
