# Memory Consolidation

Use this skill when writing, promoting, or consolidating task memory.

## Memory Types

- episodic: concrete task experience, candidate files, verification result, and patch summary.
- semantic: stable facts extracted from successful or high-confidence tasks.
- procedural: reusable debugging or implementation procedures.
- anti_pattern: failed paths, incorrect assumptions, or actions that should not be repeated.

## Promotion Rules

1. Promote only when there is evidence: test output, diff, traced files, or explicit failure.
2. High reward successful tasks can become semantic memory.
3. Reusable multi-step procedures can become procedural memory.
4. Failures with useful lessons should become anti-pattern memory.
5. Only consolidate to skill when the memory is reusable across tasks, not just one-off context.

## Skill Consolidation Format

When adding memory to a skill resource, include:

- memory id
- task trigger
- memory type
- reward signal
- evidence
- reusable lesson or procedure

## Avoid

- Do not promote noisy observations without verification evidence.
- Do not store secrets, credentials, or private user data.
- Do not turn every successful task into a skill. Skills should capture repeated workflows.
