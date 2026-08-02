# Task And Context

title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
validated_file_cache={{ validated_file_cache }}
memory_context={{ memory_context }}
compressed_context={{ compressed_context }}

# Verification Context

verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verification_capabilities={{ verification_capabilities }}

# Baseline Plan

{{ default_plan }}

# Planning Rules

Create a concise task-specific execution plan.
Prefer focused search_code_context over repository-wide file listing.
Do not add a file-listing step unless the user explicitly asks for the complete tree.
When verification is required, choose commands only from verification_capabilities; do not rely on configured verification commands.
When the user describes a concrete reproducible behavior, include a bounded smoke-verification step for that same functional path when it can be executed safely with verification_capabilities. Mark its run_shell_command as verification_kind="smoke" because it requires explicit user approval. Do not plan unsafe service orchestration or external side effects merely to satisfy this preference.
Preserve useful steps from default_plan, but remove generic or irrelevant work.
Do not plan another read for a file when validated_file_cache already covers the
source ranges needed by the current task. Plan a focused read only when the
cache is missing, stale, or does not cover the required range.

# Output Schema

Return exactly one JSON object in this shape:

```json
{
  "plan": [
    "First concrete repository action",
    "Second concrete repository action"
  ],
  "user_update": "short progress message or empty string"
}
```

plan is required and must contain 1 to 8 non-empty, ordered, task-specific steps.
Never return an empty plan. Do not reveal chain-of-thought.
