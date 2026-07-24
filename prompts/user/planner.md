# Task And Context

title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
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
Preserve useful steps from default_plan, but remove generic or irrelevant work.

# Output Schema

Return JSON with keys: plan, user_update.
plan must be an ordered list of concrete steps.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
