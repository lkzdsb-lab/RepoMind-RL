# Current State

title={{ title }}
description={{ description }}
current_step={{ current_step }}
status={{ status }}
error={{ error }}

# Existing Digest

{{ fallback_digest }}

# Context To Compress

{{ item_text }}

# Compression Rules

Preserve the current goal, hard constraints, decisions, unresolved work, stable observations, significant tool results, code changes, and memory references. Remove repetition and low-value narration. Do not invent facts.

# Output Schema

Return only a JSON object with these keys: summary, current_goal, constraints, decisions, open_tasks, completed_tasks, key_observations, tool_results, code_changes, memory_refs, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
