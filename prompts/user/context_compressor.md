# Current State

title={{ title }}
description={{ description }}
current_step={{ current_step }}
status={{ status }}
error={{ error }}

# Existing Digest

{{ previous_digest }}

# Context To Compress

{{ item_text }}

# Compression Rules

Preserve the current goal, hard constraints, decisions, unresolved work, stable observations, significant tool results, code changes, and memory references. Remove repetition and low-value narration. Do not invent facts.
Merge the existing digest with the NEW distilled events below; preserve previous conclusions unless new evidence supersedes them. Aim for at most {{ target_tokens }} tokens. Reading source is not proof it was analyzed. Omitted excerpts are not unread source; distinguish source_truncated from excerpt_truncated. A failing reproduction can establish a bug in a read-only audit and does not require a code fix.

# Output Schema

Return only a JSON object with these keys: summary, current_goal, constraints, decisions, open_tasks, completed_tasks, key_observations, tool_results, code_changes, memory_refs, user_update.
summary and current_goal are strings. constraints, decisions, open_tasks, completed_tasks, key_observations, code_changes and memory_refs are arrays of strings (never objects). tool_results is an array of objects with string fields name, status, summary.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
