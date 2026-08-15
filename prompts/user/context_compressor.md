Return only a JSON object with these keys: summary, current_goal, constraints, decisions, open_tasks, completed_tasks, key_observations, tool_results, code_changes, memory_refs, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.

Current fallback digest:
{{ fallback_digest }}

Current state hints:
title={{ title }}
description={{ description }}
current_step={{ current_step }}
status={{ status }}
error={{ error }}

Compress these context items:
{{ item_text }}
