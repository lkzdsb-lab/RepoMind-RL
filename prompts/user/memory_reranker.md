# Task And Candidates

title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
queries={{ queries }}
candidates={{ candidates }}

# Selection Rules

Select up to {{ selected_limit }} memories that are directly useful to the current task.
Use only memory_id values from candidates, and select none when none are relevant.

# Output Schema

Return JSON with keys: selected, user_update. Each selected item must contain memory_id, relevance, and reason.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
