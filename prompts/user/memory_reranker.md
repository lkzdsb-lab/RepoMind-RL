Select up to {{ selected_limit }} truly relevant memories from candidates.
Return JSON with keys: selected, user_update, where each selected item has memory_id, relevance, reason.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
Use only candidate memory_id values.
Select none if none are relevant.

title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
queries={{ queries }}
candidates={{ candidates }}
