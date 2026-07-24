# Task Context

title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}

# Query Rules

Generate up to {{ max_queries }} memory retrieval queries.
Cover episodic, semantic, procedural, anti-pattern, and skill memory only when useful to the current task.

# Output Schema

Return JSON with keys: queries, rationale, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
