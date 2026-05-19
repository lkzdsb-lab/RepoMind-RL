Generate up to {{ max_queries }} memory retrieval queries.
Queries should cover episodic, semantic, procedural, anti-pattern, and skill memory when useful.
Return JSON with keys: queries, rationale.

title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
