Select up to {{ selected_limit }} skills that should influence this run.
Return JSON with keys: selected, rationale, user_update.
Each selected item must have skill_name, relevance, reason.
Select none if no registered skill is relevant.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.

title={{ title }}
description={{ description }}
project_profile={{ project_profile }}
task_analysis={{ task_analysis }}
current_step={{ current_step }}
memory_context={{ memory_context }}
code_context={{ code_context }}
available_skills={{ available_skills }}
