# Task And Available Skills

title={{ title }}
description={{ description }}
project_profile={{ project_profile }}
task_analysis={{ task_analysis }}
current_step={{ current_step }}
memory_context={{ memory_context }}
long_term_memory_context={{ long_term_memory_context }}
code_context={{ code_context }}
available_skills={{ available_skills }}

# Selection Rules

Select up to {{ selected_limit }} skills that should materially influence this run.
Select none when no registered skill is relevant.
Long-term memory may suggest a workflow but does not prove that a skill applies
to the current repository state.
Use only skills from available_skills. Relevance must be a number between 0 and 1, not a label such as high or medium.

# Output Schema

Return JSON with keys: selected, rationale, user_update. Each selected item must contain skill_name, relevance, and reason.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
