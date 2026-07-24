# Task And Evidence

title={{ title }}
description={{ description }}
project_profile={{ project_profile }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
memory_context={{ memory_context }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
default_query={{ default_query }}

# Query Rules

Generate up to {{ max_queries }} codebase context search queries.
Start from the default query when useful, then expand into focused route, handler, service, repository, model, interface, or test queries supported by the task.
Do not infer a programming language from generic words such as "main", "app", "server", or "index".
For an unspecified main function or file, begin with language-neutral queries and filenames supported by repository evidence.
Use project_profile as the primary language evidence. Generate language-specific queries only when the user, project_profile, task_analysis, skill_context, or repository candidates provide concrete support.

# Output Schema

Return JSON with keys: queries, rationale, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
