Generate up to {{ max_queries }} codebase context search queries.
Start from the default query if it is useful, but expand it into focused route, handler, service, repository, model, interface, or test queries when the task suggests them.
Return JSON with keys: queries, rationale, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
Do not assume a programming language from generic words like "main", "app", "server", or "index".
If the user asks about a main function/file without a language, generate language-neutral queries first, such as "main", "main function", "entrypoint", and likely filenames already implied by repository evidence.
Use project_profile as repository language evidence. Generate language-specific queries only when the user, project_profile, task_analysis, skill_context, or repository candidates provide concrete language evidence.

default_query={{ default_query }}
title={{ title }}
description={{ description }}
project_profile={{ project_profile }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
memory_context={{ memory_context }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
