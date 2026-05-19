Generate up to {{ max_queries }} codebase context search queries.
Start from the default query if it is useful, but expand it into focused route, handler, service, repository, model, interface, or test queries when the task suggests them.
Return JSON with keys: queries, rationale.

default_query={{ default_query }}
title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
memory_context={{ memory_context }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
