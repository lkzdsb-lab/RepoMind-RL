title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
current_step={{ current_step }}
candidate_files={{ candidate_files }}
memory_context={{ memory_context }}
compressed_context={{ compressed_context }}
verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verify_command={{ verify_command }}
default_plan={{ default_plan }}

Prefer task-specific `search_code_context` over repository-wide file listing.
Do not add a file-listing step unless the user explicitly asks for the full file tree.
Return JSON like {"plan": ["...", "..."]}.
