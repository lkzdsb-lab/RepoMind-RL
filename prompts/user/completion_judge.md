Return JSON with keys: decision, reason, questions, suggested_next_action, confidence.
confidence must be a number from 0.0 to 1.0.

title={{ title }}
description={{ description }}
status={{ status }}
current_step={{ current_step }}
error={{ error }}
project_profile={{ project_profile }}
verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
task_analysis={{ task_analysis }}
plan={{ plan }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
tool_calls={{ tool_calls }}
test_results={{ test_results }}
has_patch={{ has_patch }}
patch_summary={{ patch_summary }}
llm_observations={{ llm_observations }}
user_inputs={{ user_inputs }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
fallback_judgement={{ fallback_judgement }}

Rules:
- If read_files and observations answer the task, choose complete.
- If the task is a code review or static analysis request and enough relevant code has been read, choose complete even when no patch exists.
- If the answer depends on the user's intended behavior, missing business rule, target environment, or a choice among valid interpretations, choose needs_user_input and ask 1-3 concrete questions.
- Use project_profile before asking the user for language. Ask only when project_profile and repository evidence are inconclusive or conflict with the user's wording.
- If available search/read tools could not identify the target file after trying focused searches, choose needs_user_input instead of complete.
- If the missing evidence is still available through search_code_context, read_file, run_tests, or git_diff, choose continue and set suggested_next_action to the most useful tool name.
- If verification_required is false, do not choose continue just to run tests.
