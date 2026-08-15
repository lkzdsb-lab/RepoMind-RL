Return JSON with keys: summary, work_done, candidate_files, test_results, has_patch, patch_status, next_steps, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
The report must explain:
- what was done
- which candidate files were found
- test results
- whether there is a patch
- recommended next steps

title={{ title }}
description={{ description }}
status={{ status }}
error={{ error }}
verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verification_stale={{ verification_stale }}
verification_commands={{ verification_commands }}
command_results={{ command_results }}
plan_mode={{ plan_mode }}
plan_mode_approved={{ plan_mode_approved }}
debug_technical_plan={{ debug_technical_plan }}
plan_mode_evaluation={{ plan_mode_evaluation }}
plan={{ plan }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
test_results={{ test_results }}
edit_results={{ edit_results }}
change_summaries={{ change_summaries }}
has_patch={{ has_patch }}
patch_summary={{ patch_summary }}
tool_calls={{ tool_calls }}
llm_observations={{ llm_observations }}
user_inputs={{ user_inputs }}
fallback_report={{ fallback_report }}
