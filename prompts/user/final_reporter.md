Return JSON with keys: summary, work_done, candidate_files, test_results, has_patch, patch_status, next_steps.
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
plan={{ plan }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
test_results={{ test_results }}
has_patch={{ has_patch }}
patch_summary={{ patch_summary }}
tool_calls={{ tool_calls }}
llm_observations={{ llm_observations }}
user_inputs={{ user_inputs }}
fallback_report={{ fallback_report }}
