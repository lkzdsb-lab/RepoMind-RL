# Task And Final State

title={{ title }}
description={{ description }}
status={{ status }}
error={{ error }}

# Verification Evidence

verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verification_stale={{ verification_stale }}
verification_commands={{ verification_commands }}
command_results={{ command_results }}
test_results={{ test_results }}

# Plan And Work Evidence

plan_mode={{ plan_mode }}
plan_mode_approved={{ plan_mode_approved }}
technical_plan={{ technical_plan }}
plan_mode_evaluation={{ plan_mode_evaluation }}
plan={{ plan }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
edit_results={{ edit_results }}
change_summaries={{ change_summaries }}
has_patch={{ has_patch }}
patch_summary={{ patch_summary }}
tool_calls={{ tool_calls }}
llm_observations={{ llm_observations }}
user_inputs={{ user_inputs }}

# Fallback Report

{{ fallback_report }}

# Reporting Rules

Report only outcomes supported by the supplied evidence.
Explain what was done, which relevant files were identified, verification results, whether a patch exists, and any necessary next steps.
Do not claim success when status, error, stale verification, or command evidence indicates otherwise.
Keep the report concise and user-facing.

# Output Schema

Return JSON with keys: summary, work_done, candidate_files, test_results, has_patch, patch_status, next_steps, user_update.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
