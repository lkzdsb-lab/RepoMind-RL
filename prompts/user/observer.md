Return JSON with keys: latest_tool, status, summary, new_findings, hypotheses, invalidated_hypotheses, facts, risks, next_actions, memory_candidates, missing_context, next_search_terms, confidence, user_update.
status must be one of ok, error, inconclusive, complete.
status=complete is only an observation for later review; final task termination is decided by completion_judgement.
Use short list fields. If uncertain, use empty lists instead of guessing.
facts should capture stable information useful for future actions.
memory_candidates should expose potential memory entries only; do not assume they will be written.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
Focus on the latest delta first. Do not re-open the whole task unless observation_mode is full or the latest delta clearly invalidates the current execution plan.
If the latest action only confirms an already-known fact, return minimal output and avoid inventing new next_actions.
When current_execution is present, keep the observation scoped to that execution item. Do not redirect unrelated files or rewrite the overall plan.

observation_mode={{ observation_mode }}
title={{ title }}
description={{ description }}
task_analysis={{ task_analysis }}
current_step={{ current_step }}
plan={{ plan }}
candidate_files={{ candidate_files }}
test_results_tail={{ test_results_tail }}
patch_summary={{ patch_summary }}
memory_context={{ memory_context }}
compressed_context={{ compressed_context }}
latest_context_event={{ latest_context_event }}
latest_tool_call={{ latest_tool_call }}
observation_delta={{ observation_delta }}
current_execution={{ current_execution }}
recent_observations={{ recent_observations }}
