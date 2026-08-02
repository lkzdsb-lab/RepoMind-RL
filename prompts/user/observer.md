# Observation Scope

Focus on the latest delta first. Do not re-open the whole task unless observation_mode is full or the latest delta clearly invalidates the current execution plan.
If the latest action only confirms an already-known fact, return minimal output and avoid inventing new next_actions.

Use this evidence priority:
1. Current-run test, command, edit, and tool output.
2. Current-run repository reads and diffs.
3. Current-run observations grounded in those outputs.
4. Plans, candidate files, task-analysis history, and session memory.

Level 4 is search guidance only. Do not copy its claims into facts or
new_findings without independent confirmation from levels 1-3. If current
evidence shows fewer findings than history, report only the currently supported
findings and do not preserve the historical count.

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
recent_observations={{ recent_observations }}
read_file_context={{ read_file_context }}

# Output Schema

Return JSON with keys: latest_tool, status, summary, new_findings, finding_candidates, hypotheses, invalidated_hypotheses, facts, risks, next_actions, memory_candidates, missing_context, next_search_terms, confidence, user_update.
status must be one of ok, error, inconclusive, complete.
status=complete is only an observation for later review; final task termination is decided by completion_judgement.
Use short list fields. If uncertain, use empty lists instead of guessing.
facts should capture stable information useful for future actions.
Every fact and new_finding must be traceable to evidence from the current run.
When current source or tool evidence supports a concrete defect or reportable risk, add it to finding_candidates immediately. Each candidate contains claim, locations, related_tests, confidence, severity, and category. Use risks only for unresolved cautions that are not yet evidence-grounded findings. Do not wait for the finish action to preserve a finding.
memory_candidates should expose potential memory entries only; do not assume they will be written.
When concrete failures or findings exist, user_update should report the useful evidence rather than only a finding count. Include relevant file paths or line numbers, observed behavior, and whether the root cause is confirmed or still being investigated. Use concise line-separated bullets when there are multiple findings.
user_update may be empty when there is no meaningful progress to report. Do not reveal chain-of-thought.
