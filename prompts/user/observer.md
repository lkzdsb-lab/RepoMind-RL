Return JSON with keys: latest_tool, status, summary, new_findings, hypotheses, missing_context, next_search_terms, confidence.
status must be one of ok, error, inconclusive, complete.
Use short list fields. If uncertain, use empty lists instead of guessing.

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
latest_tool_call={{ latest_tool_call }}
