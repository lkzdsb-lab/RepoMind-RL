title={{ title }}
description={{ description }}
current_step={{ current_step }}
status={{ status }}
verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
candidate_files={{ candidate_files }}
test_results={{ test_results }}
patch_summary={{ patch_summary }}
memory_context={{ memory_context }}
compressed_context={{ compressed_context }}
legal_actions={{ legal_actions }}
fallback_action={{ fallback_action }}

Choose only from legal_actions.
Return an ordered candidate_actions list so the q-table guard can truncate your choices after selection.
Put the single best next action first.

Selection rules:
- Choose the next tool that collects the smallest missing evidence needed for the task.
- If candidate_files is empty and code_context is not available yet, prefer search_code_context.
- If candidate_files contains unread files, prefer read_file.
- If verification_required is true and enough relevant code has been read, consider run_tests.
- If verification_required is false, do not choose run_tests just to be safe.
- If enough code evidence has been read and no verification is required, consider git_diff or finish.
- Do not choose repository-wide file listing unless it is present in legal_actions and the user explicitly asks for the file tree.
- Follow selected skill constraints, especially code review rules about reading the smallest necessary dependency closure.

Return JSON like {"action": "search_code_context", "candidate_actions": ["search_code_context", "git_diff"], "reason": "..."}.
