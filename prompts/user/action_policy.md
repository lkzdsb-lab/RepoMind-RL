title={{ title }}
description={{ description }}
current_step={{ current_step }}
status={{ status }}
verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verification_stale={{ verification_stale }}
verification_commands={{ verification_commands }}
command_results={{ command_results }}
plan_mode={{ plan_mode }}
plan_mode_approved={{ plan_mode_approved }}
debug_technical_plan={{ debug_technical_plan }}
plan_mode_evaluation={{ plan_mode_evaluation }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
selected_code_context_summary={{ selected_code_context_summary }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
full_read_requirements={{ full_read_requirements }}
test_results={{ test_results }}
patch_summary={{ patch_summary }}
editing_enabled={{ editing_enabled }}
edit_results={{ edit_results }}
user_inputs={{ user_inputs }}
pending_resolution={{ pending_resolution }}
memory_context={{ memory_context }}
compressed_context={{ compressed_context }}
legal_actions={{ legal_actions }}
action_constraints={{ action_constraints }}
fallback_action={{ fallback_action }}

Choose only from legal_actions.
Each legal action includes a short description, required_fields when applicable, and permissions/notes when relevant.
Return an ordered candidate_actions list so the q-table guard can truncate your choices after selection.
Put the single best next action first.
Use action_constraints as the compact summary of current execution-phase restrictions, focus files, full-read requirements, and fallback routing.
If a selected tool has required_fields, you must fill them. Do not leave required fields blank.
Include user_update as a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.

Selection rules:
- Choose the next tool that collects the smallest missing evidence needed for the task.
- Before any code-changing action, choose EnterPlanMode and provide a detailed Debug/Refactor technical plan. Do this after enough local evidence has been collected to make the plan concrete.
- When plan_mode is true, do not choose any code-changing action. Your only implementation-related job is to refine/evaluate the technical plan, ask the user if uncertainty remains, then choose ExitPlanMode.
- When plan_mode is true and debug_technical_plan is already present, do not choose EnterPlanMode again.
- Choose ExitPlanMode only when debug_technical_plan is concrete, risks are evaluated, verification commands are identified, and remaining_uncertainties is empty.
- If candidate_files is empty and code_context is not available yet, prefer search_code_context.
- Use search_text for focused regex/fixed-string grep when structured context is missing, stale, or too broad.
- Use selected_code_context_summary as the primary structured evidence about where the relevant logic lives.
- Treat full_read_requirements as a hard signal: when a file appears there, summaries/excerpts are not enough yet, so read the complete file before concluding it is already correct or before patching it.
- If candidate_files contains unread files, prefer read_file only when you need exact source text for the file you intend to patch or when selected_code_context_summary is still insufficient.
- If a candidate file is already present in read_files with full_read=true, use that evidence instead of requesting it again. Treat read_files as durable memory for this run.
- If a candidate file is already present in read_files but full_read=false and it appears in full_read_requirements, you still need a complete read before making a final diagnosis or patch decision.
- If pending_resolution is non-empty, treat it as the highest-priority unresolved runtime state from the previous step. Follow its required_next_action before changing topic.
- If verification_required is true and enough relevant code has been read, prefer run_shell_command with purpose="verification". run_tests is only a compatibility alias.
- After apply_code_patch succeeds, verification_stale becomes true. Do not choose finish, git_diff, or write_memory until a verification command has run.
- When verification_stale is true, choose run_shell_command with purpose="verification" and the narrowest useful command.
- When verification_stale is true, do not ask the user for original file content, expected behavior, or review scope. Verify the current repository state with tools.
- If verification_required is false, do not choose run_tests just to be safe.
- If editing_enabled is true, choose apply_code_patch only after reading the exact file content that must change, ensuring target files have full_read=true when they are relevant patch targets, and after plan_mode_approved is true.
- For apply_code_patch, prefer exact replacement changes whose old_text comes from read_files. Use append only for end-of-file additions, and use insert_after or insert_before only with an exact anchor old_text that comes from read_files.
- Preserve the file's existing structural conventions. Keep package/import sections at the top for Go, keep Python import blocks at the beginning, and do not place executable logic before them.
- Prefer the smallest structurally valid edit. Reuse existing formatting, indentation, blank-line spacing, and declaration order unless the task explicitly requires a reorganization.
- If expected behavior, business rule, target file, compatibility impact, data migration, permission/auth behavior, or public API behavior is uncertain, choose request_user_input only when you can ask 1-3 concrete questions.
- If the user only asks to modify/edit a file but does not specify the desired behavior, locate and read the file if needed, then choose request_user_input with concrete questions about the intended change. Do not infer an arbitrary code change only because the file contains suspicious code.
- If you cannot write a concrete user question, do not choose request_user_input; use the available tools to gather more evidence or choose finish when enough evidence exists.
- Do not edit files that were not read in this run.
- Do not edit generated, vendor, dependency, lock, secret, or runtime artifact files unless the user explicitly requires it.
- If enough code evidence has been read and no verification is required, consider git_diff or finish.
- Do not choose repository-wide file listing unless it is present in legal_actions and the user explicitly asks for the file tree.
- Follow selected skill constraints, especially code review rules about reading the smallest necessary dependency closure.

Return JSON like {"action":"search_code_context","candidate_actions":["search_code_context","git_diff"],"reason":"...","action_input":{},"uncertainty_questions":[],"confidence":0.8,"user_update":"我会先定位相关代码，再决定是否需要修改。"}.
