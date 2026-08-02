# Current Task

title={{ title }}
description={{ description }}
status={{ status }}
current_step={{ current_step }}
error={{ error }}
project_profile={{ project_profile }}
task_brief={{ task_brief }}
work_plan={{ work_plan }}
runtime_facts={{ runtime_facts }}
task_analysis={{ task_analysis }}

# Draft Findings Requiring Independent Review

draft_findings={{ draft_findings }}
candidate_evidence_packets={{ candidate_evidence_packets }}

# Runtime Evidence

verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verification_stale={{ verification_stale }}
verification_commands={{ verification_commands }}
step_approval_history={{ step_approval_history }}
plan_mode={{ plan_mode }}
plan_mode_approved={{ plan_mode_approved }}
technical_plan={{ technical_plan }}
plan_mode_evaluation={{ plan_mode_evaluation }}
plan={{ plan }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
tool_calls={{ tool_calls }}
test_results={{ test_results }}
edit_results={{ edit_results }}
has_patch={{ has_patch }}
patch_summary={{ patch_summary }}
llm_observations={{ llm_observations }}
user_inputs={{ user_inputs }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
fallback_judgement={{ fallback_judgement }}

# Decision Rules

- Judge semantic completion against task_brief.objective and every acceptance item.
- Current user intent is authoritative. Historical findings are search guidance only, not current facts or edit authorization.
- Rank evidence as follows: current test/command/tool output, current repository reads/diffs, grounded current-run observations, then plans/candidates/history.
- Before choosing complete, ensure every confirmed finding mentioned in the reason is explicitly supported by current-run evidence.
- Do not claim that a test confirmed a finding unless that finding appears in the current test output or is independently confirmed by another current-run tool result.
- Do not preserve a historical finding count. If current evidence supports fewer findings, use the current count and omit invalidated or unconfirmed historical claims.
- When current repository or tool evidence conflicts with historical context, current evidence wins.
- Choose complete only when repository/tool evidence supports the requested outcome.
- Passing tests are supporting evidence, not proof that unexercised implementation paths are correct.
- For diagnose, review, and broad bug-finding tasks, require evidence that the relevant task_brief.review_focus dimensions were considered against implementation code. If a material dimension remains inspectable with repository tools, choose continue with a focused action.
- Every evidence-grounded finding_candidate from Policy or Observer must remain represented in draft_findings and receive an explicit review verdict. Do not discard a confirmed risk merely because it is outside visible tests.
- For diagnose, review, or explain, no patch is required unless the current message explicitly requested one.
- For implement, verify that the requested behavior was actually changed; do not infer completion merely because a file was edited.
- When a safe user-scenario smoke verification was applicable, require its result before claiming that the described functional path was verified. A generic unit or repository-wide test does not substitute for that evidence. If the smoke path could not be run because approval was not granted, or because of credentials, external services, side effects, or command restrictions, completion may rely on other evidence only when the limitation is stated explicitly; do not request the same declined command again unless the user changes direction.
- If plan_mode is active, choose continue with ExitPlanMode.
- If the latest code revision is unverified or verification_stale is true, choose continue with run_shell_command.
- A failed verification may reveal missing implementation. Suggest read_file, search_text, or apply_code_patch as appropriate rather than blindly repeating the command.
- If repository tools can obtain missing evidence, choose continue and name the most useful action.
- Choose needs_user_input only for a concrete ambiguity that repository tools cannot resolve, and ask 1-3 questions.
- Do not use work_plan status as proof by itself; corroborate it with tool evidence.
- Do not use task_analysis, task_brief historical_context, plans, candidate_files, selected skills, or fallback_judgement as proof.
- Review every draft finding independently from first principles. Policy confidence and explanations are advisory only.
- Match findings and evidence exclusively by candidate_id and evidence_id. Never use evidence from another candidate packet without explicitly explaining the cross-reference.
- Return exactly one reviewed_findings item for every draft candidate, with verdict confirmed, rejected, or needs_more_evidence.
- Confirm source-proven behavior even when a prior panic prevented its test from running, but distinguish source evidence from runtime test evidence.
- Reject a candidate when current source or language semantics contradict it. Do not preserve the Policy finding count.
- Use needs_more_evidence only when the available tools can materially resolve uncertainty, and provide a focused recommended_next_action.
- If any candidate needs more evidence, choose continue. Do not omit a candidate to allow completion.

# Output

Return JSON with keys: decision, reason, questions, suggested_next_action, reviewed_findings, missing_evidence, confidence, user_update.
Each reviewed_findings item must contain candidate_id, verdict, claim, evidence_refs, reason, and recommended_next_action.
confidence must be from 0.0 to 1.0. Do not reveal chain-of-thought.
