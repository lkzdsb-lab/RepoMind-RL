# Current Task

title={{ title }}
description={{ description }}
current_step={{ current_step }}
status={{ status }}
task_brief={{ task_brief }}
work_plan={{ work_plan }}
runtime_facts={{ runtime_facts }}
completion_judgement={{ completion_judgement }}
draft_findings={{ draft_findings }}

# Runtime Facts

verification_required={{ verification_required }}
verification_reason={{ verification_reason }}
verification_stale={{ verification_stale }}
verification_commands={{ verification_commands }}
command_results={{ command_results }}
verification_capabilities={{ verification_capabilities }}
plan_mode={{ plan_mode }}
plan_mode_approved={{ plan_mode_approved }}
technical_plan={{ technical_plan }}
plan_mode_evaluation={{ plan_mode_evaluation }}
editing_enabled={{ editing_enabled }}
edit_results={{ edit_results }}
pending_resolution={{ pending_resolution }}

# Working Context

attention_focus={{ attention_focus }}
candidate_files={{ candidate_files }}
read_files={{ read_files }}
selected_code_context_summary={{ selected_code_context_summary }}
test_results={{ test_results }}
patch_summary={{ patch_summary }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
user_inputs={{ user_inputs }}
memory_context={{ memory_context }}
compressed_context={{ compressed_context }}

# Available Actions

legal_actions={{ legal_actions }}
action_constraints={{ action_constraints }}
decision_feedback={{ decision_feedback }}

# Decision Rules

- Choose exactly one action from legal_actions. Never invent an action.
- Always return top-level confidence as a number between 0 and 1. It is required for every action; never omit it, set it to null, or place it inside action_input.
- When decision_feedback contains required_action, keep that action and repair only its action_input using validation_errors and expected_input_fields.
- Build action_input from the selected action's flat input_fields list. Do not use aliases such as search_query, and do not place top-level response fields inside action_input. Required fields must be present and valid.
- For a ranged read, return separate named fields, for example: `{"file_path":"server.go","start_line":30,"end_line":131,"max_chars":12000}`. Never encode a line range as an unnamed array.
- The current task_brief is authoritative. Historical memory supplies context but never grants permission to edit.
- You own semantic sequencing. Decide whether to inspect, patch, verify, ask, or finish from the evidence; no execution queue will decide this for you.
- A failed verification is evidence. If it identifies missing implementation, inspect or patch before running the same command again.
- Passing tests prove only the exercised behavior. They do not replace independent implementation analysis for diagnose, review, or broad bug-finding tasks.
- Use task_brief.review_focus to inspect relevant correctness, boundary, concurrency, security, resource, API-contract, and test-gap risks before finishing. Apply this proportionally; do not expand a targeted fix into an unrelated audit.
- Choose verification from the observed risk: concurrency may require race-oriented checks, input arithmetic may require boundary cases or fuzzing, and filesystem security may require adversarial path checks. Do not run checks without a code-grounded reason.
- For debug or implement tasks with a concrete user-described behavior, prefer one bounded smoke verification that exercises the same input and functional path. Reproduce it before the fix when safe and useful, then repeat it after the fix. Use run_shell_command with purpose="verification" and verification_kind="smoke"; the runtime will request user approval before execution.
- Never relabel a smoke command as verification_kind="standard" to avoid approval. If approval is not granted, treat the response as user direction and replan without executing that smoke path.
- Do not label ordinary unit, compile, lint, or repository-wide test commands as smoke. If the functional path needs credentials, external services, destructive writes, shell composition, or commands outside verification_capabilities, do not bypass the guard; use standard safe verification and disclose the unverified behavior.
- Do not repeat an identical command when repository state and evidence have not changed.
- After list_files identifies concrete files, read the relevant candidates instead of listing or searching for the same files again. A deterministic repository lookup cannot be repeated until an edit changes repository evidence.
- Decide the necessary file scope yourself. Prefer focused line ranges for large files and full reads only when whole-file structure matters.
- The read_files section contains exact source ranges from the current validated cache. Reuse them instead of reading the same file again when they cover the required code.
- Before patching, ensure every exact old_text anchor appears in source read during this run. The runtime validates this requirement.
- Copy patch old_text exactly from one supplied source range. Never reconstruct an anchor from memory, plans, summaries, or historical findings. If the required source is absent, request a focused read_file range.
- After a successful patch, verification_stale becomes true. You may inspect or patch further, but finish remains invalid until the latest edit is successfully verified.
- For diagnose, review, and explain intents, do not enter Plan Mode or patch. Finish once repository evidence answers the current acceptance criteria.
- For implement intent, use Plan Mode before the first patch. After it is approved, continue to patch, inspect, and verify as evidence requires.
- Use request_user_input only when repository tools cannot resolve a concrete ambiguity.
- When completion_judgement requests more evidence for a draft finding, use its focused recommended action before attempting finish again.
- When work_plan changes, return plan_update with exactly steps, current_focus, and open_questions. Each step requires id, description, and status; status must be pending, in_progress, done, or blocked. Include the complete current step list. Return an empty plan_update when nothing changed. Do not claim a step is done without evidence.

# Output

Return exactly one JSON object:

{
  "action": "one legal action name",
  "reason": "brief evidence-based reason",
  "action_input": {},
  "uncertainty_questions": [],
  "confidence": 0.0,
  "draft_findings": [
    {
      "candidate_id": "optional; runtime assigns the stable id",
      "claim": "candidate conclusion to review",
      "locations": [
        {
          "file_path": "relative/path",
          "symbol": "optional symbol",
          "start_line": 1,
          "end_line": 1
        }
      ],
      "related_tests": [],
      "confidence": 0.0,
      "severity": "low | medium | high | critical",
      "category": "correctness | boundary | concurrency | security | resource | api_contract | test_gap | other"
    }
  ],
  "plan_update": {
    "steps": [
      {
        "id": "stable step id",
        "description": "work item",
        "status": "pending | in_progress | done | blocked"
      }
    ],
    "current_focus": "current work",
    "open_questions": []
  },
  "user_update": "short user-facing update or empty string"
}

Submit an evidence-grounded candidate as soon as it is discovered, regardless
of the selected action. The runtime merges candidates across decisions. When
choosing finish, include any additional distinct issue not already present in
draft_findings. Never omit an existing candidate to make completion easier.
Include precise current-run file locations and related test names when
available. Do not hide a candidate merely because its confidence is low.

Keep plan_update empty when no plan state changed. Do not reveal chain-of-thought.
