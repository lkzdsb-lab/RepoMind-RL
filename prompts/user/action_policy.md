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
finding_evidence={{ finding_evidence }}
finding_update_errors={{ finding_update_errors }}

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
analysis_records={{ analysis_records }}
analysis_progress={{ analysis_progress }}
read_file_inventory={{ read_file_inventory }}
read_files={{ read_files }}
read_file_projection={{ read_file_projection }}
selected_code_context_summary={{ selected_code_context_summary }}
test_results={{ test_results }}
patch_summary={{ patch_summary }}
selected_skills={{ selected_skills }}
skill_context={{ skill_context }}
user_inputs={{ user_inputs }}
memory_context={{ memory_context }}
long_term_memory_context={{ long_term_memory_context }}
compressed_context={{ compressed_context }}

# Available Actions

legal_actions={{ legal_actions }}
action_constraints={{ action_constraints }}
decision_feedback={{ decision_feedback }}

# Decision Rules

- Choose exactly one action from legal_actions. Never invent an action.
- Return analysis_updates after analyzing shown source, including when no defect was found. Each update needs file_path, full file_revision from shown_ranges, start_line, end_line, dimension, conclusion, status (partial or complete), finding_ids and open_questions. A read alone is not completed analysis. Complete means the stated range and dimension are resolved; use partial when questions remain.
- Reuse non-stale analysis_records even when their source is omitted in this prompt. Revisit analyzed code only to answer a concrete new question or reconcile evidence. Omitted source does not reset analysis progress. Do not repeatedly reconfirm unchanged conclusions before finish.
- Finding IDs in analysis_updates must be existing runtime candidate IDs; leave finding_ids empty for a new finding whose runtime ID is not yet assigned.
- analysis_updates.dimension must be correctness, boundary, concurrency, security, resource, api_contract, test_gap, or other. Reuse the same dimension for the same question; do not invent labels to represent progress.
- Always return top-level confidence as a number between 0 and 1. It is required for every action; never omit it, set it to null, or place it inside action_input.
- When decision_feedback contains required_action, keep that action and repair only its action_input using validation_errors and expected_input_fields.
- When decision_feedback.reason is read_range_already_shown, the requested range is already fully present in this prompt. Use that source to continue analysis; changing max_chars adds no information. Decision repair attempts are bounded.
- Build action_input from the selected action's flat input_fields list. Do not use aliases such as search_query, and do not place top-level response fields inside action_input. Required fields must be present and valid.
- For a ranged read, return separate named fields, for example: `{"file_path":"server.go","start_line":30,"end_line":131,"max_chars":12000}`. Never encode a line range as an unnamed array.
- The current task_brief and active user requirements define the task. Historical findings are hypotheses; only explicit user requirements authorize edits within their recorded scope.
- Long-term memory is historical guidance, not current-run evidence. Re-read current files and revalidate applicable claims before relying on it.
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
- read_file_inventory describes cached coverage; read_files contains the source actually shown in this prompt. read_file_projection lists shown_ranges and omitted_ranges. If needed source is cached but omitted, request a focused read_file range: the runtime can return it from cache. Reuse ranges already fully shown instead of reading them again. A successful read does not itself mean its code has been analyzed.
- Before patching, ensure every exact old_text anchor appears in source read during this run. The runtime validates this requirement.
- Copy patch old_text exactly from one supplied source range. Never reconstruct an anchor from memory, plans, summaries, or historical findings. If the required source is absent, request a focused read_file range.
- After a successful patch, verification_stale becomes true. You may inspect or patch further, but finish remains invalid until the latest edit is successfully verified.
- For diagnose, review, and explain intents, do not enter Plan Mode or patch. Finish once repository evidence answers the current acceptance criteria.
- For implement intent, use Plan Mode before the first patch. After it is approved, continue to patch, inspect, and verify as evidence requires.
- Use request_user_input only when repository tools cannot resolve a concrete ambiguity.
- When completion_judgement requests more evidence for a draft finding, use its focused recommended action before attempting finish again.
- When completion_judgement.cached is true, the runtime reused a successful review because the evidence is unchanged. Address its missing_evidence or recommended_next_action; rewriting the plan or calling finish again adds no evidence. Repeated unchanged finish requests terminate the run as incomplete.
- When work_plan changes, return plan_update with exactly steps, current_focus, and open_questions. Each step requires id, description, and status; status must be pending, in_progress, done, or blocked. Include the complete current step list. Return an empty plan_update when nothing changed. Do not claim a step is done without evidence.

# Output

Session memory and draft_findings are bounded projections. Inspect omitted items/details
with read_session_memory; paginate using next_offset and revision. Historical candidates
must be reviewed or explicitly left unresolved. Do not preserve them as current facts
without current source/tool evidence. Reuse a historical candidate_id only for that
same defect; memory reads do not count as independent verification.

Return exactly one JSON object:

{
  "action": "one legal action name",
  "reason": "brief evidence-based reason",
  "action_input": {},
  "uncertainty_questions": [],
  "confidence": 0.0,
  "analysis_updates": [],
  "draft_findings": [
    {
      "operation": "create | enrich | revise",
      "candidate_id": "reuse the existing ID for an update; empty string for a new defect",
      "evidence_refs": ["tool:0"],
      "revision_reason": "required for revise; otherwise empty",
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

Candidate entries are update commands. Use create with an empty candidate_id for a
distinct root cause. Use enrich with the existing ID and EXACT existing claim to add
evidence, locations or tests. Use revise with the existing ID, revision_reason and
current evidence to correct the conclusion; this invalidates its previous review.
Every command needs evidence_refs from finding_evidence. tool:N refers to the current
run's tool call, not a history item. Memory reads are not evidence. Citing a tool is
necessary but does not prove that it supports the conclusion: explain only what its
actual output establishes. Do not submit updates merely to reword an existing claim
or raise confidence. Submit [] when nothing changed. Read omitted candidate details
with read_session_memory before updating them. Retain an existing source anchor;
different root causes in the same file/function must remain separate candidates.
Text edits, repeated evidence and toggling analysis status are not new progress.

Keep plan_update empty when no plan state changed. Do not reveal chain-of-thought.
