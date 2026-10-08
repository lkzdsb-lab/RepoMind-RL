# Current Task

Title: {{ title }}

Description:
{{ description }}

Existing task-type hint: {{ current_task_type }}

# Repository Context

Project profile:
{{ project_profile }}

Available runtime capabilities:
{{ registry_snapshot }}

# Historical Session Context

{{ session_memory }}

Current-run follow-up user messages:
{{ user_inputs }}

Historical outcomes require revalidation. Explicit, unrevoked user requirements
can authorize their original scope when continuing the same topic. A test-only
permission never authorizes business-code edits. Current explicit instructions override
earlier ones. If task reference/scope is ambiguous, preserve that uncertainty.

# Historical Long-Term Memory

{{ long_term_memory_context }}

Treat long-term memory as historical search guidance. It may be stale or scoped
to an older repository state. Never turn it into current acceptance criteria or
current facts without repository evidence.

# Intent Rules

- `diagnose`: inspect, debug, identify, or explain what is wrong without changing code.
- `review`: review or audit code without changing code.
- `explain`: answer a conceptual or code-understanding question without changing code.
- `implement`: the current message or an active user requirement in the continued topic explicitly asks for changes.
- Generic wording such as "find bugs", "看看有什么问题", or "分析一下" is diagnose, even if historical memory contains fixes.
- A follow-up such as "修复刚才的问题" is implement and may use historical findings.
- Use `DIAGNOSE` for diagnose, review, and explain. Use `BUG_FIX` or `FEATURE_IMPL` only for implement.

# Review Scope Rules

- Tests and README examples are evidence and partial contracts, not proof that untested code is correct.
- For broad bug finding, diagnose, or review requests, define a proportional independent review scope from the implementation. Consider correctness, boundary inputs, concurrency, security, resource handling, API contracts, and test gaps when relevant to the repository.
- For a narrowly targeted fix, keep the review focus adjacent to the changed behavior rather than expanding into an unrelated audit.
- Acceptance criteria must describe the user's requested outcome. Do not turn the names of existing tests into an exhaustive specification.

# Output

Also return session_relation (new/continue/switch), topic_id (an existing supplied
topic ID for continue/switch, empty for new), objective (the actual ongoing goal,
not merely "continue"), and requirement_updates. Do not create a new topic for retries.
Each requirement update has item_id (empty for add), operation (add/replace/revoke),
quote (an exact substring of a current-run user message), description, and edit_scope
(none/tests/repository). Preserve every distinct user constraint, including conditional
requests such as adding missing tests. Reuse existing requirements rather than copying
them on every continuation. Only user messages can revoke or replace requirements.
Memory items omitted from the projection still exist; never assume they are completed.

Return exactly one JSON object:

{
  "intent": "diagnose | implement | explain | review",
  "task_type": "BUG_FIX | FEATURE_IMPL | DIAGNOSE",
  "task_category": "short category or empty string",
  "entities": ["current-task grounded symbol, file, component, or concept"],
  "acceptance_criteria": ["user-visible outcomes"],
  "risk_notes": ["task-specific constraints or risks"],
  "review_focus": ["relevant independent review dimension"],
  "search_hints": ["grounded search terms"],
  "historical_context": ["relevant prior findings clearly labeled as historical"],
  "user_update": "brief progress message or empty string"
}

search_hints must be a list of strings; use [] when there are no grounded search terms.

Do not invent repository facts. Keep historical findings separate from current
acceptance criteria. Do not invent dependencies, evidence policies or execution queues.
Completed requirements retain their explicit permission/constraints within the topic;
only a later user replacement or revocation changes them. Raw-request work items are
authoritative user text, not assistant summaries.
