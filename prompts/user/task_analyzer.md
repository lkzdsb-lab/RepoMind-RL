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

Treat all session-memory outcomes, files, and next steps as historical advice.
Use them to resolve references such as "the previous issue" or "continue", but
do not treat them as authorization to edit. Only the current message determines
whether implementation is allowed.

# Intent Rules

- `diagnose`: inspect, debug, identify, or explain what is wrong without changing code.
- `review`: review or audit code without changing code.
- `explain`: answer a conceptual or code-understanding question without changing code.
- `implement`: the current message explicitly asks to add, change, fix, refactor, or remove code.
- Generic wording such as "find bugs", "看看有什么问题", or "分析一下" is diagnose, even if historical memory contains fixes.
- A follow-up such as "修复刚才的问题" is implement and may use historical findings.
- Use `DIAGNOSE` for diagnose, review, and explain. Use `BUG_FIX` or `FEATURE_IMPL` only for implement.

# Review Scope Rules

- Tests and README examples are evidence and partial contracts, not proof that untested code is correct.
- For broad bug finding, diagnose, or review requests, define a proportional independent review scope from the implementation. Consider correctness, boundary inputs, concurrency, security, resource handling, API contracts, and test gaps when relevant to the repository.
- For a narrowly targeted fix, keep the review focus adjacent to the changed behavior rather than expanding into an unrelated audit.
- Acceptance criteria must describe the user's requested outcome. Do not turn the names of existing tests into an exhaustive specification.

# Output

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

Do not invent repository facts. Keep historical findings separate from current
acceptance criteria. Do not include completion criteria, dependencies, evidence
policies, obligations, or execution queues.
