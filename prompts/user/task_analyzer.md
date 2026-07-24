# Task

Title: {{ title }}

Description:
{{ description }}

Existing task-type hint: {{ current_task_type }}

# Repository Evidence

Project profile:
{{ project_profile }}

Available runtime capabilities:
{{ registry_snapshot }}

# Analysis Procedure

1. Determine whether the requested outcome is a bug fix, feature implementation, or read-only diagnosis/review.
2. Extract acceptance criteria as user-visible outcomes.
3. Convert only the necessary work into runtime-verifiable completion criteria.
4. Add dependencies only when one criterion genuinely requires evidence from another.
5. Produce search hints and entities grounded in the task or repository evidence.

Do not force every task through diagnose, implement, and verify. A read-only explanation may need only one diagnose criterion. A direct implementation may not need a separate diagnosis criterion. Include verification only when command-based evidence is necessary for the requested outcome.

# Completion Criteria

Each completion_criteria item must contain:

- id: a stable snake_case identifier.
- kind: diagnose, implement, or verify.
- description: the required outcome, not merely an action such as "read a file".
- required: whether the task may finish without this criterion.
- depends_on: IDs of criteria that must pass first.
- evidence_policy: repository_evidence, diagnosis_evidence, command_evidence, patch_applied, or verification_passed.

Evidence policy meanings:

- repository_evidence: source or repository evidence is sufficient for a read-only investigation.
- diagnosis_evidence: concrete failure or root-cause evidence is required.
- command_evidence: an allowed build/test command must run and its result is evidence, regardless of exit code.
- patch_applied: an actual repository change is required.
- verification_passed: an allowed command must complete successfully.

Consistency requirements:

- An implement criterion must use patch_applied.
- A verify criterion must use verification_passed.
- A diagnose criterion may use repository_evidence, diagnosis_evidence, or command_evidence.
- depends_on may reference only IDs declared in the same response.
- Do not add implement when the user requested only explanation, review, or diagnosis.
- Do not add verify merely as a precaution when repository inspection can fully satisfy the task.
- For executable project audits that explicitly ask to find problems, add a command_evidence diagnose criterion when an allowed build or test command can reveal objective failures.
- Keep acceptance_criteria user-facing; keep completion_criteria runtime-verifiable.

# Repository Grounding

- Use project_profile as the primary language evidence.
- Do not infer a language from a generic word such as "main".
- Add language-specific criteria only when supported by the task or project_profile.
- When project_profile has one clear primary language and the task is language-neutral, use that language context.
- If information is uncertain, use an empty list or string instead of guessing.

# Output Schema

Return exactly one JSON object with this shape:

{
  "task_type": "BUG_FIX | FEATURE_IMPL | DIAGNOSE",
  "task_category": "short category or empty string",
  "entities": ["task-grounded symbol, file, component, or concept"],
  "acceptance_criteria": ["user-visible outcome"],
  "completion_criteria": [
    {
      "id": "stable_criterion_id",
      "kind": "diagnose | implement | verify",
      "description": "runtime-verifiable required outcome",
      "required": true,
      "depends_on": [],
      "evidence_policy": "repository_evidence | diagnosis_evidence | command_evidence | patch_applied | verification_passed"
    }
  ],
  "risk_notes": ["task-specific risk"],
  "search_hints": ["grounded search term"],
  "user_update": "brief progress message or empty string"
}
