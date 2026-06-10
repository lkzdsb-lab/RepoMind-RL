# Code Edit Workflow

Use this workflow when the task requires modifying repository code, implementing a
small change, or fixing a localized bug.

## Workflow

1. Start from the external symptom, requested behavior, failing test, or target
   entity.
2. Use `search_code_context` before raw grep when the codebase index is
   available.
3. Use `search_text` for focused regex or fixed-string grep when the structured
   index is missing, stale, or too broad.
4. Read the smallest relevant file set before editing. Do not edit a file that
   has not been read in this run.
5. Before any code edit, call `EnterPlanMode` and write a detailed
   Debug/Refactor Technical Plan covering target files, intended changes, risks,
   assumptions, and verification commands. While in Plan Mode, do not use any
   code-changing tool.
6. Evaluate the plan. If it is feasible and no uncertainty remains, call
   `ExitPlanMode` with `approved=true`. If uncertainty remains, ask the user.
7. If behavior, constraints, target file, or compatibility impact are uncertain,
   choose `request_user_input` and ask 1-3 concrete questions.
8. Use `apply_code_patch` with minimal exact-replacement edits. The `old_text`
   must come from the already-read file content.
9. Run the narrowest useful verification command after applying edits, using
   `run_shell_command` with `purpose="verification"` when a project-specific
   command is needed.
10. Inspect `git_diff` before finishing and ensure the patch matches the original
   request.

## Must Ask User When

- Multiple valid behaviors or fixes are possible.
- The intended business rule, state transition, permission behavior, or public
  API contract is missing.
- The target file or function cannot be identified from repository evidence.
- The edit affects data migration, auth, billing, permissions, external API
  compatibility, or persistent data.
- The required change would touch generated files, dependency files, lock files,
  secrets, runtime artifacts, or vendor code.
- The verification command is missing and the change is risky.

## Avoid

- Do not edit unread files.
- Do not make broad refactors while fixing a narrow bug.
- Do not infer business rules from names alone.
- Do not create, delete, rename, or move files unless the user explicitly asks
  and the editing guard allows it.
- Do not use `apply_code_patch` when unresolved uncertainty remains; ask the
  user first.
