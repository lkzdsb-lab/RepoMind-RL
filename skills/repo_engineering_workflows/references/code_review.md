# Code Review Workflow

Use this workflow when the user asks to review code, check whether code is correct,
find bugs or risks, or inspect implementation quality without necessarily
requesting a patch.

This workflow is evidence-driven. Do not make final claims before reading the
smallest necessary dependency context.

## Workflow

1. Determine the review scope from the user request.
   - First identify the repository language from project evidence such as
     `project_profile.primary_language`, dominant extensions, package manifests,
     and candidate files.
   - Combine repository language evidence with the user's natural language; do
     not let either one override concrete evidence from the other without noting
     uncertainty.
   - If the user names files, functions, APIs, or modules, start there.
   - If the scope is vague, use task analysis and `search_code_context` to find
     candidate files first.
   - Treat names like `main`, `app`, `server`, or `index` as language-neutral
     identifiers until repository evidence proves the language.
2. Build the smallest necessary dependency closure before judging correctness:
   - target file or entrypoint
   - directly called functions or methods
   - key types, interfaces, models, DTOs, constants, and config
   - route, middleware, service, repository, or storage boundary when relevant
   - related tests or fixtures when available
3. Read dependencies incrementally. Stop expanding when additional files no
   longer change the correctness argument.
4. Prefer structured search:
   - query likely route, handler, service, repository, model, interface, and test names
   - use `code_context` categories to prioritize files that appear in multiple signals
   - use raw `search_code` only for exact strings, error text, symbols, or config keys
5. Keep review conclusions tied to evidence from files that were actually read.

## Output Rules

- Lead with findings, ordered by severity.
- For each finding include:
  - file or symbol
  - why it is a problem
  - possible impact
  - suggested fix or follow-up
- If no issues are found, say so explicitly and list the context that was checked.
- Mention remaining uncertainty when dependencies, tests, runtime config, or external
  contracts were not available.

## Verification

- Do not run `verify_command` just because this is a review.
- Whether verification is needed comes from `task_analysis.verification_required`.
- If verification is not required, use available code and tests as evidence only.
- If verification is required and test results exist, incorporate the command and
  exit code into the review.

## Avoid

- Do not review only the first file when the behavior depends on callers, callees,
  interfaces, models, or configuration.
- Do not read the entire repository by default.
- Do not infer Go, Python, Java, Rust, or C/C++ solely from generic entrypoint
  terms such as `main`.
- Do not infer behavior from file names alone.
- Do not claim a bug exists without a concrete code path or contract mismatch.
- Do not bury findings under a long summary.
