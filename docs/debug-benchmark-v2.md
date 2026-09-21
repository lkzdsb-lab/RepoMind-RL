# Debug fixture v2

The canonical exercise project is `../agent test`, outside this repository.
The in-repository `agent test` is a synchronized working copy; evaluation cases
explicitly point to the parent fixture. Do not silently switch between copies.

The service adds project-aware task creation/update/filtering, priorities,
statistics caching, batch updates, idempotency and optimistic versions. It keeps
the original API expectations and existing Go tests. It deliberately retains
faults and must only be run on loopback, never as a public service.

## Ground truth

`evaluation/baselines/go_web_debug.json` is the single source of truth for
13 seeded faults: 5 easy, 4 medium and 4 hard. Each contains root-cause location,
symptom, contract, fresh-state prerequisites, executable request steps and review
status. All labels remain `pending_human_review`. A reproducer demonstrates a
specific failure; it does not certify exhaustive correctness or difficulty.

`go_web_debug.json` and its easy/medium/hard variants load finding labels from
this baseline. Difficulty variants select scoring/acceptance subsets on the same
faulty fixture, not separate projects containing only that difficulty. Their
initial prompts therefore do not expose ground-truth file or bug identifiers.
The full case also requires the existing `go test ./...` checks after repairs.

Ground truth is not copied into the agent's workspace. README contains only the
public API contract, not a list of root causes. This prevents accidental context
leakage, but is not an OS security boundary against a malicious agent traversing
outside its configured repository.

## Local acceptance (no Agent or LLM)

```powershell
.\.venv\Scripts\python.exe -m evaluation.debug_http --workspace "../agent test" --baseline evaluation/baselines/go_web_debug.json --expect-bugs --skip-race --report .repomind/evaluation/debug-reproduction.json --timeout 240
```

`--expect-bugs` expects contract violations in the faulty fixture, while normal
smoke paths must still pass. Omit this flag for a repaired copy: all selected HTTP
contracts must pass, and the full suite also runs the existing Go tests. Each bug
gets a fresh process and seed state. The service advertises an OS-allocated
loopback port; processes are terminated by the runner on completion.

Race validation is explicitly excluded at the user's request. Cases currently
set `verification.skip_race=true`. The race label remains in ground truth but the
report marks its check `skipped`, never passed, and `coverage_complete=false`.
No C compiler installation is needed for the selected checks.

Executables/logs use `.repomind/evaluation/tmp`; Go build/module/temp caches use
`.repomind/evaluation/go-cache`. No new Go unit tests are added. Existing test
assertions, go.mod and static assets remain unchanged. Set `EVALUATION_GO` to an
installed Go binary if it is not on PATH.

## Validation artifacts

The original v1 project backup is under
`.repomind/evaluation/fixture-backups/agent-test-v1-20260913/source`.
The same directory contains:

- `reproduction-no-race.json`: 12 faults reproduced, race skipped.
- `reference-fixed/`: isolated reference repairs of those 12 faults only.
- `reference-acceptance.json`: repaired HTTP contracts and existing Go tests.

Reference repairs are not copied to the exercise fixture. Concurrency is not
repaired or certified by this reference copy. No full Agent evaluation has been
run as part of fixture verification.
