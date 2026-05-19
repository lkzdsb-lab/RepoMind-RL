# Test Failure Triage

Use this skill when verification fails or a task starts from failing tests.

## Workflow

1. Identify the failing command, package, test name, assertion, and first meaningful error.
2. Search for the test name, assertion text, or error string before changing source code.
3. Map test files to source files:
   - `foo_test.go` -> `foo.go`
   - `test_foo.py` -> `foo.py` or package under test
   - integration tests -> route/service/client boundary
4. Read the test and the smallest related source files.
5. Classify the failure:
   - assertion mismatch
   - panic/exception
   - import/build error
   - timeout
   - flaky ordering or race
   - fixture/setup issue
6. Patch the behavior or test setup that matches the classification.
7. Re-run the narrow failing test first, then broaden verification.

## Signals

- Import/build errors usually require dependency, path, or symbol fixes before logic changes.
- Timeout failures often point to network calls, goroutines, locks, retries, or waits.
- Flaky failures need ordering, clock, randomness, race, or fixture isolation checks.

## Avoid

- Do not patch production logic before reading the failing test.
- Do not use broad test output as the only search query when a test name is available.
- Do not mark verification complete without recording command and exit code.
