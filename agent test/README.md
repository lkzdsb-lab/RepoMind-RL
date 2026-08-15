# Agent Test Go Web Demo

This is a small Go web project for testing debugging agents.

It intentionally contains classic web/API bugs while keeping the codebase small:

- Unsafe file path handling in `GET /files`.
- Off-by-one pagination in `GET /todos`.
- Missing HTTP method enforcement in `DELETE /todos/{id}`.
- Missing not-found handling in `GET /users/{id}`.
- In-memory store without concurrency protection.

Run the server:

```powershell
go run .
```

Run the tests:

```powershell
go test ./...
```

The tests describe the expected correct behavior. A debugging agent should inspect the code, fix the implementation, and make the tests pass without changing the test expectations.
