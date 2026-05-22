# Go Backend Debug

Use this skill when the task involves a Go backend, HTTP route, handler, service,
repository, DB model, middleware, or `go test` failure.

## Workflow

1. Start from the external symptom: route, API name, error string, test name, or domain entity.
2. Use `search_code_context` before raw grep when the codebase index is available.
3. Follow backend flow in this order when possible:
   - route -> middleware -> handler/controller
   - handler/controller -> service
   - service -> repository/client
   - repository -> DB model/table
   - changed source -> related test file
4. Read the smallest set of files that explain the flow. Prefer files returned by route,
   function, DB model, call graph, and test mapping indexes.
5. Run the narrowest useful `go test` command first, then broaden to package or repo scope.
6. Before finishing, inspect `git_diff` and verify the patch against the original symptom.

## Signals

- Route or API names usually point to handler/controller files first.
- GORM tags, `TableName()`, SQL strings, and repository method names usually identify DB model flow.
- Interface implementations often split behavior across service and repository packages.
- A failing Go test name is often a better search term than the top-level error message.

## Avoid

- Do not patch repository or DB code before tracing the handler/service contract.
- Do not run full repo tests before a package-level test if a narrower command is available.
- Do not infer business state transitions without reading the model and repository boundary.
