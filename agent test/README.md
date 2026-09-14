# Project Task Service

A small, local-only Go HTTP service for debugging-agent evaluation.
Run `go run .` (Go 1.25 or later). Default listener: 127.0.0.1:8080.
Set `AGENT_TEST_ADDR=127.0.0.1:0` for a dynamically assigned port; the actual
address is printed at startup. Do not expose this exercise service publicly.

## Data and API contract

Each process starts with two projects (1 Engine, 2 Console), users 1 and 2,
and four tasks. Tasks 1/2 belong to project 1; tasks 3/4 to project 2.
Only task 1 is initially completed. Every task starts at version 1.
The service is intended to support concurrent requests.

- GET /health: service status.
- GET /projects: project list.
- GET /todos?page=1&limit=20&project_id=1&done=false: optional project/status
  filters, then one-based pagination; results are ordered by increasing ID.
  Default page is 1, default limit is 20, maximum limit is 100.
- POST /todos: JSON title, user_id, project_id, priority. IDs must exist.
  Title must contain non-whitespace characters; priority is an integer in [1,5].
  Success returns 201 and the created task.
- Optional Idempotency-Key on creation: retries with the same key WITHIN ONE
  project return the original result with status 200. Another project using the
  same key is an independent creation. The first accepted payload wins within
  the same project/key pair.
- PATCH /todos/{id}: JSON version plus optional title/done/priority. Version must
  match current version; success returns 200 and increments it once. A stale
  version returns 409 without changing any field or consuming a version.
- DELETE /todos/{id}: delete a task, return 204. Other methods (apart from PATCH)
  return 405 and must not mutate data.
- PATCH /todos/batch: JSON items array containing id, version and update fields.
  1..100 distinct IDs, all validated and applied atomically. Any missing ID or
  stale version rejects the entire batch (404 or 409) without changes.
- GET /projects/{id}/stats: project_id, total, completed. Results must remain
  project-specific and immediately reflect successful creates, updates and
  deletions, whether or not a previous response was cached.
- GET /users/{id}: user details; missing users return 404.
- GET /files?name=welcome.txt: serve files beneath static only; attempts to escape
  that directory return 400.

Run existing checks with `go test ./...`. The HTTP contract above defines correct
behavior; this exercise implementation is not guaranteed to satisfy it.
Do not change the contract or existing test expectations to make a repair pass.
Detailed evaluator labels and acceptance scenarios are maintained outside the
project so they are not part of the agent's repository context.
