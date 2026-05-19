# Codebase Context Workflow

Use this skill when the agent needs structured code search beyond raw grep.
The skill describes how to use the codebase context tools; it does not replace the tools.

## Workflow

1. Build or refresh the index with `build_codebase_context` when the index is missing,
   stale, or the task depends on route/symbol/model relationships.
2. Query `search_code_context` with task-specific identifiers from:
   - task title and description
   - task analyzer entities and search hints
   - failing test names or error strings
   - memory context and selected skills
3. Inspect search result categories separately:
   - files: likely files to read
   - functions/symbols: likely implementation entry points
   - api_routes: route to handler/controller mapping
   - db_models: table/model relationships
   - call_graph: lightweight caller/callee hints
   - test_mappings: source-to-test links
4. Prefer reading files that appear in more than one category.
5. Use query refinement when results are broad: add route path, method name, table name,
   entity status, or failing assertion text.

## Go Flow

- handler -> service -> repository -> model
- route -> middleware -> controller
- interface -> implementation -> DB table -> test file

## Avoid

- Do not rely only on raw `search_code` when structured route/symbol/model context exists.
- Do not read every returned file. Rerank using result category, score, and task fit.
- Do not rebuild the index repeatedly inside one run unless files changed materially.
