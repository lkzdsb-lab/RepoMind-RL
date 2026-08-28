# Long-term Memory: Phase 2 Markdown Store

Phase 2 implements the canonical, human-readable long-term memory store. It does
not extract memories from Task Archive and does not introduce SQLite catalogs,
embeddings, retrieval, or skill publication.

## Storage contract

```text
.repomind/memory/
  global/<memory_type>/<memory_id>.md
  user/<memory_type>/<memory_id>.md
  repo/<memory_type>/<memory_id>.md
```

Module, file, and symbol scopes use the `repo` bucket; their precise scope stays
in front matter. A memory ID has exactly one path. Changing its memory type or
top-level scope requires a new memory ID, preventing non-transactional moves and
ambiguous duplicate documents.

## Canonical Markdown format

Documents use YAML Front Matter with JSON-inline values. JSON-inline values are
valid YAML while providing deterministic key ordering, escaping, and parsing
without a YAML runtime dependency.

```markdown
---
schema_version: 1
memory_id: "mem_example"
memory_type: "semantic"
title: "Example repository fact"
status: "draft"
scope: {"files":[],"level":"repo","module":"","repo_id":"repo-1","symbols":[]}
source: {"archive_hash":"...","archive_path":"...","pipeline_version":"memory-v1","repo_revision":"...","task_id":"task-1"}
triggers: ["example"]
tags: ["memory"]
confidence: 0.7
evidence_strength: 0.5
created_at: "..."
updated_at: "..."
evidence: []
extensions: {}
---

# Example repository fact

## Knowledge
<!-- repomind:knowledge:start -->
The durable fact.
<!-- repomind:knowledge:end -->

## Applicability
<!-- repomind:applicability:start -->
When this fact should be used.
<!-- repomind:applicability:end -->
```

Evidence and invalidation use the same explicit section markers. Markers avoid
ambiguity when memory content contains ordinary Markdown headings. Reserved
RepoMind markers are rejected inside user content.

The codec guarantees:

- the same `MemoryDocument` produces identical UTF-8 bytes;
- structured evidence and its rendered section cannot silently diverge;
- title, schema, body sections, enums, scores, and required fields are checked;
- newer unsupported schemas fail explicitly;
- older schemas require a registered one-version-at-a-time migration;
- unknown fields within a supported version remain in `extensions`.

## Store behavior

`MarkdownMemoryDocumentStore` implements the storage-neutral
`MemoryDocumentStore` port.

- Writes use a flushed sibling temporary file followed by atomic replacement.
- Saving identical content performs no rewrite.
- Duplicate IDs and path-changing updates fail explicitly.
- Deprecation changes status to `deprecated` and records `deprecation_reason`; it does
  not delete or move the evidence-bearing document.
- Content hashes are calculated from canonical Markdown bytes.
- Full validation checks schema, canonical bytes, duplicate IDs, and expected
  paths.

## CLI

```text
lee-agent memory list
lee-agent memory show <memory_id>
lee-agent memory validate
lee-agent memory deprecate <memory_id> --reason "superseded"
```

These commands only load repository and memory-path configuration. They do not
start the Agent or require a valid LLM configuration.

## Phase 1 hardening included

Task Archive now receives `session_snapshot.json`, exported from the same
bounded snapshot builder used by online `prepare_turn()`. This prevents online
context and archived short-term context from developing separate semantics.

Task Archive remains the replay source. Markdown remains the authoritative
long-term knowledge expression. A later consolidation stage will transform the
former into the latter through the existing domain interfaces.
