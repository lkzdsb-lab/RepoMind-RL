# Long-term Memory: Phase 0 and Phase 1

This document fixes the architecture and implementation boundary for the first
two phases of RepoMind long-term memory. RL, Markdown consolidation, catalogs,
embeddings, retrieval, and skill promotion are intentionally outside this
change.

## Stable layering

The complete system uses four durable layers:

1. Task Archive: immutable evidence from completed runs and the replay source
   for offline consolidation.
2. Markdown Memory: reviewed, versioned long-term knowledge produced from task
   archives.
3. SQLite Catalog: a disposable projection for metadata, FTS, jobs, and
   consistency tracking.
4. Vector Index: a disposable semantic-search projection.

SQLite and vector indexes must be rebuildable from Markdown. Markdown must be
recomputable from Task Archive, although LLM-based reconsolidation is only
semantically reproducible unless normalized consolidation output is retained.

## Phase 0: domain contracts

The storage-neutral contracts live in `agent_runtime/memory/domain/`.

- `MemoryDocument` is the future canonical long-term memory representation.
- `MemoryScope`, `MemoryEvidence`, and `MemorySource` make applicability and
  provenance explicit.
- `MemoryQuery`, `MemoryHit`, and `MemoryRelation` define retrieval without
  selecting a database.
- `TaskArchiveManifest` and `ArchiveFileRecord` version and verify archives.
- Protocols isolate the runtime from Markdown, SQLite, vector, and skill
  adapters.

Schema versions reject incompatible data rather than silently accepting it.
Unknown future MemoryDocument fields are retained in `extensions` so readers
can remain forward-compatible within a supported schema version.

## Phase 1: immutable Task Archive

Each terminal task is stored at:

```text
.repomind/archive/tasks/<task_id>/
  manifest.json
  events.jsonl
  final_state.json
  final_report.json
  verification.json
  memory_candidates.json
  session_snapshot.json
  source_snapshots.json
  diff.patch                 # only when a diff exists
```

The writer creates a sibling temporary directory, flushes every artifact, writes
the manifest last, and atomically renames the directory into place. Existing
valid task IDs are returned unchanged; invalid existing archives are never
overwritten.

`manifest.json` records the archive schema, repository identity and revision,
workspace fingerprint, pipeline version, and the byte size and SHA-256 of every
artifact. `TaskArchiveStore.verify()` detects missing or changed artifacts.

The archive is written before session memory is committed. An archive failure
does not change the completed task result, but the state exposes
`task_archive_status=failed` and `task_archive_error`; failed archives are not
eligible for future consolidation.

## Evidence and safety rules

- Candidate, edited, and read files are captured within per-file and total byte
  budgets.
- Paths outside the repository are rejected.
- `.env`, credentials, private keys, and certificate/key containers are never
  source-snapshotted.
- Structured secrets and common bearer/API-key text patterns are redacted before
  archive artifacts are encoded.
- Oversized strings remain valid JSON and carry an explicit truncation marker.
- The repository origin is hashed for identity and is not written in clear text.

Task Archive is evidence, not trusted knowledge. Later consolidation stages must
still validate every candidate before producing verified Markdown memory.

`session_snapshot.json` is exported through the same bounded snapshot builder
used by `SessionMemoryService.prepare_turn()`. It captures the topic, recent
turns, playbook, and active short-term memory layers that preceded the current
task; the current task remains represented by final state and events.

## Configuration

```json
{
  "archive": {
    "enabled": true,
    "path": ".repomind/archive/tasks",
    "max_text_chars": 200000,
    "max_source_file_bytes": 200000,
    "max_source_total_bytes": 2000000
  }
}
```

All relative archive paths are resolved against the target repository, matching
the existing trace, memory, log, code-index, and RL artifact isolation policy.

## Next implementation boundary

Phase 2 is implemented by the deterministic Markdown codec and
`MarkdownMemoryDocumentStore`. See
[`long-term-memory-phase-2.md`](long-term-memory-phase-2.md).
