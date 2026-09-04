# Long-term Memory Phase 4: SQLite Catalog and Keyword Retrieval

Phase 4 adds a fast, local projection over canonical Markdown memory. Markdown
remains authoritative. The Catalog contains no information that cannot be
recreated by reading the documents again.

## Data flow

```text
MarkdownMemoryDocumentStore
  -> SQLiteMemoryCatalog.synchronize
  -> memories + memory_facets + memory_fts
  -> keyword_search(MemoryQuery)
  -> deterministic MemoryHit ranking
```

Consolidation writes Markdown first and then synchronizes the Catalog. A Catalog
failure marks a new consolidation result partial but does not roll back or
invalidate the canonical document. Replaying an existing consolidation run also
restores missing Markdown and refreshes its Catalog projection.

## Schema and compatibility

The Catalog uses three logical structures:

- `memories`: one structured metadata row per memory document;
- `memory_facets`: one reusable table for tag, trigger, file, symbol, and source
  task values;
- `memory_fts`: an FTS5 projection of searchable text and normalized tokens.

`PRAGMA user_version` controls schema migration. `catalog_metadata` records the
tokenizer and ranking versions. A mismatched version refuses incremental writes
and search and asks for a rebuild, preventing a mixed index produced by two
different algorithms.

Every row records the Markdown-relative path and SHA-256 content hash. Catalog
status checks compare those values to the current document store and also report
missing rows, orphans, missing/duplicate FTS rows, version mismatches, and SQLite
integrity failures.

SQLite connections are short lived. Writes use `BEGIN IMMEDIATE`, foreign keys,
a busy timeout, full synchronous durability, and WAL mode. Document metadata,
facets, and FTS replacement are committed in one SQLite transaction.

## Portable tokenization

The built-in FTS5 `unicode61` tokenizer is supplemented by a deterministic
application tokenizer. It preserves code-oriented components, splits paths,
snake_case and CamelCase, and emits Chinese characters plus adjacent bigrams.
Indexing and queries share the same implementation.

This avoids a runtime dependency on optional SQLite segmentation extensions.
Changing the algorithm requires a new tokenizer version and a Catalog rebuild.
The Python SQLite build must include FTS5; initialization fails explicitly when
that standard feature is unavailable instead of silently degrading search.

## Search policy

The first candidate set is ordered by weighted FTS5 BM25 fields. SQL applies
exact filters before application ranking:

- repository/global/user visibility;
- memory type;
- requested status, or active statuses by default;
- required tags;
- exact module/file/symbol/trigger hints.

Application ranking then uses FTS rank, exact title matches, title/facet token
matches, verified status, confidence, and evidence strength. `MemoryHit.reasons`
records the explainable factors. Deprecated, stale, and rejected memories are
excluded unless explicitly requested.

The Catalog does not perform embedding generation, semantic similarity, fuzzy
memory merging, or LLM reranking. Those remain separate, rebuildable projections
for later phases.

## Synchronization and recovery

Incremental synchronization compares content hashes, updates changed documents,
skips unchanged documents, and removes projections whose Markdown was deleted.

Full rebuild writes a new temporary SQLite database, indexes every valid Markdown
document, runs `PRAGMA integrity_check`, checkpoints the old database, and only
then atomically replaces the Catalog path. Invalid Markdown stops a CLI rebuild
before replacement.

Commands:

```bash
lee-agent memory catalog status
lee-agent memory catalog sync
lee-agent memory catalog rebuild
lee-agent memory search "archive integrity"
lee-agent memory search "归档完整性" --type semantic --status verified
lee-agent memory search "TaskArchiveStore" --scope agent_runtime/memory/archive/store.py
```

Configuration:

```json
{
  "long_term_memory": {
    "document_path": ".repomind/memory",
    "catalog_path": ".repomind/memory/catalog.sqlite3",
    "catalog_busy_timeout_ms": 5000,
    "keyword_candidate_multiplier": 8
  }
}
```
