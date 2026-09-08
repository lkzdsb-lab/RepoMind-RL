# Long-term Memory Phase 6: Semantic Consolidation

Phase 6 improves offline memory quality. It does not add online vector retrieval
or split Markdown documents into chunks.

## Pipeline

```text
Task Archive
  -> evidence-backed candidates
  -> extracted proposed memories
  -> exact knowledge hash matching
  -> knowledge embedding candidate discovery
  -> LLM pairwise relation assessment
  -> deterministic evolution policy
  -> canonical Markdown
  -> keyword and semantic projections
```

Markdown remains authoritative. `catalog.sqlite3` and `semantic.sqlite3` are
rebuildable projections.

## Identity and revisions

Memory schema 2 uses a source-stable ID derived from repository ownership,
archive hash, candidate ID, type, and scope. Knowledge is not part of the ID.
Each document stores a normalized `knowledge_hash` and an integer `revision`.

## Semantic merge modes

- `disabled`: exact matching only; no embedding calls.
- `observe`: record semantic matches and LLM decisions, but create the proposed
  memory instead of applying semantic mutations.
- `apply`: permit deterministic duplicate merges and refinements that pass the
  configured confidence and validation gates.

Similarity only selects comparison candidates. It never authorizes a merge.
Conflicts do not overwrite existing knowledge.

## Audit artifacts

Every consolidation-v3 run records `semantic_matches.json`,
`relation_resolutions.json`, and `memory_mutations.json` in addition to the
existing evidence, extraction, and final-document artifacts.

## Configuration

```json
{
  "llm": {
    "memory_relation": {}
  },
  "long_term_memory": {
    "pipeline_version": "consolidation-v3",
    "semantic_index_path": ".repomind/memory/semantic.sqlite3",
    "semantic_merge_mode": "observe",
    "semantic_top_k": 5,
    "semantic_min_similarity": 0.78,
    "semantic_relation_min_confidence": 0.85,
    "embedding": {
      "provider": "openai_compatible",
      "model": "<embedding-model>",
      "api_base": "https://<host>/v1",
      "api_key_env": "LLM_API_KEY",
      "dimensions": 0,
      "batch_size": 32
    }
  }
}
```

When semantic processing fails, exact matching and canonical Markdown writes
remain available, while the consolidation result is marked `partial`. Both
SQLite projections can be reconstructed from schema-2 Markdown documents.

## Incremental consolidation

The implementation removes the full Markdown load and semantic backfill from
the consolidation hot path. The changes were implemented in this order:

1. Add Catalog schema 3 with a composite exact-match index on type, repository,
   full scope, knowledge hash, status, and memory ID.
2. Load only exact-match and semantic Top K Markdown documents, validating their
   projection hashes/revisions. ID lookup probes the fixed scope/type paths
   instead of enumerating all Markdown files.
3. Register durable projection intents before saving Markdown, synchronize only
   the changed document, and retain failed work in `projection_pending.sqlite3`.
4. Retry at most 16 due projection items per consolidation. Failures wait 60
   seconds before another automatic retry. Each retry reads current Markdown,
   rather than applying an older queued document snapshot.

Catalog repair backlog blocks further matching, since an incomplete exact index
could create duplicate memories. Semantic failures produce partial results and
warnings. Successful projections are independently acknowledged. Processing a
candidate updates the Catalog immediately, so the next candidate can find it
without loading a historical document collection.

Full maintenance is explicit:

```text
lee-agent memory catalog sync
lee-agent memory catalog rebuild
lee-agent memory semantic rebuild
```

After enabling semantic processing for an existing memory collection, changing
the embedding model, or losing an index, rebuild it before relying on complete
semantic matching. A bounded startup diagnostic warns about a missing index or
a differing sampled model fingerprint; it is not a full coverage audit.
Manual Markdown edits require explicit synchronization/rebuild. Catalog schema
2 is not migrated automatically; rebuild the derived Catalog as schema 3.

The remaining cosine search scans the vectors within the requested scope in
Python. This change removes global Markdown synchronization costs but does not
introduce ANN search or claim constant-time vector retrieval.
