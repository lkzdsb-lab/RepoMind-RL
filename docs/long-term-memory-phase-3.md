# Long-term Memory Phase 3: Offline Consolidation

Phase 3 turns one immutable Task Archive into evidence-backed canonical Markdown
memory. It is deliberately outside the online task completion transaction:
archiving and session-memory commit remain the durable online boundary, while
consolidation can be scheduled, retried, or replayed later.

## Pipeline

```text
Task Archive
  -> integrity verification
  -> normalized evidence bundle
  -> bounded rule-based candidates
  -> semantic extraction (rule-based or LLM)
  -> deterministic evidence/scope/promotion policy
  -> exact-identity deduplication
  -> canonical Markdown documents
  -> immutable consolidation run
```

The extractor describes candidate knowledge, but it does not decide whether a
claim is verified, set confidence, invent evidence, or widen scope. Those are
deterministic application-policy decisions. This keeps the durable result
auditable even when an LLM extractor is enabled.

## Evidence and promotion

Evidence is collected only from manifest-covered archive artifacts: final
report, verification results, diff, source snapshots, trajectory, and runtime
memory candidates. Every extracted memory must refer to candidate and evidence
IDs from that normalized bundle.

Promotion in `consolidation-v1` is conservative:

- finished episodic summaries can be verified;
- semantic facts require source/diff evidence and a passing verification;
- anti-patterns require failed verification evidence;
- preferences require an explicit `user_constraint` runtime candidate;
- procedural memories remain draft for later cross-task validation.

LLM output cannot override these rules. Non-preference output also cannot widen
itself beyond repository scope.

## Replay and idempotency

Each run is stored under:

```text
.repomind/consolidation/runs/<task_id>/<pipeline_version>/
```

The run contains the evidence bundle, candidates, raw normalized extraction,
generated documents, outcome, and a checksum manifest. A valid existing run is
returned instead of recomputing it. If its Markdown document is missing, the
document can be restored from the recorded normalized output.

Changing extraction or promotion behavior requires a new pipeline version. This
preserves old results and makes re-consolidation explicit rather than silently
rewriting history.

Exact duplicate identity is derived from repository/user scope, memory type,
scope, and normalized knowledge. Repeated evidence is merged into the same
document. Semantic similarity merging is intentionally deferred until a later
catalog/retrieval phase; Phase 3 never performs an unsafe fuzzy overwrite.

## Extractors

`rule_based` is the stable default and needs no external service. `llm` uses the
shared OpenAI-compatible LLM adapter and falls back to the rule extractor if the
call fails. A fallback produces a `partial` outcome and records a warning in the
immutable run.

Configuration:

```json
{
  "llm": {
    "memory_extractor": {}
  },
  "long_term_memory": {
    "document_path": ".repomind/memory",
    "consolidation_path": ".repomind/consolidation/runs",
    "pipeline_version": "consolidation-v1",
    "extractor_mode": "rule_based",
    "max_candidates": 24
  }
}
```

Run one completed archive:

```bash
lee-agent memory consolidate <task_id>
lee-agent memory consolidate <task_id> --extractor-mode llm
lee-agent memory consolidate <task_id> --pipeline-version consolidation-v2
```

Phase 3 does not add a background worker, queue, retrieval ranking, vector index,
or skill publication. Its output and immutable run record are the stable inputs
for those later phases.
