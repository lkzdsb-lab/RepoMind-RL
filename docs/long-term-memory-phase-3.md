# Long-term Memory Phase 3: Offline Consolidation

Phase 3 turns one immutable Task Archive into evidence-backed canonical Markdown
memory. It runs outside the online task-completion transaction, so consolidation
can be scheduled, retried, or replayed independently.

## Pipeline

```text
Task Archive
  -> integrity verification
  -> provenance-rich evidence bundle
  -> provenance-preserving candidate seeds
  -> exact evidence resolution
  -> semantic extraction (rule-based or LLM)
  -> deterministic type/scope/promotion policy
  -> exact-identity deduplication
  -> canonical Markdown documents
  -> Catalog synchronization
  -> immutable consolidation run
```

The extractor describes and classifies candidate knowledge. It does not decide
whether a claim is verified, set durable-memory confidence, invent evidence, or
widen scope. Those remain deterministic application-policy decisions.

## Candidate and evidence provenance

Evidence is collected only from manifest-covered artifacts: original user
statements, final report, verification results, diff, source snapshots,
trajectory, and runtime memory candidates.

Candidate seeds retain explicit archive references, event IDs, files, symbols,
and commands. `CandidateEvidenceResolver` uses only exact relationships:

- exact archive reference;
- identical source-event ID;
- identical normalized file or symbol;
- identical command.

Memory type and topical similarity never authorize evidence. The former
type-based behavior that attached every source/diff or verification in the task
has been removed.

Each immutable consolidation run stores `candidate_seeds`, resolved
`candidates`, and `evidence_resolution` diagnostics, so every final Evidence ID
has an explainable binding reason.

## Preference boundary

User-message importance controls online context retention only; it does not
create a preference. Original user statements become neutral candidate seeds.
The semantic extractor may classify a statement as task, session, repository,
or user scoped.

Only an explicit, durable repository/user preference with semantic confidence
of at least `0.75` can be stored. Task/session constraints are rejected. A
repository preference remains repository scoped; it is not promoted to a global
user preference.

Without an LLM, the rule extractor accepts only conservative, explicit durable
wording with an unambiguous repository or user scope. Ambiguous statements are
discarded rather than guessed.

## Promotion

Promotion in `consolidation-v2` is conservative:

- finished episodic summaries can be verified;
- semantic facts require directly related source/diff and passing verification;
- anti-patterns require a directly related failed verification;
- procedural memories require a directly related trajectory and passing
  verification;
- preferences require original `user_statement` evidence and the durable-scope
  decision described above.

LLM output cannot override these gates. Non-preference output cannot escape
repository scope, and file/symbol scope must be backed by candidate provenance.

## Replay, idempotency, and deduplication

Each run is stored under:

```text
.repomind/consolidation/runs/<task_id>/<pipeline_version>/
```

A valid existing run is returned instead of recomputing it. Missing Markdown can
be restored from the normalized run output. Exact memory identity is derived
from repository/user identity, memory type, scope, and normalized knowledge.
Repeated evidence and source task IDs are merged into the same document.

Semantic similarity merging and vector indexing are intentionally outside this
phase. Another module may later recall similar Memory IDs, but it does not alter
the evidence-authorization rules here.

## Extractors

`rule_based` needs no external service and is deliberately conservative. `llm`
uses the shared LLM adapter for semantic classification and wording, and falls
back to the rule extractor if the call fails. A fallback produces a `partial`
outcome and records a warning.

Configuration:

```json
{
  "llm": {
    "memory_extractor": {}
  },
  "long_term_memory": {
    "document_path": ".repomind/memory",
    "consolidation_path": ".repomind/consolidation/runs",
    "pipeline_version": "consolidation-v2",
    "extractor_mode": "rule_based",
    "max_candidates": 24
  }
}
```

Run one terminal-task archive:

```bash
lee-agent memory consolidate <task_id>
lee-agent memory consolidate <task_id> --extractor-mode llm
```

## Development schema boundary

Task Archive `memory_candidates.json` uses schema version 2 and newly written
manifests identify `archive-v2`. Legacy Task Archives are intentionally rejected
by `consolidation-v2`; no compatibility or type-based evidence inference is
performed during development.
