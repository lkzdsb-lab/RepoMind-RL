# Online knowledge-vector recall

The default remains `keyword`. `semantic` uses knowledge embeddings only;
`hybrid` merges keyword and semantic channel rankings with weighted RRF.
This does not change consolidation's exact-scope matching or add chunks/ANN.

Configure the existing `long_term_memory` section (merge these keys into your
configuration; do not replace your other settings):

```json
{
  "retrieval_mode": "hybrid",
  "semantic_candidates": 24,
  "semantic_min_score": 0.5,
  "semantic_timeout": 5.0,
  "query_cache_size": 256,
  "rrf_k": 60,
  "keyword_weight": 1.0,
  "semantic_weight": 1.0,
  "embedding": {
    "provider": "openai_compatible",
    "model": "YOUR_EMBEDDING_MODEL",
    "api_base": "YOUR_COMPATIBLE_ENDPOINT",
    "api_key_env": "MEMORY_EMBEDDING_API_KEY",
    "dimensions": 0,
    "batch_size": 32,
    "timeout": 60
  }
}
```

Set the named environment variable and run the existing `memory semantic rebuild`
maintenance command before enabling semantic recall. Online lookup is read-only:
it never creates, migrates, or repairs the vector database. Changing the model
requires an offline rebuild. Missing/partial projections can reduce recall.

## Execution

The existing bootstrap/refined/resume/scope_refresh triggers remain unchanged.
Queries are deduplicated and embedded in batches, with a bounded, per-service LRU
cache keyed by normalized text and model fingerprint. Keyword lookup runs first;
semantic work has an independent caller deadline, disabled SDK retries, and at
most one daemon worker per service. Late results are discarded. A stuck provider
cannot accumulate workers; subsequent refreshes report busy until it finishes.
The deadline bounds waiting, not forced cancellation of third-party HTTP code.

Each channel merges its subqueries by maximum score. Markdown is read once per
candidate to validate channel-specific content/version metadata. Invalid channel
contributions are discarded before RRF. Keyword thresholds and semantic cosine
thresholds remain separate; the keyword threshold is not applied to RRF scores.
The state/scope checks, quotas, context budget, and audience sections remain
shared. Scope now uses the three-state policy described in
`long-term-memory-scope-retrieval.md`: unconfirmed technical memories may enter
as bounded search leads, while unconfirmed preferences are excluded.

Online vector lookup accepts current-repository and user/global records, optional
types and statuses; it does not require identical full scope. SQL prefilters and
Python cosine comparisons are exact, linear in the eligible corpus, with bounded
retained Top K metadata. No ANN acceleration is claimed.

Diagnostics include channel rank/score, elapsed time, cache hits, scanned vectors,
warnings and timeouts. In hybrid mode the surviving channel remains usable.
Semantic-only failures never silently invoke keyword retrieval. Context exposure
does not prove that an LLM adopted a memory.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m evaluation --case evaluation/cases/memory_recall.json --config config.json
```

This entry uses the production service and isolated Markdown/Catalog/vector
copies. `$current` repository IDs are remapped to the isolated workspace.
It compares all three modes on the same K, corpus, quotas and context budget.
Recall is macro-averaged across query gold ID sets, using the final budgeted
`batch.memories`. Missing embeddings/provider failures are reported and fail the
run rather than disappearing from the average. No LLM judges the gold labels.

The included six-document fixture is a small illustrative regression dataset,
not a statistically representative quality benchmark. Its minimum recall is zero
until a model-specific baseline is established. Add reviewed real examples before
making production quality claims. Workspaces and corpus hashes are retained for
inspection; reports, per-query traces and effective settings are under the
existing `.repomind/evaluation/<case>/<run>/` hierarchy.
