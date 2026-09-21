# Scope-aware online recall

Online scope policy `scope-v2` separates retrieval relevance from applicability.
It changes no Markdown schema, Catalog schema, vector data, or consolidation's
exact scope equality. No index rebuild is needed for this policy change.

## Shared context

`ScopeContextBuilder` is called once per refresh check. Its immutable result is
shared by the query planner, revision fingerprint, and matcher. It records:

- Current repository ID.
- Explicit candidate paths verified against the current filesystem, preserving
  case, normalizing separators and rejecting paths outside the repository.
- Directory ancestors of confirmed paths (not guessed package names).
- File-qualified symbols from selected code context, falling back to retrieved
  code context. The existing symbol record's file and line are rechecked for its
  name. This is a bounded location check, not an AST or semantic proof.
- Unverified entities/search hints and ambiguous filenames/symbol names.

No recursive repository scan or extra LLM request is used. Only supplied paths
and up to 100 symbol records/locations are examined; symbol lines above 10000 are
left unconfirmed. Directory listings are reused within one context build.

The `scope` query label remains, but online queries no longer pass mixed entities
as `scope_hints` SQL requirements. Confirmed locations enrich the query text;
unverified names remain text hints. Catalog's explicit facet filtering API is
unchanged for other callers. Both keyword and semantic candidates use the same
post-retrieval matcher.

## Decisions and selection

- `matched`: same-repository repo scope, user/global scope, or an exact confirmed
  file/directory/file-qualified symbol. Directory matching respects boundaries.
- `unknown`: insufficient or ambiguous location information. Absence from the
  candidate list does not establish that a memory is irrelevant.
- `mismatched`: different repository or unsupported scope.

Confirmed candidates are selected before unknown candidates, so unknowns cannot
consume their type/global/context budgets. Scores are multiplied by scope
weights, rather than adding incomparable constants to keyword/cosine/RRF scores.
Unknown preference memories are excluded. Other unknown memories have a separate
quota and an explicit warning before their knowledge text in every audience's
rendered context: use as search leads only, verify the referenced location first.
Existing status/hash checks, type quotas, overall limit and rendering budget apply.

Configure these keys in `long_term_memory`:

```json
{
  "scope_matched_weight": 1.1,
  "scope_unknown_weight": 0.5,
  "scope_unknown_limit": 2
}
```

Set `scope_unknown_limit` to zero to exclude unconfirmed memories entirely.
Scope context, policy version and settings participate in refresh fingerprints.
Hit records and executor events include scope status/reason, match source,
pre-adjustment score/rank, adjusted rank and unknown-quota rejection reasons.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m evaluation --case evaluation/cases/memory_scope.json --config config.json
```

The deterministic keyword fixture covers unknown initial scope, verified paths,
basename ambiguity, matching and same-name/different-file symbols, directory
boundaries and cross-repository/unconfirmed-preference exclusion. It does not
require embedding. Add `semantic` and `hybrid` to the case's modes to compare a
configured real embedding model on the same cases.

Queries can declare `forbidden_memory_ids`, checked against ALL injected memory
IDs, not just the Recall@K prefix. They may also specify `expected_scope_status`
for isolated scope assertions, evaluated independently of candidate retrieval.
These boundary failures fail the run; they are not additional quality metrics.
Recall@K remains based on final returned memory IDs, including explicitly marked
unknown search leads. Scope assertions in reports distinguish those from matched
memories. A perfect score on this tiny fixture is not evidence of production
quality or LLM compliance with the warning.
