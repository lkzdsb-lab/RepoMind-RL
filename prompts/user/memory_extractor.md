Evaluate the following archive candidates for long-term storage.

## Task context (orientation only; not candidate evidence)

{{ task_context }}

## Candidate packets

Each packet contains one proposed candidate and that candidate's complete
evidence allow-list. Evaluate packets independently. Do not use task context or
another packet as proof.

{{ candidate_packets }}

## Required response

Return a structured `memories` array with exactly one object per candidate
packet, in the same order. Preserve every `candidate_id` exactly.

For every object provide:

- `candidate_id`, `should_store`, `memory_type`
- `title`, `knowledge`, `applicability`, `invalidation`
- `evidence_ids`, containing only IDs from that packet's `allowed_evidence`
- `scope_level`, `module`, `files`, `symbols`, `triggers`, and `tags`
- `decision_reason` and `rejection_reason`
- `constraint_scope`, `constraint_durability`, `constraint_explicit`, and
  `semantic_confidence`

If rejected, set `should_store=false`, use the most specific permitted
`rejection_reason`, and keep unsupported content out of `knowledge`. If accepted,
set `rejection_reason` to an empty string. For non-preference types, leave
constraint scope/durability empty, set constraint_explicit=false, and set
semantic_confidence=0.
