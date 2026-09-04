# Long-term Memory Phase 5: Agent Retrieval

Phase 5 connects the Phase 4 keyword Catalog to the online agent. It is a
read-only retrieval layer: task execution can discover and consume memory, but
cannot create, edit, promote, or delete canonical Markdown documents.

## Runtime flow

```text
task input
  -> rule-based memory query plan
  -> SQLite Catalog keyword recall
  -> status, repository, scope, score, and type-quota filters
  -> authoritative Markdown re-read and content-hash check
  -> bounded audience-specific context
  -> analyzer / skill selector / planner / code search / action policy
  -> usage events in AgentState and the immutable task archive
```

SQLite is only the rebuildable index. A Catalog hit is accepted only after its
Markdown document can be loaded and still matches the indexed content hash.
Missing or invalid Catalog data degrades to an empty memory context plus a
warning; it does not fail the task.

## Retrieval phases and cache

The agent may retrieve at three points:

- `bootstrap`: immediately after initial state creation, using the user task.
- `refined`: after task analysis, using normalized analysis, entities, and hints.
- `scope_refresh`: during execution when files, symbols, or failures add useful
  scope. Resume from user input follows the same fingerprint-based mechanism.

Each request has a deterministic fingerprint. An unchanged fingerprint and
Catalog revision reuses the current batch without database or document reads.
The configured refresh ceiling bounds repeated work in a long action loop.

## Query and selection policy

The rule-based planner creates a small set of explainable queries from task
title, description, analysis, entities, code-search hints, current files and
symbols, and runtime errors. Phase 5 does not use an LLM to decide whether a
memory exists; an LLM may consume the retrieved context but cannot bypass the
selection policy.

Selection applies these gates in order:

1. repository and declared scope visibility;
2. active status (`verified`, plus draft/review states when configured);
3. minimum keyword score;
4. per-memory-type quotas and global result limit;
5. Markdown integrity and the global character budget.

Every accepted or skipped hit retains its score and reason for diagnosis.

## Prompt audiences

One retrieval batch is rendered into audience-specific sections so prompt
builders share policy instead of duplicating filters:

| Audience | Memory types |
| --- | --- |
| Task analyzer | preference, semantic, episodic |
| Skill selector | preference, semantic, procedural |
| Planner / action policy | preference, semantic, procedural, anti-pattern |
| Code search | semantic, anti-pattern, episodic |

All sections label memory as historical guidance. Current repository reads,
tool output, and verification remain the authority for statements about the
current task. Observer, completion judgement, and final reporting do not receive
long-term memory directly, which keeps historical claims out of current-run
evidence.

## State and audit trail

`AgentState` holds planned queries, ranked hits, selected document projections,
rendered sections, warnings, Catalog revision, and refresh count. A separate
usage event is recorded only when a prompt or deterministic search component
actually consumes an audience section.

The task archive writes this information to `long_term_memory_usage.json`. The
offline consolidation pipeline does not treat that file as new task evidence,
so merely recalling a memory cannot reinforce or promote it.

## Ownership boundary

`LongTermMemoryReader` exposes only conditional retrieval. `DebugAgent` depends
on that interface rather than Catalog or document-store write APIs. In a future
multi-agent runtime, sub-agents may receive selected context or the same
read-only capability, while one coordinator owns archive finalization and the
separate offline consolidation writer. SQLite's locking remains a safety net,
not the authority model.

## Extension point

`retrieval_mode` is currently `keyword`. A later vector projection can implement
the same reader contract and reuse scope filters, Markdown validation, quotas,
rendering, state fields, and usage tracing. The canonical document and Agent
integration contracts therefore do not depend on a particular index engine.
