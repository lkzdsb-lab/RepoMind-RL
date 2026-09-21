# Local execution subagent

`execution_task` is an ordinary main-agent ToolSpec: input validation, permissions,
result normalization and state reduction use the existing tool infrastructure.
Its independent worker makes bounded decisions using the shared LLMJsonNode and
LLM client. System/user templates live in prompts/system/execution_specialist.md
and prompts/user/execution_specialist.md and use the existing template loader.

## Configuration and boundaries

```json
{
  "execution_enabled": true,
  "execution_timeout": 300,
  "execution_model_calls": 16,
  "execution_tool_calls": 32,
  "execution_retries": 8,
  "execution_startup_timeout": 30,
  "execution_operation_timeout": 30,
  "execution_cleanup_timeout": 5,
  "execution_resources": {
    "memory_limit_mb": 2048,
    "cpu_limit_percent": 50,
    "max_processes": 64,
    "mode": "strict"
  },
  "execution_env": {}
}
```

Disabled by default. No Docker, image preparation, dependency installation or
container helper is involved. Install project dependencies, including psutil for
listener ownership checks, through the normal project setup.
The specialist uses the resolved action-model configuration. Its SDK retries and
structured-output fallback are disabled so each budgeted call is one API attempt.
The configured endpoint must support the existing native structured-output API.

This is trusted local project execution, NOT an OS sandbox. Commands have the
current user's permissions and can access files/network or write source.
Tool restrictions and command guards reduce mistakes but do not confine arbitrary
project code. Do not use this mode for untrusted repositories.

Commands run in the actual target project, regardless of the Agent's launch
directory. cwd defaults to "."; a contained project subdirectory or "@work" is
accepted. Disposable output belongs in the task work directory.
Project commands receive a filtered runtime environment plus operator-supplied
execution_env. Configured model credential variable names are removed; the worker
gets its selected model credential separately. This is not a general secret scanner:
explicitly supplied environment values and local files still need operator review.
No credentials or execution_env values are written to request.json.

## Ownership and reuse

The worker selects tools, handles observations and decides whether to recover.
The Tool Host is a separate ordinary Python process hosting the restricted
ToolRegistry, ProcessManager and evidence collection. The parent Supervisor never
dispatches tools, scans source or validates the evidence log synchronously.
Worker and Host share the existing sequential mailbox; no additional LLM is added.
The main Agent still waits for the result, but slow tool I/O cannot block the
parent's deadline/cancellation checks.

Two reusable primitives live under tools/shell_tools:

* process.py: shared launch gate, managed output and synchronous
  command adapter. run_shell_command's non-shell path uses the same launch primitive.
  Its pre-existing explicit shell mode is unchanged.
* process_owner.py: Windows Job Object / POSIX process-group ownership. Native
  structure definitions and named constants are module-level; kernel32 signatures
  are bound once, on first use of the Windows path.
* local_http.py: bounded numeric-loopback HTTP without proxies or redirects.

ExecutionTools adapts these to ToolSpec, applies task policy and appends evidence.
The main and restricted registries share model/agent/tool_registry.py; the common
registry does not import application tools, avoiding circular imports.
It registers managed_process, local_request, read_file, list_files and read_evidence
in a restricted registry, not the main Agent's full editing/memory tool set.
The Host/worker exchange sequential tool_call.json and tool_reply.json envelopes
with sequence numbers and a Host-enforced call limit. These are local task
mailboxes, not an A2A protocol or another agent framework. A reply-delivery failure
never causes the same operation to be executed automatically again.

On Windows a parent-held TaskProcessScope owns an outer kill-on-close Job Object;
both Host and Worker attach before their launch gates open. All descendants inherit
the task Job. Inner per-command Jobs retain individual stop support. A stuck or
crashed Host cannot prevent the parent from terminating the complete task.
Launch-gate writes and process readiness have startup deadlines. Each Host operation
publishes a fixed deadline before dispatch; HTTP gets at most its requested timeout
plus a one-second delivery margin, capped by the operation budget. Background service
lifetime is governed by the task deadline, not the start operation timeout.
Source preparation and final verification run inside the supervised Host as well.
No tool-host restart or command replay occurs after a hard operation timeout.
Cleanup has a separate bounded budget after execution; unconfirmed exit is failure.

## Task-wide resource budgets

Windows enforces aggregate committed-memory, active-process and CPU hard limits via
the outer Job. CPU percentage means a fraction of total system CPU capacity; reaching
it throttles the task, rather than failing a valid busy build. Memory and process
limits deny further growth; observed Job limit notifications produce
RESOURCE_LIMIT_EXCEEDED with an inconclusive verdict. An unexplained crash is not
automatically labelled a memory violation. Resource peaks and capabilities are
reported in result.resources. OS notification delivery is not an infallible audit
log; the limits remain enforced even if a notification is not observed.
Budgets cover Worker, Host, launchers, virtualenv redirectors and all commands
together, not one allowance per launch. The supervisor is outside the limited Job.
Only trusted runtime configuration sets budgets; LLM tool input cannot raise them.

Strict mode refuses to start when hard limits cannot be installed. This version
implements hard limits on Windows only; Linux cgroup integration is not claimed.
Explicit best_effort uses psutil sampling if hard limits are unavailable: RSS,
discovered descendant count and sampled CPU. CPU excess terminates in this fallback,
not throttles. Results mark these capabilities monitored, never hard. Short-lived
or detached children may escape sampling; POSIX parent crashes are not contained.
Use strict mode when those gaps are unacceptable. No automatic privilege elevation.

These are trusted-project resource controls, not file/network isolation. No disk
quota, workspace confinement or Docker dependency is introduced.

## Tool input and behavior

```json
{
  "objective": "Start the project and verify its health endpoint",
  "context": "Read project startup instructions; use disposable data.",
  "acceptance_criteria": [
    {
      "criterion_id": "health",
      "description": "Health endpoint returns HTTP 200",
      "check": "http_status",
      "expected": "200"
    }
  ]
}
```

Workspace, paths, budgets and environment policy come from trusted runtime context,
not model arguments. Only one execution task is admitted per main process.

managed_process supports start/status/read_output/write_stdin/stop/list. Start uses
argv without shell interpolation and reuses identical argv/cwd until explicit stop.
There are at most eight launches per task. Output is continuously drained into a
bounded tail; cursor/next_cursor support incremental reads. output_complete means
an exited process's output has finished draining. partial/truncated output cannot
justify whole-output substring assertions. Stop cleans the owned command tree.

local_request requires process_id and port. psutil checks that the numeric IPv4
loopback listener belongs to that running task process or its descendants before
sending anything. Unowned/not-ready ports are rejected; unknown ownership fails
closed. Existing host services are not authorized by default. Ownership checking
reduces accidental cross-service access but is not a race-free OS security boundary.
The request has a deadline and bounded body; HTTP error statuses remain observations.
An uncertain write prevents subsequent writes; reads can investigate the outcome.

Source inspection uses the existing evidence filter (hidden/cache/evaluation and
credential-pattern files excluded). Commands themselves are not restricted to that
filter. Fingerprints before/after execution invalidate verdicts when included project
files change. They are provenance checks, not immutable execution snapshots.
The task does not create a source copy.

## Results, recovery and artifacts

execution_status and verdict remain separate: completed can mean a verified failure.
Every criterion is represented; pass/fail requires attempt-local runtime evidence.
The Host independently verifies IDs, evidence coverage, deterministic checks and
aggregation; the parent checks result identity and controls final cleanup status.
Natural-language observation criteria still involve model judgment.

Invalid arguments and invalid evidence references consume the shared recovery budget.
Transient model failures use bounded backoff; permanent failures stop.
Quiet running commands use backoff without treating quietness as failure.
Model/tool/recovery limits, repeated no-progress observations and the total deadline
bound execution. Whole tasks are not automatically restarted.
Cleanup failure or incomplete execution makes every criterion inconclusive.

Task artifacts under .repomind/execution/<attempt_id>/ are UTF-8:

* work/: disposable command output and runtime data.
* request.json: task, trusted paths, tool schemas, budgets and non-secret model config.
* tool_call.json / tool_reply.json: most recent sequential tool exchange.
* progress.json: progress and counters.
* decisions.jsonl: structured decisions, not private reasoning.
* evidence.jsonl: runtime observations with IDs, timestamp and source fingerprint.
* worker.log: worker diagnostics.
* host.log / host_status.json: tool execution diagnostics and fixed operation deadlines.
* finalize.json / host_result.json: final validation request and Host-validated result.
* excerpts.json: bounded evidence excerpts assembled by the Host.
* result.json: final result, resource capabilities/usage and parent cleanup status.

Artifacts are retained; no automatic retention policy or global evidence ledger is added.
Subagent results feed existing execution_results/command_results and compact summaries;
they do not silently satisfy the requirement to run a regression suite.

## Validation

No unit-test files were added. Windows runtime smoke checks exercised real owned
processes, stdin, separate stdout/stderr, nonzero exit, service startup, owned-port
HTTP, argument rejection and port release after cleanup.
A local scripted model endpoint drove the actual worker, shared LLM node, tool
registry, local command runtime, evidence validation and final cleanup end to end.
Cancellation, deadline and forced worker-crash checks after service startup also
released service ports and the execution admission lock.
Boundary checks covered actual working directories, credential filtering, configured
environment values, synchronous timeout output, and uncertain HTTP write replay guards.
These checks establish runtime plumbing, not real-model task-solving quality.
POSIX behavior requires validation on a POSIX host; no Docker checks are applicable.

The independent-Host refactor was checked with a real Host, nested Windows Jobs,
normal finalization, slow HTTP response headers exceeding the operation deadline,
and cancellation while the HTTP tool was blocked. Cleanup released the services'
listening ports. These are bounded manual diagnostics, not newly added unit tests.
Small-budget checks observed memory/process-limit notifications and CPU hard-cap
throttling. A local scripted model endpoint also exercised the refactored real
Worker -> Host -> evidence validation -> Supervisor cleanup path end to end.
