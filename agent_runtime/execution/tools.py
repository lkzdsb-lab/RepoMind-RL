"""Restricted tools, evidence production and side-effect-aware failure handling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading

from pydantic import ValidationError

from model.agent.tools import ToolSpec, tool_spec_prompt_dict
from model.agent.tool_registry import ToolRegistryBase
from tools.shell_tools.local_http import LocalHttpClient, failure
from .artifacts import EvidenceStore, source_files
from .backend import BackendUnavailable, ProcessManager
from .contracts import EvidenceInput, ListInput, ProcessInput, ReadInput, RequestInput


class ExecutionTools:
    schemas = {"managed_process": ProcessInput, "local_request": RequestInput, "read_file": ReadInput,
               "read_evidence": EvidenceInput, "list_files": ListInput}

    def __init__(self, backend: ProcessManager, source: Path, evidence: EvidenceStore):
        self.backend, self.source, self.evidence = backend, source, evidence
        self.processes = backend.processes
        self.http = LocalHttpClient()
        self.registry = ToolRegistryBase()
        for name, schema in self.schemas.items():
            self.registry.register(ToolSpec(name=name, description=f"Execution specialist: {name}",
                input_schema=schema, permissions=["execution:local"],
                runner=lambda repo, args, name=name, schema=schema: self._invoke(name, schema.model_construct(**args))))
        self.launch_keys: dict[str, str] = {}
        self.uncertain_writes: set[str] = set()
        self.uncertain_stdin: set[str] = set()

    def dispatch(self, name: str, arguments: dict) -> dict:
        result = self.registry.run(name, str(self.source), arguments, allowed_permissions=["execution:local"])
        if not isinstance(result.get("error", {}), dict):
            detail = result.get("error", "Invalid tool call")
            if result.get("validation_errors"):
                detail += ": " + json.dumps(result["validation_errors"], ensure_ascii=False)
            result = failure("INVALID_ARGUMENT", detail,
                             category="validation", retryable=True)
        # Common ToolSpec normalization is kept at the boundary, not duplicated in evidence.
        result = {key: value for key, value in result.items() if key not in {"data", "status", "message", "artifacts"}}
        if name == "read_evidence" and result.get("ok"):
            return result
        return self.evidence.append(name, arguments, result)

    def manifest(self):
        return [tool_spec_prompt_dict(spec) for spec in self.registry.items().values()]

    def _invoke(self, name, args):
        try:
            self.backend.remaining()
            if name == "list_files":
                files = [p.relative_to(self.source).as_posix() for p in source_files(self.source, self.backend.deadline)]
                result = {"ok": True, "files": files[:400], "truncated": len(files) > 400}
            else:
                if name == "read_evidence":
                    # Reuse the original ID, without recursively persisting copies of evidence.
                    return {"ok": True, "record": self.evidence.records[args.evidence_id],
                            "evidence_ids": [args.evidence_id]}
                result = getattr(self, name)(args)
        except (ValidationError, ValueError, KeyError) as exc:
            result = failure("INVALID_ARGUMENT", str(exc), category="validation", retryable=True)
        except BackendUnavailable as exc:
            result = failure("BACKEND_FAILED", str(exc))
        except PermissionError as exc:
            result = failure("PERMISSION_DENIED", str(exc), category="permission")
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            result = failure("OPERATION_TIMEOUT", str(exc), effects="unknown")
        except OSError as exc:
            result = failure("IO_ERROR", str(exc), effects="unknown")
        return result

    def read_file(self, args: ReadInput) -> dict:
        path = (self.source / args.path).resolve()
        if not path.is_relative_to(self.source.resolve()) or not path.is_file():
            raise ValueError("Read requires a file inside the project")
        if path not in set(source_files(self.source, self.backend.deadline)):
            raise PermissionError("File excluded by the project evidence filter")
        if path.stat().st_size > 5_000_000:
            raise ValueError("File exceeds 5 MB")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"ok": True, "path": args.path,
                "content": "\n".join(lines[args.start_line - 1:args.start_line + 199])[:12000]}

    def managed_process(self, args: ProcessInput) -> dict:
        if args.action == "list":
            return {"ok": True, "processes": {key: proc.snapshot() for key, proc in self.processes.items()}}
        if args.action == "start":
            if not args.argv or not args.argv[0] or any("\x00" in arg or len(arg) > 8000 for arg in args.argv):
                raise ValueError("start requires a nonempty argv array without null bytes")
            launch_key = json.dumps([args.argv, args.cwd])
            if launch_key in self.launch_keys:
                previous_id = self.launch_keys[launch_key]
                previous = self.processes[previous_id].snapshot()
                # Explicit stop is required before launching identical arguments again.
                return {"ok": True, "effects": "none", "process_id": previous_id, "reused": True, **previous}
            if len(self.processes) >= 8:
                return failure("PROCESS_LIMIT", "At most 8 process launches per task", category="budget")
            process_id = f"process-{len(self.processes) + 1}"
            proc = self.backend.start(process_id, args.argv, args.cwd)
            self.launch_keys[launch_key] = process_id
            return {"ok": True, "effects": "applied", "process_id": process_id, **proc.snapshot()}
        proc = self.processes.get(args.process_id)
        if proc is None:
            raise ValueError("Unknown process_id for this attempt")
        if args.action == "stop":
            proc.close()
            self.launch_keys = {key: value for key, value in self.launch_keys.items() if value != args.process_id}
        elif args.action == "write_stdin":
            if args.process_id in self.uncertain_stdin:
                return failure("UNSAFE_REPLAY", "Previous stdin delivery is uncertain", category="permission", effects="unknown")
            if proc.process.poll() is not None:
                raise ValueError("Cannot write to an exited process")
            # A pipe can block indefinitely; use a bounded writer and abandon it on timeout.
            errors = []
            def write():
                try:
                    proc.process.stdin.write(args.text.encode("utf-8"))
                    proc.process.stdin.flush()
                except (OSError, ValueError) as exc:
                    errors.append(exc)
            thread = threading.Thread(target=write, daemon=True)
            thread.start()
            thread.join(self.backend.remaining(2))
            if thread.is_alive():
                self.uncertain_stdin.add(args.process_id)
                return failure("STDIN_TIMEOUT", "Input delivery uncertain; do not repeat", effects="unknown")
            if errors:
                self.uncertain_stdin.add(args.process_id)
                raise errors[0]
        return {"ok": True, "process_id": args.process_id, **proc.snapshot(args.cursor)}

    def local_request(self, args: RequestInput) -> dict:
        if not args.path.startswith("/") or args.path.startswith("//") or any(c in args.path for c in "\r\n"):
            raise ValueError("path must be an origin-relative HTTP path")
        key = f"{args.port}:{args.method}:{args.path}"
        if self.uncertain_writes and args.method not in {"GET", "HEAD"}:
            return failure("UNSAFE_REPLAY", "Previous write outcome unknown; inspect state and end inconclusive if unresolved",
                           category="permission", effects="unknown")
        if not self.backend.owns_listener(args.process_id, args.port):
            return failure("SERVICE_NOT_READY_OR_NOT_OWNED",
                           "Port is not listening under the specified running task process; inspect startup logs",
                           category="transient", retryable=True)
        try:
            values = args.model_dump(exclude={"process_id", "timeout"})
            result = self.http.request(**values, timeout=self.backend.remaining(args.timeout))
        except (TimeoutError, subprocess.TimeoutExpired, BackendUnavailable, OSError, ValueError) as exc:
            result = failure("REQUEST_UNCERTAIN", str(exc), effects="unknown",
                             retryable=args.method in {"GET", "HEAD"})
        if args.method not in {"GET", "HEAD"} and result.get("effects") == "unknown":
            self.uncertain_writes.add(key)
        return result

    def close(self):
        self.backend.cleanup()
