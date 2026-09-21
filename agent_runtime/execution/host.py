"""Disposable tool execution host. No model client or credential is needed here."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import time

from .artifacts import EvidenceStore, fingerprint, read_json, write_json
from .backend import ProcessManager
from .contracts import Decision, ExecutionResult, ExecutionTaskInput, Limits
from .tools import ExecutionTools
from .transport import ToolServer
from .validation import aggregate_verdict, validate_results


def run(directory: Path):
    """ 执行工具、管理命令、记录证据服务、编译器等命令进程及其后代"""
    backend = None
    try:
        request = read_json(directory / "request.json")
        limits = Limits.model_validate(request["limits"])
        root = Path(request["workspace"])
        task = ExecutionTaskInput.model_validate(request["task"])
        deadline = request["deadline"]
        write_json(directory / "host_status.json", {
            "phase": "preparing", "deadline": min(deadline, time.monotonic() + limits.operation_timeout)})
        source = fingerprint(root, deadline=deadline)
        backend = ProcessManager(root, Path(request["work_directory"]), deadline, dict(os.environ))
        evidence = EvidenceStore(directory, source)
        tools = ExecutionTools(backend, root, evidence)
        request.update(source_fingerprint=source, tools=tools.manifest())
        write_json(directory / "request.json", request)
        server = ToolServer(directory, tools, limits.tool_calls, limits.operation_timeout)
        write_json(directory / "host_status.json", {"phase": "ready", "deadline": 0})
        while not (directory / "finalize.json").exists():
            backend.remaining()
            server.poll()
            time.sleep(0.05)
        # 接受到结束，并开始校验结果
        write_json(directory / "host_status.json", {
            "phase": "validating", "deadline": min(deadline, time.monotonic() + limits.operation_timeout)})
        completed = ExecutionResult.model_validate(read_json(directory / "result.json"))
        if (completed.task_id, completed.attempt_id, completed.source_fingerprint) != (
                request["task_id"], request["attempt_id"], source):
            raise ValueError("Worker result identity mismatch")
        if completed.execution_status != "completed" and completed.verdict != "inconclusive":
            raise ValueError("Incomplete execution cannot claim a verification verdict")
        if completed.execution_status == "completed":
            validate_results(task, Decision(action="finish", criteria_results=completed.criteria_results), evidence)
            completed.verdict = aggregate_verdict(completed.criteria_results)
        if fingerprint(root, deadline=deadline) != source:
            completed.execution_status = "failed"
            completed.verdict = "inconclusive"
            completed.summary = "Project files changed during verification; observations do not prove current source"
            for item in completed.criteria_results:
                item.verdict = "inconclusive"
        write_json(directory / "host_result.json", completed.model_dump())
        selected = {key for item in completed.criteria_results for key in item.evidence_ids}
        excerpts = []
        for key, record in evidence.records.items():
            if key in selected and len(excerpts) < 12:
                observation = record["result"]
                excerpts.append({"evidence_id": key, "operation": record["operation"],
                                 "exit_code": observation.get("exit_code"),
                                 "status_code": observation.get("status_code"),
                                 "excerpt": str(observation.get("body", observation.get("output", "")))[:800]})
        write_json(directory / "excerpts.json", {"items": excerpts})
        write_json(directory / "host_status.json", {"phase": "done", "deadline": 0})
    except BaseException as exc:
        write_json(directory / "host_status.json", {"phase": "failed", "error": f"{type(exc).__name__}: {exc}"[:2000],
                                                    "deadline": 0})
        raise
    finally:
        # Parent task scope remains the fallback even if this cleanup blocks.
        if backend is not None:
            backend.cleanup()


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
