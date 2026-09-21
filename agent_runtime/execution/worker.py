"""Bounded specialist decisions; the Tool Host executes local tools."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from pydantic import ValidationError

from config import LLMConfig

from .artifacts import EvidenceStore, read_json, write_json
from .backend import Deadline
from .budget import BudgetExceeded, RecoveryBudget
from .contracts import Decision, ExecutionError, ExecutionResult, ExecutionTaskInput, Limits
from .transport import ToolClient
from agent_runtime.llm.llm_nodes import LLMJsonNode
from prompts.templates import load_prompt, render_prompt
from .validation import aggregate_verdict, validate_results


def run(directory: Path) -> ExecutionResult:
    request = read_json(directory / "request.json")
    task = ExecutionTaskInput.model_validate(request["task"])
    limits = Limits.model_validate(request["limits"])
    result = ExecutionResult(task_id=request["task_id"], attempt_id=request["attempt_id"],
                             execution_status="failed", source_fingerprint=request["source_fingerprint"])
    evidence = EvidenceStore(directory, request["source_fingerprint"])
    backend = Deadline(request["deadline"])
    tools = ToolClient(directory, backend, evidence)
    counters = {"model_calls": 0, "tool_calls": 0, "retries": 0, "total_tokens": 0}
    recovery = RecoveryBudget(limits.retries, counters, backend.remaining)
    last_signature = ""
    repeat_count = 0
    history: list[dict] = []
    try:
        llm_config = LLMConfig(**request["llm"])
        state = {"task_id": request["task_id"]}
        node = LLMJsonNode(
            name="execution_specialist", llm_config=llm_config,
            system_prompt=load_prompt("system/execution_specialist.md"),
            build_prompt=lambda state, context: render_prompt("user/execution_specialist.md",
                payload=json.dumps(context, ensure_ascii=False)),
            fallback=None, response_model=Decision, raise_on_error=True)
        # 下面 subagent 对 llm 的访问受总次数的限制
        for _ in range(limits.model_calls):
            backend.remaining()
            counters["model_calls"] += 1
            write_json(directory / "progress.json", {"message": "Execution specialist running", **counters})
            try:
                data = node.run(state, {
                    "task": task.model_dump(), "budgets": limits.model_dump(), "used": counters,
                    "workspace": request["workspace"], "work_directory": request["work_directory"],
                    "tools": request["tools"], "recent_operations": history[-8:],
                    "evidence_inventory": [{"id": key, "operation": value["operation"],
                        "ok": value["result"].get("ok")} for key, value in evidence.records.items()],
                }, publish_update=False)
                counters["total_tokens"] = state.get("llm_token_usage", {}).get("total_tokens", 0)
                decision = Decision.model_validate({key: value for key, value in data.items()
                    if key in Decision.model_fields})
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = status in {408, 429, 500, 502, 503, 504} or type(exc).__name__ in {
                    "APIConnectionError", "APITimeoutError"}
                malformed = isinstance(exc, (ValidationError, json.JSONDecodeError))
                if not transient and not malformed:
                    raise
                history.append({"error": type(exc).__name__, "message": str(exc)[:1000]})
                recovery.recover(2 ** min(counters["retries"], 3))
                continue
            # 记录 llm 决定执行的命令以及每个阶段的记录
            write_json(directory / "progress.json", {"message": decision.summary, **counters})
            with (directory / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(decision.model_dump_json() + "\n")
            if decision.action == "blocked":
                result.execution_status = "blocked"
                result.summary = decision.summary
                break
            try:
                # 结束时对结果进行校验
                if decision.action == "finish":
                    validate_results(task, decision, evidence)
                    result.execution_status = "completed"
                    result.criteria_results = decision.criteria_results
                    result.verdict = aggregate_verdict(decision.criteria_results)
                    result.summary = decision.summary
                    break
                arguments = json.loads(decision.arguments_json)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments_json must contain an object")
            except (ValueError, ValidationError) as exc:
                history.append({"validation_error": str(exc)[:1000]})
                recovery.recover()
                continue
            if counters["tool_calls"] >= limits.tool_calls:
                raise BudgetExceeded("Tool call budget exhausted")
            counters["tool_calls"] += 1
            observation = tools.dispatch(decision.action, arguments)
            history.append({"action": decision.action, "arguments": arguments, "result": observation})
            # Ignore generated evidence IDs when detecting repeated observations.
            stable = {key: value for key, value in observation.items() if key != "evidence_ids"}
            signature = json.dumps([decision.action, arguments, stable], sort_keys=True)
            repeat_count = repeat_count + 1 if signature == last_signature else 1
            last_signature = signature
            waiting = observation.get("ok") and observation.get("running")
            if repeat_count > limits.repeated_errors and not waiting:
                raise BudgetExceeded("No progress: repeated identical operation and observation")
            if waiting:
                # Quiet long-running commands are not failures. Polling still consumes call/time budgets.
                time.sleep(min(2 ** min(repeat_count - 1, 3), backend.remaining()))
            elif not observation.get("ok") or repeat_count > 1:
                error = observation.get("error", {})
                if error.get("category") in {"permission", "budget"} or (
                        error.get("category") == "runtime" and not error.get("retryable")
                        and observation.get("effects", "none") != "unknown"):
                    result.execution_status = "blocked" if error.get("category") == "permission" else "failed"
                    result.errors.append(ExecutionError.model_validate(error))
                    result.summary = error["message"]
                    break
                recovery.recover(2 ** min(repeat_count - 1, 3))
        else:
            raise BudgetExceeded("Model call budget exhausted")
    except TimeoutError as exc:
        result.execution_status = "timed_out"
        result.summary = str(exc)
    except BudgetExceeded as exc:
        result.summary = str(exc)
        result.errors.append(ExecutionError(code="BUDGET_EXHAUSTED", category="budget", message=str(exc)))
    except Exception as exc:
        result.summary = f"{type(exc).__name__}: {exc}"[:4000]
        result.errors.append(ExecutionError(code="EXECUTION_FAILED", category="runtime", message=result.summary))
    finally:
        result.usage = counters
        # The supervisor owns the complete task scope, including crash cleanup.
        write_json(directory / "result.json", result.model_dump())
    return result


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
