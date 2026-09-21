"""Parent-side task tool: trusted configuration, supervision and result handoff."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import threading
import time
from uuid import uuid4

from agent_runtime.user_updates import emit_user_update
from config import resolve_llm_config

from .artifacts import read_json, write_json
from .backend import BackendUnavailable, command_environment
from .contracts import CriterionResult, ExecutionError, ExecutionResult, ExecutionTaskInput, Limits, ResourceLimits
from .resources import ResourceExceeded
from .supervisor import ExecutionSupervisor


_ACTIVE = threading.Lock()


def run_execution_task(repo_path: str, args: dict) -> dict:
    context = args.get("_runtime_context") or {}
    config = context.get("config")
    task_id = str(context.get("task_id") or "")
    attempt_id = uuid4().hex
    result = ExecutionResult(task_id=task_id, attempt_id=attempt_id, execution_status="blocked")
    if config is None or not config.execution_enabled:
        result.summary = "Execution subagent is disabled or trusted runtime context is missing"
        result.errors.append(ExecutionError(code="EXECUTION_DISABLED", category="permission", message=result.summary))
        result.cleanup_status = "completed"
        return tool_result(result)
    # 检查是否已经有 sub 在运行
    if not _ACTIVE.acquire(blocking=False):
        result.summary = "Another execution subagent is active"
        result.errors.append(ExecutionError(code="EXECUTION_BUSY", category="runtime", message=result.summary))
        result.cleanup_status = "completed"
        return tool_result(result)
    directory = None
    supervisor = None
    task = None
    try:
        task = ExecutionTaskInput.model_validate({key: value for key, value in args.items() if key != "_runtime_context"})
        limits = Limits(timeout=config.execution_timeout, model_calls=config.execution_model_calls,
                        tool_calls=config.execution_tool_calls, retries=config.execution_retries,
                        startup_timeout=config.execution_startup_timeout,
                        operation_timeout=config.execution_operation_timeout,
                        cleanup_timeout=config.execution_cleanup_timeout)
        resources = ResourceLimits.model_validate(config.execution_resources)
        deadline = time.monotonic() + limits.timeout
        # 限制访问的目录
        root = Path(repo_path).resolve()
        if not root.is_dir():
            raise ValueError("Execution workspace must be an existing directory")
        artifact_root = root / ".repomind" / "execution"
        if not artifact_root.resolve().is_relative_to(root):
            raise ValueError("Execution artifacts must stay inside the target workspace")
        directory = artifact_root / attempt_id
        directory.mkdir(parents=True, exist_ok=False)
        work = directory / "work"
        work.mkdir()
        llm = resolve_llm_config(config.llm_config, config.action_llm_config)
        if not llm.model or llm.provider in {"", "disabled", "none"}:
            raise BackendUnavailable("Execution requires an enabled LLM configuration")
        if not os.environ.get(llm.api_key_env):
            raise BackendUnavailable(f"Missing model credential environment variable: {llm.api_key_env}")
        llm_values = asdict(llm)
        llm_values["structured_fallback"] = False
        llm_values["max_retries"] = 0
        llm_values["timeout"] = min(llm.timeout, limits.timeout)
        secret_names = {llm.api_key_env, config.llm_config.api_key_env}
        for value in vars(config).values():
            if getattr(value, "api_key_env", None):
                secret_names.add(value.api_key_env)
        environment = command_environment(work, config.execution_env, secret_names)
        request = {"schema_version": 1, "task_id": task_id, "attempt_id": attempt_id,
                   "task": task.model_dump(), "limits": limits.model_dump(),
                   "resource_limits": resources.model_dump(),
                   "source_fingerprint": "",
                   "workspace": str(root), "work_directory": str(work),
                   "llm": llm_values, "deadline": deadline}
        write_json(directory / "request.json", request)
        # 注册监督者，加载全局资源限制
        supervisor = ExecutionSupervisor(directory, limits, resources, deadline, context.get("cancel_event"))
        # 启动 host 文件初始化运行环境，比如工具注册
        supervisor.launch("host", environment)
        supervisor.wait_ready()
        result.source_fingerprint = read_json(directory / "request.json")["source_fingerprint"]
        # Separate environment copies: project commands never inherit the worker's API key.
        environment = command_environment(work, {}, secret_names)
        environment[llm.api_key_env] = os.environ[llm.api_key_env]
        environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                            "TEMP": str(directory), "TMP": str(directory), "TMPDIR": str(directory)})
        supervisor.launch("worker", environment)
        supervisor.wait_worker(lambda progress: emit_user_update(
            {"source": "execution_agent", "message": progress.get("message", "")}))
        # 写文件通知 host 已经结束
        write_json(directory / "finalize.json", {})
        completed = ExecutionResult.model_validate(supervisor.wait_result())
        if (completed.task_id, completed.attempt_id, completed.source_fingerprint) != (
                task_id, attempt_id, result.source_fingerprint):
            raise ValueError("Worker result identity mismatch")
        if completed.execution_status != "completed" and completed.verdict != "inconclusive":
            raise ValueError("Incomplete execution cannot claim a verification verdict")
        result = completed
    except KeyboardInterrupt:
        result.execution_status = "cancelled"
        result.summary = "Execution cancelled"
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        result.execution_status = "timed_out"
        result.summary = str(exc) or "Execution deadline exceeded"
        result.errors.append(ExecutionError(code=getattr(exc, "code", "TASK_DEADLINE_EXCEEDED"),
                                            category="budget", message=result.summary))
    except (BackendUnavailable, FileNotFoundError) as exc:
        result.execution_status = "blocked"
        result.summary = str(exc)[:4000]
        result.errors.append(ExecutionError(code="BACKEND_UNAVAILABLE", category="permission", message=result.summary))
    except Exception as exc:
        # Assignment/creation can fail before the first supervision tick. Only a
        # confirmed OS event, not an arbitrary crash, upgrades it to a limit error.
        if supervisor is not None and not isinstance(exc, ResourceExceeded):
            try:
                supervisor.scope.sample()
            except ResourceExceeded as resource_error:
                exc = resource_error
            except Exception:
                pass
        resource_failure = isinstance(exc, ResourceExceeded)
        result.execution_status = "failed"
        result.summary = f"{type(exc).__name__}: {exc}"[:4000]
        result.errors.append(ExecutionError(
            code="RESOURCE_LIMIT_EXCEEDED" if resource_failure else "EXECUTION_FAILED",
            category="budget" if resource_failure else "runtime", message=result.summary))
    finally:
        cleanup_errors = []
        if supervisor is not None:
            try:
                supervisor.close()
            except Exception as exc:
                cleanup_errors.append(str(exc))
            result.resources = supervisor.scope.usage
        result.cleanup_status = "failed" if cleanup_errors else "completed"
        if cleanup_errors:
            result.errors.append(ExecutionError(code="CLEANUP_FAILED", category="runtime",
                                                 message="; ".join(cleanup_errors)[:2000]))
            result.summary = (result.summary + "; cleanup failed: " + "; ".join(cleanup_errors))[:4000]
        if result.execution_status != "completed" or cleanup_errors:
            result.verdict = "inconclusive"
            for item in result.criteria_results:
                item.verdict = "inconclusive"
        if task is not None:
            represented = {item.criterion_id for item in result.criteria_results}
            result.criteria_results.extend(
                CriterionResult(criterion_id=item.criterion_id, verdict="inconclusive",
                                explanation=result.summary[:2000])
                for item in task.acceptance_criteria if item.criterion_id not in represented
            )
        if not result.usage and directory is not None and (directory / "progress.json").is_file():
            try:
                progress = read_json(directory / "progress.json", max_bytes=16384)
                result.usage = {key: int(progress.get(key, 0)) for key in
                                ("model_calls", "tool_calls", "retries", "total_tokens")}
            except (OSError, ValueError, TypeError):
                pass
        try:
            if directory is not None and directory.is_dir():
                write_json(directory / "result.json", result.model_dump())
        except OSError as exc:
            result.execution_status = "failed"
            result.verdict = "inconclusive"
            result.summary = f"Could not persist execution result: {exc}"[:4000]
            result.errors.append(ExecutionError(code="ARTIFACT_WRITE_FAILED", category="runtime", message=result.summary))
        finally:
            _ACTIVE.release()
    return tool_result(result, directory)


def tool_result(result: ExecutionResult, directory: Path | None = None) -> dict:
    ok = result.execution_status == "completed" and result.cleanup_status == "completed"
    excerpts = []
    if directory is not None and (directory / "excerpts.json").is_file():
        try:
            excerpts = read_json(directory / "excerpts.json", max_bytes=65536)["items"]
        except (OSError, ValueError, KeyError):
            excerpts = []
    return {"ok": ok, "status": "success" if ok else "failed",
            "summary": result.summary, "execution_result": result.model_dump(),
            "evidence_excerpts": excerpts,
            "error": "" if ok else result.summary,
            "artifacts": ([{"type": "execution", "path": str(directory / "result.json")}]
                          if directory is not None else [])}
