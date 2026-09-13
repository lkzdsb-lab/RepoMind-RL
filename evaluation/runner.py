from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluation.models import EvaluationCase, EvaluationResult, VerificationResult
from evaluation.reporting import markdown_report
from evaluation.scoring import score_evaluation
from evaluation.workspace import (
    changed_files,
    copy_fixture,
    snapshot_digest,
    snapshot_workspace,
    unified_workspace_diff,
)


def run_evaluation(
    case_path: str | Path,
    *,
    config_path: str | Path = "config.json",
    keep_workspace: bool = False,
    timeout_override: int | None = None,
) -> EvaluationResult | dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    case_file = Path(case_path).resolve()
    config_file = Path(config_path).resolve()
    if json.loads(case_file.read_text(encoding="utf-8")).get("kind") == "memory_retrieval":
        from evaluation.memory_retrieval import run_memory_evaluation
        return run_memory_evaluation(
            case_file, config_path=config_file, keep_workspace=keep_workspace,
            timeout_override=timeout_override,
        )
    case = EvaluationCase.model_validate_json(case_file.read_text(encoding="utf-8"))
    fixture = Path(case.fixture)
    if not fixture.is_absolute():
        fixture = project_root / fixture

    run_id = _run_id()
    run_dir = project_root / ".repomind" / "evaluation" / case.case_id / run_id
    workspace = run_dir / "workspace"
    traces_dir = run_dir / "traces"
    raw_result_path = run_dir / "raw_result.json"
    request_path = run_dir / "request.json"
    worker_log_path = run_dir / "worker.log"
    diff_path = run_dir / "diff.patch"
    result_path = run_dir / "result.json"
    report_path = run_dir / "report.md"
    run_dir.mkdir(parents=True, exist_ok=False)
    copy_fixture(fixture.resolve(), workspace)
    before = snapshot_workspace(workspace)
    fixture_hash = snapshot_digest(before)

    request = {
        "case": case.model_dump(mode="json"),
        "workspace": workspace.as_posix(),
        "config_path": config_file.as_posix(),
        "raw_result_path": raw_result_path.as_posix(),
        "traces_dir": traces_dir.as_posix(),
    }
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    worker_timed_out = False
    worker_exit_code: int | None = None
    timeout = timeout_override or case.limits.case_timeout
    with worker_log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "evaluation.worker", request_path.as_posix()],
                cwd=project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            worker_exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            worker_timed_out = True
            log_file.write(f"\nEvaluation worker timed out after {timeout} seconds.\n")

    _copy_agent_log(workspace, run_dir)
    raw_result = _read_json_or_error(raw_result_path)
    verification = _run_verification(case, workspace)
    after = snapshot_workspace(workspace)
    changed = changed_files(before, after)
    diff_path.write_text(unified_workspace_diff(before, after), encoding="utf-8")
    artifacts = {
        "run_dir": run_dir.as_posix(),
        "raw_result": raw_result_path.as_posix(),
        "report": report_path.as_posix(),
        "diff": diff_path.as_posix(),
        "worker_log": worker_log_path.as_posix(),
        "traces": traces_dir.as_posix(),
    }
    config_fingerprint = _config_fingerprint(config_file)
    result = score_evaluation(
        run_id=run_id,
        case=case,
        raw_result=raw_result,
        changed=changed,
        verification=verification,
        fixture_hash=fixture_hash,
        config_fingerprint=config_fingerprint,
        artifacts=artifacts,
        worker_timed_out=worker_timed_out,
        worker_exit_code=worker_exit_code,
    )
    result_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown_report(result), encoding="utf-8")
    if result.passed and not keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    else:
        result.artifacts["workspace"] = workspace.as_posix()
        result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def _run_verification(case: EvaluationCase, workspace: Path) -> VerificationResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            case.verification.argv,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=case.verification.timeout,
            shell=False,
            check=False,
        )
        return VerificationResult(
            argv=case.verification.argv,
            exit_code=completed.returncode,
            stdout=completed.stdout[-12000:],
            stderr=completed.stderr[-12000:],
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except subprocess.TimeoutExpired as exc:
        return VerificationResult(
            argv=case.verification.argv,
            stdout=_process_text(exc.stdout),
            stderr=_process_text(exc.stderr),
            timed_out=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except OSError as exc:
        return VerificationResult(
            argv=case.verification.argv,
            start_error=str(exc),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def _copy_agent_log(workspace: Path, run_dir: Path) -> None:
    source = workspace / ".repomind" / "logs" / "agent.log"
    if source.is_file():
        shutil.copy2(source, run_dir / "agent.log")


def _config_fingerprint(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        marker = f"unavailable-config:{exc.__class__.__name__}"
        return hashlib.sha256(marker.encode("utf-8")).hexdigest()
    sanitized = _sanitize_secrets(payload)
    encoded = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sanitize_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if _secret_key(str(key))
            else _sanitize_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_secrets(item) for item in value]
    return value


def _secret_key(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {
        "api_key",
        "api_key_env",
        "password",
        "secret",
        "access_token",
        "refresh_token",
    }


def _read_json_or_error(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"worker_status": "failed", "worker_error": "worker produced no result", "turns": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"worker_status": "failed", "worker_error": f"invalid worker result: {exc}", "turns": []}
    return data if isinstance(data, dict) else {"worker_status": "failed", "turns": []}


def _process_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-12000:]
    return str(value or "")[-12000:]


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"
