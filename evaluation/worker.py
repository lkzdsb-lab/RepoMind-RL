from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from agent_runtime.executor import DebugAgent
from agent_runtime.session import AgentSession
from config import (
    debug_agent_config_from_dict,
    load_config_payload,
    load_env_file,
    normalize_project_runtime_paths,
    validate_debug_agent_config,
)
from evaluation.models import EvaluationCase


def run_worker(request_path: Path) -> int:
    request = _read_json(request_path)
    case = EvaluationCase.model_validate(request["case"])
    workspace = Path(request["workspace"]).resolve()
    output_path = Path(request["raw_result_path"]).resolve()
    traces_dir = Path(request["traces_dir"]).resolve()
    config_path = Path(request["config_path"]).resolve()

    turns: list[dict[str, Any]] = []
    traces_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = load_config_payload(config_path, require_exists=True)
        config = debug_agent_config_from_dict(payload)
        config.repo_path = workspace.as_posix()
        config.max_loops = case.limits.max_loops
        config.editing_enabled = case.intent == "implement"
        config.require_step_approval = False
        config.log_to_console = False
        config.env_file = None
        load_env_file(_env_path(config_path, payload), override=config.env_override)
        normalize_project_runtime_paths(config)
        validate_debug_agent_config(config)
        session = AgentSession(DebugAgent(config))
        for index, turn in enumerate(case.turns, start=1):
            response = session.send(turn.message)
            state = response.state or {}
            copied_trace = _copy_trace(response.trace_path, traces_dir, index)
            turns.append(_turn_result(index, turn.message, response, state, copied_trace))
        raw_result = {"worker_status": "completed", "turns": turns}
        exit_code = 0
    except BaseException as exc:
        raw_result = {
            "worker_status": "failed",
            "worker_error": f"{exc.__class__.__name__}: {exc}",
            "turns": turns,
        }
        exit_code = 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(raw_result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return exit_code


def _turn_result(
    index: int,
    message: str,
    response: Any,
    state: dict[str, Any],
    trace_path: str,
) -> dict[str, Any]:
    return {
        "turn_index": index,
        "message": message,
        "task_id": str(state.get("task_id") or ""),
        "status": str(state.get("status") or response.type or ""),
        "response_type": str(response.type or ""),
        "summary": str(response.message or ""),
        "trace_path": trace_path,
        "loop_count": int(state.get("loop_count", 0) or 0),
        "tool_call_count": len(state.get("tool_calls", []) or []),
        "llm_call_count": len(state.get("llm_calls", []) or []),
        "llm_token_usage": state.get("llm_token_usage", {}),
        "llm_errors": state.get("llm_errors", []),
        "test_results": state.get("test_results", []),
        "edited_files": state.get("edited_files", []),
        "draft_findings": state.get("draft_findings", []),
        "final_report": state.get("final_report", {}),
        "error": str(state.get("error") or ""),
    }


def _copy_trace(value: str, target_dir: Path, turn_index: int) -> str:
    if not value:
        return ""
    source = Path(value)
    if not source.is_file():
        return ""
    target = target_dir / f"turn-{turn_index}-{source.name}"
    shutil.copy2(source, target)
    return target.as_posix()


def _env_path(config_path: Path, payload: dict[str, Any]) -> Path | None:
    value = payload.get("env_file", ".env")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else config_path.parent / path


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m evaluation.worker <request.json>")
    raise SystemExit(run_worker(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
