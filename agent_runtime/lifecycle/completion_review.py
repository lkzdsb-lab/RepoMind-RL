"""Bounded completion-review recovery and reuse of unchanged evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from agent_runtime.llm.completion_judge import CompletionJudge, review_failure
from agent_runtime.llm.llm_nodes import _llm_error_payload
from agent_runtime.memory.file_cache import source_revision
from model.agent.graph import AgentState


@dataclass
class CompletionReviewController:
    judge: CompletionJudge
    max_attempts: int = 2
    max_unchanged_finishes: int = 2

    def review(self, state: AgentState) -> dict[str, Any]:
        fingerprint = evidence_fingerprint(state)
        previous = state.get("completion_review_cache") or {}
        log = logger.bind(task_id=state.get("task_id"))
        if previous.get("fingerprint") == fingerprint and previous.get("result", {}).get("decision") == "continue":
            count = int(state.get("completion_review_unchanged_count", 0)) + 1
            state["completion_review_unchanged_count"] = count
            log.warning("completion review reused fingerprint={} unchanged_finishes={}", fingerprint[:16], count)
            if count >= self.max_unchanged_finishes:
                result = review_failure("no_progress", "Completion review requested more evidence, but repeated finish requests supplied no new evidence.")
                result["missing_evidence"] = previous["result"].get("missing_evidence", [])
                result["reviewed_findings"] = previous["result"].get("reviewed_findings", [])
                return {**result, "evidence_fingerprint": fingerprint, "cached": True}
            return {**previous["result"], "cached": True}

        state["completion_review_unchanged_count"] = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.judge.judge(state)
            except Exception as exc:
                error = _llm_error_payload("completion_judge", exc)
                result = review_failure(error["category"], error["message"])
            if not isinstance(result, dict) or result.get("status") not in {"succeeded", "failed"}:
                result = review_failure("response_format", "Completion judge returned no valid review status.")
            if result.get("status") == "succeeded" and result.get("decision") not in {"complete", "continue", "needs_user_input"}:
                result = review_failure("response_format", "Completion judge returned an invalid decision.")
            result = {**result, "evidence_fingerprint": fingerprint, "attempt": attempt, "cached": False}
            if result["status"] == "succeeded":
                state["completion_review_failed"] = False
                state["completion_review_cache"] = {"fingerprint": fingerprint, "result": result}
                if result["decision"] == "continue":
                    state["completion_judge_continue_count"] = int(state.get("completion_judge_continue_count", 0)) + 1
                log.info("completion review succeeded attempt={} decision={} fingerprint={}", attempt, result["decision"], fingerprint[:16])
                return result

            state["completion_review_failure_count"] = int(state.get("completion_review_failure_count", 0)) + 1
            category = result.get("error", {}).get("category", "unknown")
            retry = category in {"timeout", "connection", "server_error"} and attempt < self.max_attempts
            state["completion_review_attempts"] = list(state.get("completion_review_attempts", [])) + [result]
            log.warning("completion review failed attempt={} category={} retry={} fingerprint={}", attempt, category, retry, fingerprint[:16])
            if not retry:
                return result
        raise RuntimeError("Completion review attempt budget must be positive")


def evidence_fingerprint(state: AgentState) -> str:
    """Ignore plans, narration, timestamps and repeated identical tool observations."""
    calls = state.get("tool_calls") or []
    paths = set(state.get("candidate_files") or [])
    findings = []
    for item in state.get("draft_findings", []):
        findings.append({key: item.get(key) for key in ("candidate_id", "claim", "locations", "related_tests")})
        paths.update(location["file_path"] for location in item.get("locations", []) if location.get("file_path"))
    observations = set()
    for call in calls:
        if call.get("name") in {"finish", "request_user_input"}:
            continue
        output = call.get("output") or {}
        if isinstance(output, dict) and output.get("file_path"):
            paths.add(output["file_path"])
        observations.add(_canonical({"name": call.get("name"), "input": call.get("input"), "output": output}))
    material = {
        "task": state.get("task_brief"), "user_inputs": state.get("user_inputs"),
        "findings": sorted(findings, key=lambda item: str(item.get("candidate_id"))),
        "observations": sorted(observations),
        "source_revisions": {path: source_revision(state, path) for path in sorted(paths)},
        "edit_revision": (state.get("runtime_facts") or {}).get("edit_revision", 0),
    }
    return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(_evidence_value(value), sort_keys=True, ensure_ascii=False, default=str)


def _evidence_value(value: Any) -> Any:
    if isinstance(value, dict):
        ignored = {"created_at", "updated_at", "timestamp", "loop_count", "elapsed_ms", "duration_ms",
                   "access_seq", "source", "session_cache_reused", "reason", "confidence", "max_chars"}
        return {key: _evidence_value(item) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [_evidence_value(item) for item in value]
    return value
