"""Conversation session wrapper for the debug agent."""

from __future__ import annotations

from typing import Any

from loguru import logger

from agent_runtime.executor import DebugAgent
from agent_runtime.memory.session import SessionMemoryService
from model.agent.graph import AgentRunResult, AgentState
from model.session import ChatResponse
from utils import  _clean_string_list


class AgentSession:
    """
        Stateful chat facade over DebugAgent.run/resume.
        一个对话
    """

    def __init__(
        self,
        agent: DebugAgent,
        session_memory: SessionMemoryService | None = None,
    ) -> None:
        self.agent = agent
        self.session_memory = session_memory or SessionMemoryService.from_config(agent.config)
        self.session_id = self.session_memory.open_session(agent.config.repo_path)
        self.state: AgentState | None = None
        self.last_trace_path = ""
        self._turn_message = ""

    def send(self, message: str) -> ChatResponse:
        text = str(message or "").strip()
        if not text:
            return ChatResponse(type="empty", message="Empty message ignored.")

        if self.state and self.state.get("status") == "awaiting_user_input":
            self._turn_message = _append_follow_up(self._turn_message, text)
            result = self.agent.resume(self.state, user_answer=text)
        else:
            self._turn_message = text
            session_context = self.session_memory.prepare_turn(self.session_id, text)
            result = self.agent.run(
                title=text,
                description="",
                session_id=self.session_id,
                session_memory=session_context,
            )
        return self._record_result(result)

    def load_state(self, state: AgentState, trace_path: str = "") -> ChatResponse:
        self.state = state
        self.session_id = str(state.get("session_id") or self.session_id)
        self._turn_message = str(state.get("title") or "")
        self.last_trace_path = trace_path
        return self._to_response(state, trace_path)

    def reset(self) -> None:
        self.state = None
        self.last_trace_path = ""
        self._turn_message = ""
        self.session_id = self.session_memory.new_session(self.agent.config.repo_path)

    def _record_result(self, result: AgentRunResult) -> ChatResponse:
        self.state = result.state
        self.last_trace_path = result.trace_path
        if result.state.get("status") in {"finished", "failed"}:
            try:
                self.session_memory.commit_turn(
                    self.session_id,
                    self._turn_message or str(result.state.get("title") or ""),
                    result.state,
                )
            except Exception:
                # Memory persistence must not turn a completed agent task into a failure.
                logger.bind(
                    session_id=self.session_id,
                    task_id=result.state.get("task_id"),
                ).exception("failed to commit session memory")
            self._turn_message = ""
        return self._to_response(result.state, result.trace_path)

    def _to_response(self, state: AgentState, trace_path: str) -> ChatResponse:
        status = state.get("status")
        if status == "awaiting_user_input":
            questions = _clean_string_list(state.get("pending_user_questions"), limit=3, max_chars=300)
            reason = str(state.get("needs_user_input_reason") or "").strip()
            user_updates = _consume_user_updates(state)
            return ChatResponse(
                type="needs_user_input",
                message="Agent needs more information before continuing.",
                questions=questions,
                reason=reason,
                trace_path=trace_path,
                state=state,
                user_updates=user_updates,
                llm_token_usage=_clean_token_usage(state.get("llm_token_usage")),
                llm_errors=_clean_llm_errors(state.get("llm_errors")),
            )

        final_report = state.get("final_report") or {}
        summary = ""
        if isinstance(final_report, dict):
            summary = str(final_report.get("summary") or "").strip()
        if not summary:
            summary = f"Run finished with status={status}."
        return ChatResponse(
            type="final" if status == "finished" else "failed" if status == "failed" else "status",
            message=summary,
            trace_path=trace_path,
            state=state,
            final_report=final_report if isinstance(final_report, dict) else {},
            user_updates=_consume_user_updates(state),
            edited_files=_clean_string_list(state.get("edited_files"), limit=20, max_chars=300),
            candidate_files=_clean_string_list(state.get("candidate_files"), limit=20, max_chars=300),
            test_results=[
                item for item in state.get("test_results", []) if isinstance(item, dict)
            ],
            patch_summary=str(state.get("patch_summary") or ""),
            change_summaries=[
                item for item in state.get("change_summaries", []) if isinstance(item, dict)
            ],
            change_events=[
                item for item in state.get("change_events", []) if isinstance(item, dict)
            ],
            llm_token_usage=_clean_token_usage(state.get("llm_token_usage")),
            llm_errors=_clean_llm_errors(state.get("llm_errors")),
        )


def _consume_user_updates(state: AgentState) -> list[dict[str, Any]]:
    """
        从 user_updates 获取还没有向用户展示的提问返回
        并标记已访问
    """
    updates = state.get("user_updates")
    if not isinstance(updates, list):
        state["user_updates"] = []
        state["last_user_update"] = None
        return []

    pending: list[dict[str, Any]] = []
    marked: list[dict[str, Any]] = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or "").strip()
        if not message:
            continue
        clean_item = {
            "source": str(item.get("source") or "").strip(),
            "message": message,
            "level": str(item.get("level") or "info").strip() or "info",
            "created_at": str(item.get("created_at") or "").strip(),
            "shown": bool(item.get("shown")),
        }
        if not clean_item["shown"]:
            pending.append({**clean_item, "shown": False})
            clean_item["shown"] = True
        marked.append(clean_item)

    state["user_updates"] = marked
    state["last_user_update"] = marked[-1] if marked else None
    return pending


def _clean_token_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned = {
        "prompt_tokens": _safe_int(value.get("prompt_tokens")),
        "completion_tokens": _safe_int(value.get("completion_tokens")),
        "total_tokens": _safe_int(value.get("total_tokens")),
        "request_count": _safe_int(value.get("request_count")),
    }
    by_node = value.get("by_node")
    if isinstance(by_node, dict):
        cleaned["by_node"] = by_node
    return cleaned


def _clean_llm_errors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value[-5:] if isinstance(item, dict)]


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _append_follow_up(message: str, answer: str) -> str:
    if not message:
        return answer
    return f"{message}\nUser follow-up: {answer}"
