from dataclasses import dataclass, field
from model.agent.graph import AgentState
from typing import Any


@dataclass
class ChatResponse:
    type: str
    message: str = ""
    questions: list[str] = field(default_factory=list)
    reason: str = ""
    trace_path: str = ""
    state: AgentState | None = None
    final_report: dict[str, Any] = field(default_factory=dict)
    user_updates: list[dict[str, Any]] = field(default_factory=list)
    edited_files: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    patch_summary: str = ""
    change_summaries: list[dict[str, Any]] = field(default_factory=list)
    change_events: list[dict[str, Any]] = field(default_factory=list)
    llm_token_usage: dict[str, Any] = field(default_factory=dict)
    llm_errors: list[dict[str, Any]] = field(default_factory=list)
