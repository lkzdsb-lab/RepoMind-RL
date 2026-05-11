"""
Trajectory recorder.

Every agent step is kept in state and can also be persisted as JSON for replay,
debugging, and later RL transition extraction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from model.agent.graph import AgentState, TrajectoryStep


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    step_id: int
    node: str
    thought: str
    action: str | None = None
    action_input: Dict[str, Any] | None = None
    observation: Dict[str, Any] | None = None
    created_at: str = field(default_factory=utc_now)


class TrajectoryRecorder:
    def __init__(self, task_id: str | None = None) -> None:
        self.task_id = task_id or str(uuid4())

    def append(
        self,
        state: AgentState,
        node: str,
        thought: str,
        action: str | None = None,
        action_input: Dict[str, Any] | None = None,
        observation: Dict[str, Any] | None = None,
    ) -> AgentState:
        trajectory = state.get("trajectory", [])
        event = TraceEvent(
            step_id=len(trajectory) + 1,
            node=node,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        step = TrajectoryStep(**asdict(event))
        return {
            **state,
            "task_id": state.get("task_id") or self.task_id,
            "trajectory": trajectory + [step],
        }

    def save(self, state: AgentState, output_dir: str | Path = ".repomind/traces") -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        trace_path = path / f"{state.get('task_id', self.task_id)}.json"
        trace_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return trace_path
