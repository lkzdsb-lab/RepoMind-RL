"""Prompt context assembly from distilled runtime events."""

from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.context.distiller import DistilledEvent
from model.agent.graph import AgentState
from utils import _clean_string_list

@dataclass
class AssembledContext:
    working_context: str
    archive_context: str
    context_sections: dict[str, list[str]]


class ContextAssembler:
    def assemble(self, events: list[DistilledEvent], state: AgentState) -> AssembledContext:
        """对蒸馏过后的数据进行 聚合"""
        # Assembly preserves pending facts; the final rendered prompt has one budget.
        rendered_sections = _section_events(events, state)

        archive_lines = _archive_lines(events)
        archive_context = _render("Archived Summary", archive_lines)
        return AssembledContext(
            working_context=_render_sections(rendered_sections),
            archive_context=archive_context,
            context_sections=rendered_sections,
        )


def _section_events(events: list[DistilledEvent], state: AgentState) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "Current Goal": [],
        "User Constraints": [],
        "Active Plan": [],
        "Working Facts": [],
        "Recent Critical Events": [],
        "Verification State": [],
        "Open Questions": [],
    }
    goal = " ".join(
        str(part).strip()
        for part in (state.get("title", ""), state.get("description", ""))
        if str(part).strip()
    )
    if goal:
        sections["Current Goal"].append(goal[:1000])
    if state.get("verification_reason"):
        sections["Current Goal"].append(
            f"verification_required={bool(state.get('verification_required', True))}; "
            f"reason={str(state.get('verification_reason'))[:500]}"
        )

    for event in reversed(events):
        lines = _event_lines(event)
        if event.event_type == "user_event":
            sections["User Constraints"].extend(lines)
        elif event.event_type == "plan_event":
            sections["Active Plan"].extend(lines)
        elif event.event_type == "verification_event":
            sections["Verification State"].extend(lines)
        elif event.importance in {"high", "critical"}:
            sections["Recent Critical Events"].extend(lines)
        elif event.level == "archive":
            continue
        else:
            sections["Working Facts"].extend(lines)

        for question in event.open_questions:
            sections["Open Questions"].append(question)

    if state.get("verification_stale"):
        sections["Verification State"].append("Latest code edits are stale and require verification.")
    if state.get("edited_files"):
        sections["Working Facts"].append(f"edited_files={state.get('edited_files')}")
    return {name: _clean_string_list(lines, len(lines), None) for name, lines in sections.items()}


def _event_lines(event: DistilledEvent) -> list[str]:
    lines = []
    for fact in event.facts:
        lines.append(f"{event.event_type}/{event.source}: {fact}")
    for risk in event.risks:
        lines.append(f"risk: {risk}")
    for action in event.next_actions:
        lines.append(f"next: {action}")
    if not lines and event.summary:
        lines.append(f"{event.event_type}/{event.source}: {event.summary}")
    return lines


def _archive_lines(events: list[DistilledEvent]) -> list[str]:
    lines: list[str] = []
    for event in events:
        if event.level != "archive":
            continue
        line = f"{event.event_type}/{event.source}: {event.summary}"
        if line not in lines:
            lines.append(line[:600])
    return lines


def _render_sections(sections: dict[str, list[str]]) -> str:
    chunks: list[str] = ["# Working Context"]
    for title, lines in sections.items():
        if not lines:
            continue
        chunks.append(f"\n## {title}")
        for line in lines:
            chunks.append(f"- {line}")
    return "\n".join(chunks)


def _render(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    chunks = [f"# {title}"]
    for line in lines:
        chunks.append(f"- {line}")
    return "\n".join(chunks)


