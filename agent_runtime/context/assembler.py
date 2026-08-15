"""Prompt context assembly from distilled runtime events."""

from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.context.distiller import DistilledEvent
from agent_runtime.context.token_counter import estimate_tokens
from model.agent.graph import AgentState
from utils import _clean_string_list

# todo 后续考虑用配置
# 每个板块的最大 token 数
weights = {
        "Current Goal": 0.12,
        "User Constraints": 0.16,
        "Active Plan": 0.18,
        "Working Facts": 0.24,
        "Recent Critical Events": 0.18,
        "Verification State": 0.14,
        "Open Questions": 0.10,
        "Recent Tool Evidence": 0.18,
    }

@dataclass
class AssembledContext:
    working_context: str
    archive_context: str
    context_sections: dict[str, list[str]]


@dataclass
class ContextAssembler:
    max_tokens: int = 32000
    # 预留 llm 回复的兜底 token
    reserved_tokens: int = 8000

    def assemble(self, events: list[DistilledEvent], state: AgentState) -> AssembledContext:
        """对蒸馏过后的数据进行 聚合"""
        budget = max(1200, self.max_tokens - self.reserved_tokens)
        sections = _section_events(events, state)
        rendered_sections: dict[str, list[str]] = {}
        used = 0

        for name, lines in sections.items():
            if not lines:
                continue
            kept: list[str] = []
            section_budget = _section_budget(name, budget)
            section_used = 0
            for line in lines:
                cost = estimate_tokens(line)
                # 对每个板块进行 token 限制
                if kept and section_used + cost > section_budget:
                    continue
                # 对整个 token 进行限制
                if used + cost > budget and name not in {"Current Goal", "User Constraints"}:
                    continue
                kept.append(line)
                section_used += cost
                used += cost
            if kept:
                rendered_sections[name] = kept

        archive_lines = _archive_lines(events)
        archive_context = _render("Archived Summary", archive_lines[:12])
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
        "Recent Tool Evidence": [],
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

    for event in events:
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
        if event.source in {"read_file", "search_text", "search_code_context", "run_shell_command", "run_tests"}:
            sections["Recent Tool Evidence"].extend(lines[:3])

    if state.get("verification_stale"):
        sections["Verification State"].append("Latest code edits are stale and require verification.")
    if state.get("edited_files"):
        sections["Working Facts"].append(f"edited_files={state.get('edited_files')}")
    return {name: _clean_string_list(lines, 20, None) for name, lines in sections.items()}


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


def _section_budget(name: str, total_budget: int) -> int:
    return max(300, int(total_budget * weights.get(name, 0.1)))
