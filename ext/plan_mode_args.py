"""Builders for EnterPlanMode/ExitPlanMode fallback arguments."""

from __future__ import annotations

from typing import Any, Callable

from model.agent.graph import AgentState


def build_enter_plan_mode_args(
    state: AgentState,
    *,
    query: str,
    focus_files: list[str],
    read_files: list[str],
    default_verification_command: str,
) -> dict[str, Any]:
    technical_plan = str(state.get("technical_plan") or "").strip()
    if not technical_plan:
        technical_plan = _synthesized_technical_plan(state, query=query, focus_files=focus_files)
    return {
        "technical_plan": technical_plan,
        "risks": _plan_risks(state, read_files=read_files, has_code_context=bool(state.get("selected_code_context") or state.get("code_context")), has_candidate_files=bool(state.get("candidate_files"))),
        "verification_commands": _plan_verification_commands(
            state,
            default_verification_command=default_verification_command,
        ),
        "assumptions": _plan_assumptions(
            state,
            focus_files=focus_files,
            read_files=read_files,
        ),
    }


def build_exit_plan_mode_args(
    state: AgentState,
    *,
    focus_files: list[str],
    read_files: list[str],
    default_verification_command: str,
) -> dict[str, Any]:
    uncertainties = _plan_uncertainties(state, focus_files=focus_files, read_files=read_files)
    approved = bool(state.get("technical_plan")) and not uncertainties
    next_step = (
        "Apply the planned code change and then run the selected verification command."
        if approved
        else "Resolve the listed uncertainties before changing code."
    )
    return {
        "evaluation": _plan_evaluation(
            state,
            approved=approved,
            uncertainties=uncertainties,
            focus_files=focus_files,
            default_verification_command=default_verification_command,
        ),
        "approved": approved,
        "remaining_uncertainties": uncertainties,
        "next_step": next_step,
    }


def collect_plan_focus_files(state: AgentState) -> list[str]:
    files: list[str] = []
    selected = state.get("selected_code_context")
    if isinstance(selected, dict):
        for key, field in (
            ("files", "path"),
            ("functions", "file_path"),
            ("symbols", "file_path"),
            ("api_routes", "file_path"),
            ("db_models", "file_path"),
        ):
            for item in selected.get(key, []) or []:
                if isinstance(item, dict):
                    path = str(item.get(field) or "").strip()
                    if path and path not in files:
                        files.append(path)
    for path in state.get("candidate_files", []) or []:
        value = str(path or "").strip()
        if value and value not in files:
            files.append(value)
    return files[:8]


def _synthesized_technical_plan(
    state: AgentState,
    *,
    query: str,
    focus_files: list[str],
) -> str:
    file_line = ", ".join(focus_files[:5]) if focus_files else "No target files identified yet"
    steps = [
        f"Goal: {str(state.get('title') or state.get('description') or query).strip()}",
        f"Focus files: {file_line}.",
        "Plan steps:",
        "1. Confirm the relevant control flow and affected files from structured code context and read excerpts.",
        "2. Identify the smallest code change that fixes the bug or implements the requested behavior.",
        "3. Preserve existing behavior outside the target flow and avoid unrelated refactors.",
        "4. Apply a guarded patch only after the affected files and expected verification command are clear.",
        "5. Run the most relevant low-cost verification command before finishing.",
    ]
    return "\n".join(item for item in steps if item.strip())[:12000]


def _plan_verification_commands(
    state: AgentState,
    *,
    default_verification_command: str,
) -> list[str]:
    commands: list[str] = []
    for item in state.get("verification_commands", []) or []:
        if isinstance(item, dict):
            command = str(item.get("command") or "").strip()
        else:
            command = str(item or "").strip()
        if command and command not in commands:
            commands.append(command)
    if default_verification_command and default_verification_command not in commands:
        commands.append(default_verification_command)
    return commands[:5]


def _plan_assumptions(
    state: AgentState,
    *,
    focus_files: list[str],
    read_files: list[str],
) -> list[str]:
    assumptions: list[str] = []
    if focus_files:
        assumptions.append("Structured code context has already narrowed the affected files.")
    if read_files:
        assumptions.append("Previously read file excerpts are sufficient unless exact replacement text is still missing.")
    if state.get("verification_required", True):
        assumptions.append("A lightweight verification command should run after the patch.")
    return assumptions[:5]


def _plan_risks(
    state: AgentState,
    *,
    read_files: list[str],
    has_code_context: bool,
    has_candidate_files: bool,
) -> list[str]:
    risks: list[str] = []
    if not read_files:
        risks.append("The exact source text for the intended patch may still be missing.")
    if not has_code_context:
        risks.append("Structured code context has not yet isolated the affected symbols or files.")
    if not has_candidate_files:
        risks.append("Candidate files are not yet stable, so the plan may still shift.")
    return risks[:5]


def _plan_uncertainties(
    state: AgentState,
    *,
    focus_files: list[str],
    read_files: list[str],
) -> list[str]:
    questions = state.get("pending_user_questions") or []
    normalized = [str(item).strip() for item in questions if str(item).strip()]
    if normalized:
        return normalized[:3]
    if not focus_files:
        return ["The affected file or symbol has not been isolated yet."]
    if not read_files:
        return ["The exact source text for the target file has not been read yet."]
    return []


def _plan_evaluation(
    state: AgentState,
    *,
    approved: bool,
    uncertainties: list[str],
    focus_files: list[str],
    default_verification_command: str,
) -> str:
    status = "feasible" if approved else "not yet ready"
    focus_line = ", ".join(focus_files[:4]) or "unknown files"
    verification = ", ".join(
        _plan_verification_commands(state, default_verification_command=default_verification_command)[:2]
    ) or "no verification command yet"
    detail = f"Plan is {status}. Focus files: {focus_line}. Verification: {verification}."
    if uncertainties:
        detail += f" Remaining uncertainties: {'; '.join(uncertainties[:3])}."
    return detail[:8000]
