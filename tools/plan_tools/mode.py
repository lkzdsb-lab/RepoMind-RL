"""Plan-mode primitives used to gate code-changing actions."""

from __future__ import annotations

from typing import Any, Dict
from utils import  _clean_string_list, _as_bool


def enter_plan_mode(repo_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Record a technical plan and enter the non-mutating planning gate."""
    del repo_path
    technical_plan = _clean_text(args.get("technical_plan"), limit=12000)
    if not technical_plan:
        return {
            "entered": False,
            "error": "EnterPlanMode requires technical_plan.",
            "needs_more_context": True,
        }
    return {
        "entered": True,
        "technical_plan": technical_plan,
        "risks": _clean_string_list(args.get("risks"), limit=12, max_chars=500),
        "verification_commands": _clean_string_list(
            args.get("verification_commands"),
            limit=10,
            max_chars=300,
        ),
        "assumptions": _clean_string_list(args.get("assumptions"), limit=12, max_chars=500),
    }


def exit_plan_mode(repo_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Exit planning only when the plan has been evaluated as feasible."""
    del repo_path
    evaluation = _clean_text(args.get("evaluation"), limit=8000)
    approved = _as_bool(args.get("approved"))
    remaining_uncertainties = _clean_string_list(
        args.get("remaining_uncertainties"),
        limit=8,
        max_chars=500,
    )
    if not evaluation:
        return {
            "exited": False,
            "approved": False,
            "error": "ExitPlanMode requires evaluation.",
            "needs_more_context": True,
        }
    if remaining_uncertainties:
        return {
            "exited": False,
            "approved": False,
            "evaluation": evaluation,
            "remaining_uncertainties": remaining_uncertainties,
            "needs_user_input": True,
            "reason": "Plan still has unresolved uncertainties.",
            "questions": remaining_uncertainties[:3],
        }
    if not approved:
        return {
            "exited": False,
            "approved": False,
            "evaluation": evaluation,
            "needs_more_context": True,
            "error": "Plan evaluation did not approve implementation.",
        }
    return {
        "exited": True,
        "approved": True,
        "evaluation": evaluation,
        "remaining_uncertainties": [],
        "next_step": _clean_text(args.get("next_step"), limit=1000),
    }


def _clean_text(value: Any, *, limit: int) -> str:
    return str(value or "").strip()[:limit]

