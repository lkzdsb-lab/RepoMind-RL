from __future__ import annotations

from evaluation.models import EvaluationResult


def markdown_report(result: EvaluationResult) -> str:
    verification = result.verification
    finding_lines = [
        f"- [{'x' if item.matched else ' '}] `{item.bug_id}`"
        + (f" ({item.source})" if item.source else "")
        for item in result.finding_matches
    ]
    failure_lines = [f"- {item}" for item in result.failures] or ["- None"]
    changed_lines = [f"- `{item}`" for item in result.changed_files] or ["- None"]
    return "\n".join(
        [
            f"# Evaluation: {result.case_id}",
            "",
            f"- Passed: **{result.passed}**",
            f"- Score: **{result.score}/100**",
            f"- Agent status: `{result.agent_status}`",
            f"- Verification exit code: `{verification.exit_code}`",
            f"- Findings: `{result.findings_found}/{result.findings_total}`",
            f"- Loops / tools / LLM calls: `{result.loops} / {result.tool_calls} / {result.llm_calls}`",
            f"- Total tokens: `{result.total_tokens}`",
            "",
            "## Findings",
            *(finding_lines or ["- No expected findings configured."]),
            "",
            "## Changed Files",
            *changed_lines,
            "",
            "## Failures",
            *failure_lines,
            "",
            "## Verification Output",
            "```text",
            (verification.stdout + "\n" + verification.stderr).strip()[-8000:],
            "```",
            "",
        ]
    )
