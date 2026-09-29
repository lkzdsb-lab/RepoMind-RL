"""Render the complete saved report without another model call."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.text import Text

from model.session import ChatResponse


def render_final_report(response: ChatResponse) -> Group:
    report = response.final_report or {}
    state = response.state or {}
    parts: list[Text] = [Text(str(report.get("summary") or response.message or response.type))]
    drafts = {item.get("candidate_id"): item for item in state.get("draft_findings", [])}
    reviews = {
        str(item.get("claim") or "").strip()[:600]: item
        for item in (state.get("completion_judgement") or {}).get("reviewed_findings", [])
        if item.get("verdict") == "confirmed"
    }
    findings = []
    for claim in report.get("findings", []):
        review = reviews.get(str(claim), {})
        draft = drafts.get(review.get("candidate_id"), {})
        detail = _finding_text({**draft, "claim": claim})
        if review.get("reason"):
            detail += "\n判断依据：" + str(review["reason"])
        findings.append(detail)
    _section(parts, "确认的发现", findings)
    _section(parts, "待审查的候选（尚未确认）",
             [_finding_text(item) for item in report.get("pending_findings", [])])
    for key, title in (
        ("work_done", "已完成的工作"),
        ("test_results", "测试与验证结果"),
        ("verification_commands", "验证命令"),
        ("command_results", "命令结果"),
    ):
        _section(parts, title, report.get(key, []))
    patch = report.get("patch_status") or response.patch_summary
    _section(parts, "代码修改", [patch] if patch else [])
    _section(parts, "后续建议与未完成事项", report.get("next_steps", []))
    return Group(*parts)


def _section(parts: list[Text], title: str, items: list[Any]) -> None:
    lines = [str(item).strip() for item in items if str(item).strip()]
    if not lines:
        return
    parts.append(Text("\n" + title, style="bold"))
    for index, line in enumerate(lines, start=1):
        parts.append(Text(f"{index}. {line}"))


def _finding_text(item: dict[str, Any]) -> str:
    detail = str(item.get("claim") or "")
    locations = []
    for location in item.get("locations", []):
        path = str(location.get("file_path") or "")
        if location.get("start_line"):
            path += f":{location['start_line']}"
            if location.get("end_line") != location.get("start_line") and location.get("end_line"):
                path += f"-{location['end_line']}"
        if path:
            locations.append(path)
    if locations:
        detail += "\n位置：" + ", ".join(locations)
    if item.get("severity"):
        detail += "\n严重程度：" + str(item["severity"])
    if item.get("related_tests"):
        detail += "\n相关测试：" + ", ".join(item["related_tests"]) + "（是否运行以验证结果为准）"
    return detail
