"""Final user-facing report generation for agent runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.llm.llm_nodes import LLMJsonNode
from config import LLMConfig
from model.agent.graph import AgentState
from model.llm import FinalReportResponse
from prompts.templates import load_prompt, render_prompt


class FinalReporter(Protocol):
    def report(self, state: AgentState) -> dict[str, Any]:
        ...


@dataclass
class RuleBasedFinalReporter:
    """Deterministic fallback report used when the LLM reporter is disabled."""

    def report(self, state: AgentState) -> dict[str, Any]:
        tool_names = [
            str(call.get("name"))
            for call in state.get("tool_calls", [])
            if isinstance(call, dict) and call.get("name")
        ]
        work_done = _work_done_from_tools(tool_names)
        candidate_files = _clean_list(state.get("candidate_files"), 12, 260)
        test_results = _test_result_summaries(state)
        has_patch = bool(state.get("patch"))
        patch_status = _patch_status(state)
        next_steps = _next_steps(state, has_patch)
        summary_parts = [
            f"任务状态：{state.get('status', 'unknown')}",
            f"候选文件：{len(candidate_files)} 个",
            "有 patch" if has_patch else "没有 patch",
        ]
        if not _verification_required(state):
            summary_parts.append("LLM 判定无需运行验证命令")
        elif test_results:
            summary_parts.append(f"验证结果：{test_results[-1]}")
        else:
            summary_parts.append("验证未运行")
        return {
            "summary": "；".join(summary_parts) + "。",
            "work_done": work_done,
            "candidate_files": candidate_files,
            "test_results": test_results,
            "has_patch": has_patch,
            "patch_status": patch_status,
            "next_steps": next_steps,
            "source": "rule_based",
        }


@dataclass
class LLMFinalReporter:
    llm_config: LLMConfig
    fallback: FinalReporter | None = None

    def __post_init__(self) -> None:
        self.fallback = self.fallback or RuleBasedFinalReporter()
        self.node = LLMJsonNode(
            name="final_reporter",
            llm_config=self.llm_config,
            system_prompt=load_prompt("system/final_reporter.md"),
            build_prompt=_final_report_prompt,
            fallback=lambda state, context: self.fallback.report(state) if self.fallback else {},
            response_model=FinalReportResponse,
            normalize=_normalize_final_report,
        )

    def report(self, state: AgentState) -> dict[str, Any]:
        return self.node.run(
            state,
            {"fallback_report": self.fallback.report(state) if self.fallback else {}},
        )


def _final_report_prompt(state: AgentState, context: dict[str, Any]) -> str:
    return render_prompt(
        "user/final_reporter.md",
        title=state.get("title", ""),
        description=state.get("description", ""),
        status=state.get("status", ""),
        error=state.get("error", ""),
        verification_required=json.dumps(_verification_required(state)),
        verification_reason=state.get("verification_reason", ""),
        plan=json.dumps(state.get("plan", []), ensure_ascii=False),
        candidate_files=json.dumps(state.get("candidate_files", []), ensure_ascii=False),
        test_results=json.dumps(_test_result_summaries(state), ensure_ascii=False),
        patch_summary=state.get("patch_summary") or "",
        has_patch=json.dumps(bool(state.get("patch"))),
        tool_calls=json.dumps(_tool_call_summaries(state), ensure_ascii=False, default=str),
        llm_observations=json.dumps(
            _trim_observations(state.get("llm_observations", [])),
            ensure_ascii=False,
            default=str,
        ),
        fallback_report=json.dumps(context.get("fallback_report", {}), ensure_ascii=False),
    )


def _normalize_final_report(
    data: dict[str, Any],
    state: AgentState,
    context: dict[str, Any],
) -> dict[str, Any]:
    fallback = context.get("fallback_report")
    if not isinstance(fallback, dict):
        fallback = RuleBasedFinalReporter().report(state)
    return {
        "summary": str(data.get("summary") or fallback.get("summary") or "").strip()[:1000],
        "work_done": _clean_list(data.get("work_done") or fallback.get("work_done"), 8, 280),
        "candidate_files": _clean_list(state.get("candidate_files"), 12, 260),
        "test_results": _clean_list(data.get("test_results") or fallback.get("test_results"), 8, 320),
        "has_patch": bool(state.get("patch")),
        "patch_status": str(data.get("patch_status") or fallback.get("patch_status") or "").strip()[:500],
        "next_steps": _clean_list(data.get("next_steps") or fallback.get("next_steps"), 8, 280),
    }


def _work_done_from_tools(tool_names: list[str]) -> list[str]:
    labels = {
        "list_files": "读取仓库文件结构",
        "search_code": "搜索相关代码",
        "search_code_context": "搜索结构化代码上下文",
        "read_file": "阅读候选文件",
        "run_tests": "处理验证命令",
        "git_diff": "检查工作区 diff",
        "write_memory": "写入任务记忆",
    }
    work_done: list[str] = []
    for name in tool_names:
        label = labels.get(name, name)
        if label and label not in work_done:
            work_done.append(label)
    return work_done or ["完成任务分析和运行状态整理"]


def _test_result_summaries(state: AgentState) -> list[str]:
    if not _verification_required(state):
        reason = str(state.get("verification_reason") or "LLM decided verification is not required.")
        return [f"跳过验证命令：{reason}"]
    results = []
    for item in state.get("test_results", [])[-5:]:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or state.get("verify_command") or "").strip()
        if item.get("skipped"):
            results.append(f"{command or 'verify command'} skipped: {item.get('reason', '')}")
            continue
        exit_code = item.get("exit_code")
        label = "passed" if exit_code == 0 else "failed"
        results.append(f"{command or 'verify command'} {label} with exit_code={exit_code}")
    return results


def _patch_status(state: AgentState) -> str:
    if state.get("patch_summary"):
        return str(state["patch_summary"])
    if state.get("patch"):
        return "工作区存在 patch。"
    return "未发现工作区 patch。"


def _next_steps(state: AgentState, has_patch: bool) -> list[str]:
    if state.get("error"):
        return [f"先处理当前错误：{state.get('error')}"]
    if not _verification_required(state):
        steps = ["如需验证行为，调整任务目标并让 LLM 判定需要运行验证命令。"]
    else:
        steps = []
    if has_patch:
        steps.append("审查当前 patch，并按项目规范补充或运行测试。")
    else:
        steps.append("根据候选文件继续定位问题，确认是否需要修改代码。")
    if not state.get("candidate_files"):
        steps.append("补充更具体的任务描述或搜索关键词以扩大代码定位范围。")
    return steps[:5]


def _verification_required(state: AgentState) -> bool:
    return bool(state.get("verification_required", True))


def _tool_call_summaries(state: AgentState) -> list[dict[str, Any]]:
    summaries = []
    for call in state.get("tool_calls", [])[-12:]:
        if not isinstance(call, dict):
            continue
        output = call.get("output")
        if not isinstance(output, dict):
            output = {}
        summaries.append(
            {
                "name": call.get("name"),
                "error": call.get("error"),
                "input": call.get("input"),
                "output_keys": sorted(output.keys()),
                "exit_code": output.get("exit_code"),
                "skipped": output.get("skipped"),
            }
        )
    return summaries


def _trim_observations(observations: Any) -> list[dict[str, Any]]:
    trimmed = []
    if not isinstance(observations, list):
        return trimmed
    for item in observations[-5:]:
        if not isinstance(item, dict):
            continue
        trimmed.append(
            {
                "latest_tool": item.get("latest_tool"),
                "status": item.get("status"),
                "summary": str(item.get("summary") or "")[:500],
                "new_findings": _clean_list(item.get("new_findings"), 5, 220),
                "missing_context": _clean_list(item.get("missing_context"), 5, 220),
            }
        )
    return trimmed


def _clean_list(value: Any, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:max_chars])
        if len(cleaned) >= limit:
            break
    return cleaned
