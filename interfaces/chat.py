"""Codex-style terminal chat loop for Lee-Agent."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_runtime.session import AgentSession
from tools.git_tools.diff import git_diff
from model.session import ChatResponse


class ChatShell:
    def __init__(
        self,
        agent_session: AgentSession,
        *,
        repo_path: str,
        history_path: str | Path,
        console: Console | None = None,
    ) -> None:
        self.agent_session = agent_session
        self.repo_path = repo_path
        self.console = console or Console()
        self._rendered_change_events = 0
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt = PromptSession(history=FileHistory(str(history_path)))

    def run(self, initial_response: ChatResponse | None = None) -> None:
        self._render_banner()
        if initial_response is not None:
            self.render_response(initial_response)

        while True:
            try:
                text = self.prompt.prompt("lee-agent> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye[/dim]")
                return
            if not text:
                continue
            if text.startswith("/"):
                if not self._handle_command(text):
                    return
                continue
            response = self.agent_session.send(text)
            self.render_response(response)

    def render_response(self, response: ChatResponse) -> None:
        self._render_user_updates(response.user_updates)
        self._render_change_events(response.change_events)
        if response.type == "needs_user_input":
            body = response.reason or "The agent needs more information before continuing."
            if response.questions:
                body += "\n\n" + "\n".join(
                    f"{index}. {question}"
                    for index, question in enumerate(response.questions, start=1)
                )
            title = "Approval" if _is_step_approval_response(response) else "Question"
            self.console.print(Panel(body, title=title, border_style="yellow"))
            if title == "Approval":
                self.console.print(
                    "[dim]Reply approve/yes/同意 to execute; any other reply becomes feedback.[/dim]"
                )
            else:
                self.console.print("[dim]Reply normally; your next message will resume the run.[/dim]")
            return

        style = "red" if response.type == "failed" else "green"
        self.console.print(Panel(response.message or response.type, title="Agent", border_style=style))
        details = self._response_table(response)
        if details.row_count:
            self.console.print(details)

    def _handle_command(self, text: str) -> bool:
        command, _, arg = text.partition(" ")
        command = command.strip().lower()
        arg = arg.strip()
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self._render_help()
            return True
        if command == "/status":
            self._render_status()
            return True
        if command == "/trace":
            trace = self.agent_session.last_trace_path or "(no trace yet)"
            self.console.print(trace)
            return True
        if command == "/new":
            self.agent_session.reset()
            self._rendered_change_events = 0
            self.console.print("[green]Started a new conversation state.[/green]")
            return True
        if command == "/diff":
            self._render_diff()
            return True
        if command == "/resume":
            self.console.print(
                "[yellow]/resume is available from the startup flag: "
                "lee-agent chat --resume-trace <path>[/yellow]"
            )
            return True
        self.console.print(f"[red]Unknown command:[/red] {command}. Type /help.")
        return True

    def _render_banner(self) -> None:
        self.console.print(
            Panel(
                "Type a task to start. If the agent asks a question, reply normally.\n"
                "Commands: /help, /status, /trace, /diff, /new, /exit",
                title="Lee-Agent Chat",
                border_style="cyan",
            )
        )

    def _render_help(self) -> None:
        table = Table(title="Commands")
        table.add_column("Command")
        table.add_column("Description")
        table.add_row("/help", "Show this help.")
        table.add_row("/status", "Show current task state.")
        table.add_row("/trace", "Print the latest trace path.")
        table.add_row("/diff", "Show current repository diff summary.")
        table.add_row("/new", "Forget current in-memory conversation state.")
        table.add_row("/exit", "Quit.")
        self.console.print(table)

    def _render_status(self) -> None:
        state = self.agent_session.state or {}
        table = Table(title="Session Status")
        table.add_column("Field")
        table.add_column("Value")
        for key in (
            "task_id",
            "status",
            "current_step",
            "repo_path",
            "loop_count",
            "max_loops",
        ):
            table.add_row(key, str(state.get(key, "")))
        table.add_row("trace_path", self.agent_session.last_trace_path)
        self.console.print(table)

    def _render_diff(self) -> None:
        state = self.agent_session.state or {}
        change_events = state.get("change_events", []) if isinstance(state, dict) else []
        if isinstance(change_events, list) and change_events:
            for index, event in enumerate(change_events, start=1):
                if not isinstance(event, dict):
                    continue
                panel = _render_change_event_panel(event, title=f"Change #{index}")
                if panel is not None:
                    self.console.print(panel)
            return
        output = git_diff(self.repo_path)
        if output.get("error"):
            self.console.print(f"[red]{output['error']}[/red]")
            return
        diff = str(output.get("diff") or "")
        if not diff:
            self.console.print("[dim]No change events in this session, and no git diff.[/dim]")
            return
        self.console.print(Panel(diff[-12000:], title="git diff", border_style="blue"))

    def _response_table(self, response: ChatResponse) -> Table:
        table = Table(show_header=False)
        table.add_column("Field", style="dim")
        table.add_column("Value")
        _add_row(table, "trace", response.trace_path)
        _add_row(table, "edited_files", ", ".join(response.edited_files))
        _add_row(table, "candidate_files", ", ".join(response.candidate_files[:8]))
        _add_row(table, "patch", response.patch_summary)
        _add_row(table, "llm_tokens", _format_token_usage(response.llm_token_usage))
        _add_row(table, "llm_error", _format_latest_llm_error(response.llm_errors))
        if response.test_results:
            latest = response.test_results[-1]
            command = latest.get("command", "")
            exit_code = latest.get("exit_code", "")
            _add_row(table, "latest_test", f"{command} exit_code={exit_code}")
        return table

    def _render_user_updates(self, updates: list[dict[str, Any]]) -> None:
        for item in updates:
            message = str(item.get("message") or "").strip()
            if not message:
                continue
            source = str(item.get("source") or "agent").strip() or "agent"
            self.console.print(f"[dim]{source}[/dim] {message}")

    def _render_change_events(self, events: list[dict[str, Any]]) -> None:
        if not isinstance(events, list) or self._rendered_change_events >= len(events):
            return
        for event in events[self._rendered_change_events :]:
            if not isinstance(event, dict):
                continue
            panel = _render_change_event_panel(event)
            if panel is not None:
                self.console.print(panel)
        self._rendered_change_events = len(events)

    def render_live_change_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        panel = _render_change_event_panel(event)
        if panel is not None:
            self.console.print(panel)
        self._rendered_change_events += 1


def _add_row(table: Table, key: str, value: Any) -> None:
    text = str(value or "").strip()
    if text:
        table.add_row(key, text)


def _format_change_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return ""
    files = summary.get("files")
    if not isinstance(files, list):
        files = []
    file_parts = []
    for item in files[:5]:
        if not isinstance(item, dict):
            continue
        path = str(item.get("file_path") or "").strip()
        if not path:
            continue
        file_parts.append(f"{path} +{item.get('added', 0)} -{item.get('removed', 0)}")
    if file_parts:
        suffix = " ..." if len(files) > 5 else ""
        return "; ".join(file_parts) + suffix
    return str(summary.get("summary") or "").strip()


def _render_change_event_panel(event: dict[str, Any], *, title: str = "Code Changes") -> Panel | None:
    files = [str(path).strip() for path in event.get("files", []) or [] if str(path).strip()]
    diff_summary = event.get("diff_summary") if isinstance(event.get("diff_summary"), dict) else {}
    total_added = int(diff_summary.get("total_added") or 0)
    total_removed = int(diff_summary.get("total_removed") or 0)
    changed_line_count = int(event.get("changed_line_count") or 0)
    header = Text()
    if files:
        header.append(", ".join(files), style="bold")
    else:
        header.append("Modified files", style="bold")
    stats = f"  (+{total_added} -{total_removed}, changed {changed_line_count} lines)"
    header.append(stats, style="dim")

    body: list[Any] = [header]
    reason = str(event.get("reason") or "").strip()
    if reason:
        body.append(Text(reason, style="dim"))

    for file_item in event.get("hunks", []) or []:
        if not isinstance(file_item, dict):
            continue
        file_path = str(file_item.get("file_path") or "").strip()
        if file_path:
            body.append(Text(file_path, style="bold cyan"))
        for hunk in file_item.get("hunks", []) or []:
            if not isinstance(hunk, dict):
                continue
            header_line = str(hunk.get("header") or "").strip()
            if header_line:
                body.append(Text(header_line, style="cyan"))
            body.extend(_render_hunk_lines(hunk.get("lines") or []))

    return Panel(Group(*body), title=title, border_style="blue")


def _render_hunk_lines(lines: list[dict[str, Any]]) -> list[Text]:
    rendered: list[Text] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not isinstance(line, dict):
            index += 1
            continue
        line_type = str(line.get("type") or "context")
        if (
            line_type == "remove"
            and index + 1 < len(lines)
            and isinstance(lines[index + 1], dict)
            and str(lines[index + 1].get("type") or "") == "add"
        ):
            remove_text = str(line.get("text") or "")
            add_text = str(lines[index + 1].get("text") or "")
            rendered.append(_highlight_changed_line(remove_text, add_text, removed=True))
            rendered.append(_highlight_changed_line(remove_text, add_text, removed=False))
            index += 2
            continue
        rendered.append(_plain_diff_line(str(line.get("text") or ""), line_type))
        index += 1
    return rendered


def _plain_diff_line(text: str, line_type: str) -> Text:
    style = "dim"
    if line_type == "add":
        style = "green"
    elif line_type == "remove":
        style = "red"
    return Text(text, style=style)


def _highlight_changed_line(remove_text: str, add_text: str, *, removed: bool) -> Text:
    source = remove_text if removed else add_text
    base_style = "red" if removed else "green"
    accent_style = "bold white on red" if removed else "bold black on green"
    matcher = difflib.SequenceMatcher(a=remove_text[1:], b=add_text[1:])
    text = Text(source[:1], style=base_style)
    source_body = source[1:]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if removed:
            segment = source_body[i1:i2]
            changed = tag in {"replace", "delete"}
        else:
            segment = source_body[j1:j2]
            changed = tag in {"replace", "insert"}
        if not segment:
            continue
        text.append(segment, style=accent_style if changed else base_style)
    return text


def _format_token_usage(usage: dict[str, Any]) -> str:
    if not isinstance(usage, dict) or not usage:
        return ""
    total = int(usage.get("total_tokens") or 0)
    requests = int(usage.get("request_count") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    if not total and not requests:
        return ""
    return f"total={total}, prompt={prompt}, completion={completion}, requests={requests}"


def _format_latest_llm_error(errors: list[dict[str, Any]]) -> str:
    if not isinstance(errors, list) or not errors:
        return ""
    latest = errors[-1]
    if not isinstance(latest, dict):
        return ""
    category = str(latest.get("category") or latest.get("type") or "llm_error")
    node = str(latest.get("node") or "unknown")
    message = " ".join(str(latest.get("message") or "").split())
    if len(message) > 180:
        message = message[:177] + "..."
    return f"{category} at {node}: {message}"


def _is_step_approval_response(response: ChatResponse) -> bool:
    state = response.state if isinstance(response.state, dict) else {}
    return state.get("current_step") == "awaiting_step_approval"
