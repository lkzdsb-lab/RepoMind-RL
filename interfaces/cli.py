"""Typer command entrypoint for Lee-Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from agent_runtime.executor import DebugAgent
from agent_runtime.memory.archive.repository import repository_id
from agent_runtime.memory.catalog import SQLiteMemoryCatalog
from agent_runtime.memory.documents import MarkdownMemoryDocumentStore
from agent_runtime.memory.consolidation.factory import build_consolidation_pipeline
from agent_runtime.memory.domain.models import (
    MemoryQuery,
    MemoryStatus,
    MemoryType,
)
from agent_runtime.session import AgentSession
from agent_runtime.user_updates import set_change_event_sink
from config import (
    DEFAULT_CONFIG_PATH,
    DebugAgentConfig,
    debug_agent_config_from_dict,
    ensure_default_config_file,
    load_config_payload,
    load_env_file,
    normalize_project_runtime_paths,
    validate_debug_agent_config,
)
from interfaces.chat import ChatShell
from model.session import ChatResponse


app = typer.Typer(
    add_completion=False,
    help="Lee-Agent conversational coding/debugging CLI.",
    no_args_is_help=False,
)
console = Console()
memory_app = typer.Typer(
    add_completion=False,
    help="Inspect and maintain canonical Markdown long-term memory.",
)
catalog_app = typer.Typer(
    add_completion=False,
    help="Maintain the rebuildable SQLite memory catalog.",
)
app.add_typer(memory_app, name="memory")
memory_app.add_typer(catalog_app, name="catalog")


@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
    max_loops: Optional[int] = typer.Option(None, "--max-loops", help="Maximum agent loops."),
    manifest_dir: Optional[str] = typer.Option(None, "--manifest-dir", help="Runtime registry manifest directory."),
    code_context_index_path: Optional[str] = typer.Option(
        None,
        "--code-context-index-path",
        help="Path inside the target repo for the codebase context index.",
    ),
    resume_trace: Optional[str] = typer.Option(None, "--resume-trace", help="Load an existing trace."),
    rl_enabled: bool = typer.Option(False, "--rl-enabled", help="Enable Q-learning policy."),
    rl_epsilon: Optional[float] = typer.Option(None, "--rl-epsilon", help="RL exploration rate."),
    enable_editing: bool = typer.Option(False, "--enable-editing", help="Enable guarded edits."),
    require_step_approval: bool = typer.Option(
        False,
        "--require-step-approval",
        help="Require user approval before each agent action.",
    ),
    action_policy_mode: Optional[str] = typer.Option(
        None,
        "--action-policy-mode",
        help="Action policy mode: llm or rl.",
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Runtime log level."),
    console_log: bool = typer.Option(False, "--console-log", help="Also print runtime logs."),
) -> None:
    """Start chat mode when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return
    chat(
        repo=repo,
        config_path=config_path,
        no_config=no_config,
        max_loops=max_loops,
        manifest_dir=manifest_dir,
        code_context_index_path=code_context_index_path,
        resume_trace=resume_trace,
        rl_enabled=rl_enabled,
        rl_epsilon=rl_epsilon,
        enable_editing=enable_editing,
        require_step_approval=require_step_approval,
        action_policy_mode=action_policy_mode,
        log_level=log_level,
        console_log=console_log,
    )


@app.command()
def chat(
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
    max_loops: Optional[int] = typer.Option(None, "--max-loops", help="Maximum agent loops."),
    manifest_dir: Optional[str] = typer.Option(None, "--manifest-dir", help="Runtime registry manifest directory."),
    code_context_index_path: Optional[str] = typer.Option(
        None,
        "--code-context-index-path",
        help="Path inside the target repo for the codebase context index.",
    ),
    resume_trace: Optional[str] = typer.Option(None, "--resume-trace", help="Load an existing trace."),
    rl_enabled: bool = typer.Option(False, "--rl-enabled", help="Enable Q-learning policy."),
    rl_epsilon: Optional[float] = typer.Option(None, "--rl-epsilon", help="RL exploration rate."),
    enable_editing: bool = typer.Option(False, "--enable-editing", help="Enable guarded edits."),
    require_step_approval: bool = typer.Option(
        False,
        "--require-step-approval",
        help="Require user approval before each agent action.",
    ),
    action_policy_mode: Optional[str] = typer.Option(
        None,
        "--action-policy-mode",
        help="Action policy mode: llm or rl.",
    ),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Runtime log level."),
    console_log: bool = typer.Option(False, "--console-log", help="Also print runtime logs."),
) -> None:
    """Open a Codex-style chat session."""
    try:
        config = _build_config(
            repo=repo,
            config_path=config_path,
            no_config=no_config,
            max_loops=max_loops,
            manifest_dir=manifest_dir,
            code_context_index_path=code_context_index_path,
            rl_enabled=rl_enabled,
            rl_epsilon=rl_epsilon,
            enable_editing=enable_editing,
            require_step_approval=require_step_approval,
            action_policy_mode=action_policy_mode,
            log_level=log_level,
            console_log=console_log,
        )
        # 启动对话并初始化 agent
        shell = ChatShell(
            AgentSession(DebugAgent(config, user_update_sink=_render_live_user_update)),
            repo_path=config.repo_path,
            history_path=Path(config.repo_path) / ".repomind" / "chat_history",
            console=console,
        )
        set_change_event_sink(shell.render_live_change_event)
        agent_session = shell.agent_session
        initial_response = _load_initial_response(agent_session, resume_trace)
        shell.run(initial_response)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


@memory_app.command("list")
def memory_list(
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """List canonical Markdown memory documents."""
    store = _memory_store(repo, config_path, no_config)
    table = Table("ID", "Type", "Status", "Scope", "Title")
    for document in store.list():
        table.add_row(
            document.memory_id,
            document.memory_type.value,
            document.status.value,
            document.scope.level.value,
            document.title,
        )
    console.print(table)


@memory_app.command("show")
def memory_show(
    memory_id: str = typer.Argument(..., help="Memory document ID."),
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Print one canonical Markdown memory document."""
    store = _memory_store(repo, config_path, no_config)
    document = store.get(memory_id)
    if document is None:
        console.print(f"[red]Unknown memory:[/red] {memory_id}")
        raise typer.Exit(code=1)
    console.print(store.codec.encode(document).decode("utf-8"), markup=False)


@memory_app.command("validate")
def memory_validate(
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Validate layout, schema, IDs, and canonical formatting."""
    issues = _memory_store(repo, config_path, no_config).validate()
    if not issues:
        console.print("[green]Markdown memory store is valid.[/green]")
        return
    for issue in issues:
        console.print(f"[red]{issue.path}[/red]: {issue.message}")
    raise typer.Exit(code=1)


@memory_app.command("deprecate")
def memory_deprecate(
    memory_id: str = typer.Argument(..., help="Memory document ID."),
    reason: str = typer.Option(..., "--reason", help="Why this memory is deprecated."),
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Mark one memory deprecated without moving or deleting its document."""
    _, store, catalog = _memory_runtime(repo, config_path, no_config)
    try:
        store.deprecate(memory_id, reason)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Deprecated memory:[/green] {memory_id}")
    document = store.get(memory_id)
    try:
        if document is not None:
            catalog.synchronize(document, store.document_path(document))
    except Exception as exc:
        console.print(
            "[yellow]Markdown was updated, but Catalog synchronization failed:[/yellow] "
            f"{exc}"
        )
        raise typer.Exit(code=2) from exc


@memory_app.command("consolidate")
def memory_consolidate(
    task_id: str = typer.Argument(..., help="Completed Task Archive ID."),
    extractor_mode: Optional[str] = typer.Option(
        None, "--extractor-mode", help="Extraction mode: rule_based or llm."
    ),
    pipeline_version: Optional[str] = typer.Option(
        None, "--pipeline-version", help="Immutable consolidation pipeline version."
    ),
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Consolidate one verified Task Archive into Markdown memory documents."""
    config = _load_base_config(repo, config_path, no_config)
    if extractor_mode is not None:
        config.memory_extractor_mode = extractor_mode
    if pipeline_version is not None:
        config.consolidation_pipeline_version = pipeline_version
    env_file = _resolve_config_path(config_path, config.env_file)
    load_env_file(env_file, override=config.env_override)
    normalize_project_runtime_paths(config)
    try:
        outcome = build_consolidation_pipeline(config).consolidate(task_id)
    except Exception as exc:
        console.print(f"[red]Consolidation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table("Field", "Value")
    table.add_row("task_id", outcome.task_id)
    table.add_row("status", outcome.status)
    table.add_row("pipeline", outcome.pipeline_version)
    table.add_row("extractor", outcome.extractor_source)
    table.add_row("memories", ", ".join(outcome.memory_ids) or "(none)")
    table.add_row("rejected", str(outcome.rejected_candidates))
    table.add_row("run_path", outcome.run_path)
    if outcome.warnings:
        table.add_row("warnings", "\n".join(outcome.warnings))
    console.print(table)


@memory_app.command("search")
def memory_search(
    text: str = typer.Argument(..., help="Keyword query."),
    memory_type: Optional[str] = typer.Option(
        None, "--type", help="Comma-separated memory types."
    ),
    status: Optional[str] = typer.Option(
        None, "--status", help="Comma-separated statuses; defaults to active states."
    ),
    tag: Optional[str] = typer.Option(None, "--tag", help="Comma-separated required tags."),
    scope: Optional[str] = typer.Option(
        None, "--scope", help="Comma-separated exact module/file/symbol hints."
    ),
    limit: int = typer.Option(8, "--limit", min=1, max=100),
    all_repos: bool = typer.Option(
        False, "--all-repos", help="Do not apply current repository scope filtering."
    ),
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Search the SQLite keyword projection and explain each result."""
    config, store, catalog = _memory_runtime(repo, config_path, no_config)
    try:
        query = MemoryQuery(
            text=text,
            repo_id="" if all_repos else repository_id(config.repo_path),
            memory_types=tuple(
                MemoryType(value) for value in _comma_values(memory_type)
            ),
            statuses=tuple(MemoryStatus(value) for value in _comma_values(status)),
            tags=tuple(_comma_values(tag)),
            scope_hints=tuple(_comma_values(scope)),
            limit=limit,
        )
        hits = catalog.keyword_search(query)
    except Exception as exc:
        console.print(f"[red]Memory search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table("Score", "ID", "Type", "Status", "Title", "Reasons")
    for hit in hits:
        document = store.get(hit.memory_id)
        table.add_row(
            f"{hit.score:.4f}",
            hit.memory_id,
            document.memory_type.value if document else "?",
            document.status.value if document else "?",
            document.title if document else "(Markdown missing)",
            "; ".join(hit.reasons),
        )
    console.print(table)


@catalog_app.command("rebuild")
def memory_catalog_rebuild(
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Build a temporary catalog and atomically replace the current projection."""
    _, store, catalog = _memory_runtime(repo, config_path, no_config)
    issues = store.validate()
    if issues:
        for issue in issues:
            console.print(f"[red]{issue.path}[/red]: {issue.message}")
        raise typer.Exit(code=1)
    try:
        result = catalog.rebuild(store.records())
    except Exception as exc:
        console.print(f"[red]Catalog rebuild failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Catalog rebuilt.[/green] "
        f"indexed={result.indexed} unchanged={result.unchanged} removed={result.removed}"
    )


@catalog_app.command("sync")
def memory_catalog_sync(
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Incrementally synchronize Markdown and remove orphaned projections."""
    _, store, catalog = _memory_runtime(repo, config_path, no_config)
    issues = store.validate()
    if issues:
        for issue in issues:
            console.print(f"[red]{issue.path}[/red]: {issue.message}")
        raise typer.Exit(code=1)
    try:
        result = catalog.sync(store.records())
    except Exception as exc:
        console.print(f"[red]Catalog synchronization failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Catalog synchronized.[/green] "
        f"indexed={result.indexed} unchanged={result.unchanged} removed={result.removed}"
    )


@catalog_app.command("status")
def memory_catalog_status(
    repo: Optional[str] = typer.Option(None, "--repo", help="Target repository path."),
    config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Runtime config file."),
    no_config: bool = typer.Option(False, "--no-config", help="Do not load config.json."),
) -> None:
    """Check schema, hashes, paths, FTS rows, and orphan projections."""
    try:
        _, store, catalog = _memory_runtime(repo, config_path, no_config)
        report = catalog.status(store.records())
    except Exception as exc:
        console.print(f"[red]Catalog status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Markdown documents={report.document_count}, Catalog rows={report.catalog_count}"
    )
    if report.healthy:
        console.print("[green]Catalog is consistent with Markdown memory.[/green]")
        return
    for issue in report.issues:
        identity = f" [{issue.memory_id}]" if issue.memory_id else ""
        console.print(f"[red]{issue.code}{identity}[/red]: {issue.message}")
    raise typer.Exit(code=1)


def _build_config(
    *,
    repo: str | None,
    config_path: str,
    no_config: bool,
    max_loops: int | None,
    manifest_dir: str | None,
    code_context_index_path: str | None,
    rl_enabled: bool,
    rl_epsilon: float | None,
    enable_editing: bool,
    require_step_approval: bool,
    action_policy_mode: str | None,
    log_level: str | None,
    console_log: bool,
) -> DebugAgentConfig:
    config = _load_base_config(repo, config_path, no_config)
    if max_loops is not None:
        config.max_loops = max_loops
    if manifest_dir is not None:
        config.manifest_dir = manifest_dir
    if code_context_index_path is not None:
        config.code_context_index_path = code_context_index_path
    if rl_enabled:
        config.rl_enabled = True
    if rl_epsilon is not None:
        config.rl_epsilon = rl_epsilon
    if enable_editing:
        config.editing_enabled = True
    if require_step_approval:
        config.require_step_approval = True
    if action_policy_mode:
        config.action_policy_mode = action_policy_mode
    if log_level:
        config.log_level = log_level
    if not console_log:
        config.log_to_console = False
    env_file = _resolve_config_path(config_path, config.env_file)
    load_env_file(env_file, override=config.env_override)
    normalize_project_runtime_paths(config)
    validate_debug_agent_config(config)
    return config


def _load_base_config(
    repo: str | None,
    config_path: str,
    no_config: bool,
) -> DebugAgentConfig:
    if not no_config:
        ensure_default_config_file(config_path)
    payload = {} if no_config else load_config_payload(config_path)
    config = debug_agent_config_from_dict(payload)
    if repo:
        config.repo_path = repo
    if not config.repo_path:
        config.repo_path = "."
    return config


def _memory_store(
    repo: str | None,
    config_path: str,
    no_config: bool,
) -> MarkdownMemoryDocumentStore:
    _, store, _ = _memory_runtime(repo, config_path, no_config)
    return store


def _memory_runtime(
    repo: str | None,
    config_path: str,
    no_config: bool,
) -> tuple[DebugAgentConfig, MarkdownMemoryDocumentStore, SQLiteMemoryCatalog]:
    config = _load_base_config(repo, config_path, no_config)
    normalize_project_runtime_paths(config)
    return (
        config,
        MarkdownMemoryDocumentStore.from_config(config),
        SQLiteMemoryCatalog.from_config(config),
    )


def _comma_values(value: str | None) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in str(value or "").split(",")
            if item.strip()
        )
    )


def _load_initial_response(
    agent_session: AgentSession,
    resume_trace: str | None,
) -> ChatResponse | None:
    """
        加载历史对话
    """
    if not resume_trace:
        return None
    state = _load_trace_state(resume_trace)
    return agent_session.load_state(state, trace_path=resume_trace)


def _load_trace_state(path_value: str) -> dict:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Trace file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Trace path is not a file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Trace file must contain a JSON object: {path}")
    return data


def _resolve_config_path(config_path: str | None, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(config_path or DEFAULT_CONFIG_PATH)
    if not base.is_absolute():
        base = Path.cwd() / base
    return base.parent / path


def _render_live_user_update(update: dict) -> None:
    """
        回调函数展示
    """
    message = str(update.get("message") or "").strip()
    if not message:
        return
    source = str(update.get("source") or "agent").strip() or "agent"
    console.print(f"[dim]{source}[/dim] {message}")


if __name__ == "__main__":
    main()
