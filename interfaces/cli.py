"""Typer command entrypoint for Lee-Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from agent_runtime.executor import DebugAgent
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
    if not no_config:
        ensure_default_config_file(config_path)
    payload = {} if no_config else load_config_payload(config_path)
    config = debug_agent_config_from_dict(payload)
    if repo:
        config.repo_path = repo
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
    if not config.repo_path:
        config.repo_path = "."

    env_file = _resolve_config_path(config_path, config.env_file)
    load_env_file(env_file, override=config.env_override)
    normalize_project_runtime_paths(config)
    validate_debug_agent_config(config)
    return config


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
