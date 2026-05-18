"""Loguru-based logging setup for the agent runtime."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from loguru import logger


_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """
        将系统所有的 log 统一使用 loguru
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(
    *,
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_logs: bool = False,
    console: bool = True,
    force: bool = False,
) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = _parse_level_name(level)
    logger.remove()

    if console:
        logger.add(
            sys.stderr,
            level=level_name,
            serialize=json_logs,
            format=_text_format(),
            colorize=not json_logs,
            backtrace=False,
            diagnose=False,
        )

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            path.as_posix(),
            level=level_name,
            serialize=json_logs,
            format=_text_format(),
            rotation="10 MB",
            retention="14 days",
            enqueue=True,
            encoding="utf-8",
            backtrace=False,
            diagnose=False,
        )

    root = logging.getLogger()
    root.handlers = [InterceptHandler()]
    root.setLevel(_parse_stdlib_level(level_name))
    logging.captureWarnings(True)

    _CONFIGURED = True


def configure_from_agent_config(config: Any, *, force: bool = False) -> None:
    log_file = getattr(config, "log_file", None)
    if log_file:
        path = Path(str(log_file))
        if not path.is_absolute():
            path = Path(str(config.repo_path)) / path
        log_file = path
    configure_logging(
        level=str(getattr(config, "log_level", "INFO")),
        log_file=log_file,
        json_logs=bool(getattr(config, "log_json", False)),
        console=bool(getattr(config, "log_to_console", True)),
        force=force,
    )


def _parse_level_name(level: str) -> str:
    """
        设置日志上报等级
    """
    normalized = level.strip().upper()
    if normalized in {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}:
        return normalized
    return "INFO"


def _parse_stdlib_level(level: str) -> int:
    if level == "TRACE":
        return logging.DEBUG
    if level == "SUCCESS":
        return logging.INFO
    return int(getattr(logging, level, logging.INFO))


def _text_format() -> str:
    """
        日志上报模版
    """
    return (
        "<green>{time:YYYY-MM-DDTHH:mm:ss.SSSZZ}</green> "
        "<level>{level: <8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "<level>{message}</level> "
        "<dim>{extra}</dim>"
    )
