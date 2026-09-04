"""Conservative deterministic boundaries around LLM constraint semantics."""

from __future__ import annotations

from dataclasses import dataclass


_TEMPORARY_MARKERS = (
    "这次", "本次", "本轮", "当前对话", "暂时", "先不", "现在先",
    "this time", "for now", "in this task", "for this task",
)
_DURABLE_MARKERS = (
    "以后", "始终", "总是", "默认", "长期", "每次", "我偏好", "我习惯",
    "in future", "always", "by default", "i prefer", "my preference",
)
_REPOSITORY_MARKERS = (
    "这个仓库", "本仓库", "这个项目", "本项目", "当前仓库",
    "this repository", "this repo", "this project",
)
_USER_MARKERS = (
    "所有项目", "我的项目", "对我", "我偏好", "我习惯",
    "all projects", "my projects", "i prefer", "my preference",
)


@dataclass(frozen=True)
class ConstraintDecision:
    scope: str
    durability: str
    explicit: bool
    confidence: float


def classify_explicit_constraint(content: str) -> ConstraintDecision:
    text = normalize_constraint_text(content)
    if has_temporary_marker(text):
        return ConstraintDecision("task", "temporary", True, 1.0)
    if not any(marker in text for marker in _DURABLE_MARKERS):
        return ConstraintDecision("ambiguous", "ambiguous", False, 0.0)
    if has_repository_marker(text):
        return ConstraintDecision("repository", "durable", True, 0.95)
    if any(marker in text for marker in _USER_MARKERS):
        return ConstraintDecision("user", "durable", True, 0.95)
    return ConstraintDecision("ambiguous", "ambiguous", True, 0.5)


def has_temporary_marker(content: str) -> bool:
    text = normalize_constraint_text(content)
    return any(marker in text for marker in _TEMPORARY_MARKERS)


def has_repository_marker(content: str) -> bool:
    text = normalize_constraint_text(content)
    return any(marker in text for marker in _REPOSITORY_MARKERS)


def normalize_constraint_text(content: str) -> str:
    return " ".join(str(content or "").casefold().split())
