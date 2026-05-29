import re
import json
from datetime import datetime, timezone
from typing import Any


def _tokens(text: str) -> list[str]:
    """
    从输入 text 获取不重复的词
    """
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+", text.lower())


def utc_now() -> str:
    """ 获取对应的时区的时间 """
    return datetime.now(timezone.utc).isoformat()


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def _truncate_text(text: str, limit: int) -> str:
    """
        对于过长的返回进行拼接
    """
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _safe_float(value: Any, default: float) -> float:
    """
        安全转义
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value: Any, default: float, info: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{info}: {value}")
    return max(0.0, min(default, parsed))


def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    """
        安全添加元素
    """
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    target[key] = value


def _clean_string_list(value: Any, limit: int, max_chars: int) -> list[str]:
    """
        提取干净的字符串列表
    """
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:max_chars])
        if 0 <= limit <= len(cleaned):
            break
    return cleaned

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "approved"}
    return bool(value)

def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
