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

def _clamp_float(value: Any, info: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{info}: {value}")
    return max(0.0, min(1.0, parsed))

def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    """
        安全添加元素
    """
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    target[key] = value