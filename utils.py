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

def _truncate(value: str, max_chars: int) -> str:
    """
        对 str 进行裁剪
    """
    text = str(value or "").strip()
    return text[:max_chars]


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
    return max(0.0, max(default, parsed))


def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    """
        安全添加元素
    """
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    target[key] = value


def _clean_string_list(value: Any, limit: int, max_chars: int | None) -> list[str]:
    """
        提取干净的字符串列表
    """
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            if max_chars and len(text) > max_chars:
                cleaned.append(text[:max_chars])
            else:
                cleaned.append(text)
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

def _language_counts(index: Any) -> dict[str, int]:
    """ 获取 不同语言文件类型 个数"""
    counts: dict[str, int] = {}
    for entry in index.tree:
        counts[entry.language] = counts.get(entry.language, 0) + 1
    return counts


def _line_number(content: str, offset: int) -> int:
    """ 获取行数 """
    return content.count("\n", 0, offset) + 1


def _line_at(content: str, offset: int) -> str:
    """ 获取每一行的内容 """
    start = content.rfind("\n", 0, offset) + 1
    end = content.find("\n", offset)
    if end == -1:
        end = len(content)
    return content[start:end]


def _embedding_doc(
    doc_id: str,
    kind: str,
    title: str,
    content: str,
    file_path: str = "",
    symbol: str = "",
    metadata: dict | None = None,
) -> Any:
    from agent_runtime.codebase_context.models import EmbeddingDoc

    return EmbeddingDoc(
        doc_id=doc_id,
        kind=kind,
        title=title,
        content=content[:4000],
        file_path=file_path,
        symbol=symbol,
        tokens=sorted(set(_tokens(" ".join([title, content])))),
        metadata=metadata or {},
    )
