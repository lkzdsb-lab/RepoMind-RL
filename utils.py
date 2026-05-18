import re
from datetime import datetime, timezone

def _tokens(text: str) -> list[str]:
    """
    从输入 text 获取不重复的词
    """
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+", text.lower())

def utc_now() -> str:
    """ 获取对应的时区的时间 """
    return datetime.now(timezone.utc).isoformat()