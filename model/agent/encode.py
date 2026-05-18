from dataclasses import dataclass
from typing import Any


@dataclass
class EncodedState:
    key: str
    features: dict[str, Any]