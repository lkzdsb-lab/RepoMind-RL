from dataclasses import dataclass, field, asdict
from typing import Any
from utils import utc_now
from uuid import uuid4


@dataclass
class Transition:
    state_key: str
    action: str
    reward: float
    next_state_key: str
    done: bool = False
    state_features: dict[str, Any] = field(default_factory=dict)
    next_state_features: dict[str, Any] = field(default_factory=dict)
    action_args: dict[str, Any] = field(default_factory=dict)
    reward_reasons: list[str] = field(default_factory=list)
    task_id: str = ""
    transition_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})