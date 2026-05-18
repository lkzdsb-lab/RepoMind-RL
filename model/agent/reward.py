from dataclasses import dataclass
from typing import Any


@dataclass
class RewardBreakdown:
    reward: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"reward": self.reward, "reasons": self.reasons}