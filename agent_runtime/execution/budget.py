"""A single recovery budget shared by model, tool and result-validation failures."""
from __future__ import annotations

import time


class BudgetExceeded(RuntimeError):
    pass


class RecoveryBudget:
    def __init__(self, limit: int, counters: dict[str, int], remaining):
        self.limit, self.counters, self.remaining = limit, counters, remaining

    def recover(self, delay: float = 0) -> None:
        if self.counters["retries"] >= self.limit:
            raise BudgetExceeded("Recovery budget exhausted")
        self.counters["retries"] += 1
        if delay:
            time.sleep(min(delay, self.remaining()))
