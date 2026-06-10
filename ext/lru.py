from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Entry:
    key: Any | None = None
    value: Any | None = None

@dataclass
class ListNode:
    entry: Entry | None = None
    next: ListNode | None = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.map: Dict[Any, Entry] = {}
        head = ListNode()
        self.cache: head

    def put(self, entry: Entry) -> None:
        ...

    def get(self, key: Any) -> Any:
        ...

    def move_to_head(self, entry: Entry) -> None:
        ...

    def delete_from_end(self) -> None:
        ...