from dataclasses import dataclass, field
from typing import Any

@dataclass
class SearchQueryPlan:
    query: str
    terms: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)
    code_terms: list[str] = field(default_factory=list)
    memory_terms: list[str] = field(default_factory=list)
    skill_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "terms": self.terms,
            "identifiers": self.identifiers,
            "domain_terms": self.domain_terms,
            "code_terms": self.code_terms,
            "memory_terms": self.memory_terms,
            "skill_terms": self.skill_terms,
        }
