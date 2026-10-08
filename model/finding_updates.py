"""Commands for changing candidates; stored findings retain their existing schema."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FindingOperation(str, Enum):
    CREATE = "create"
    ENRICH = "enrich"
    REVISE = "revise"


class FindingChange(str, Enum):
    CREATED = "created"
    METADATA = "metadata"
    EVIDENCE = "evidence"
    REVISED = "revised"
    UNCHANGED = "unchanged"


class FindingUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    operation: FindingOperation
    change: FindingChange
    evidence_refs: list[str]
    evidence_keys: list[str]
    reason: str = ""


class FindingUpdateResult(BaseModel):
    findings: list[dict[str, Any]] = Field(default_factory=list)
    events: list[FindingUpdateEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
