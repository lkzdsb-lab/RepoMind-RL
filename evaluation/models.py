from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class EvaluationTurn(BaseModel):
    message: str = Field(min_length=1)


class ExpectedFinding(BaseModel):
    bug_id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    symbol: str = ""
    category: str = ""
    keywords: list[str] = Field(default_factory=list)


class VerificationSpec(BaseModel):
    argv: list[str] = Field(min_length=1)
    timeout: int = Field(default=120, ge=1, le=1800)
    http_baseline: str = ""
    skip_race: bool = False


class EvaluationLimits(BaseModel):
    case_timeout: int = Field(default=600, ge=1, le=7200)
    max_loops: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=30, ge=1)
    max_total_tokens: int = Field(default=80000, ge=1)


class EvaluationCase(BaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    version: int = Field(default=1, ge=1)
    fixture: str = Field(min_length=1)
    intent: Literal["diagnose", "implement", "review"]
    turns: list[EvaluationTurn] = Field(min_length=1)
    expected_findings: list[ExpectedFinding] = Field(default_factory=list)
    bug_baseline: str = ""
    bug_difficulties: list[Literal["easy", "medium", "hard"]] = Field(default_factory=list)
    allowed_changed_files: list[str] = Field(default_factory=list)
    forbidden_changed_files: list[str] = Field(default_factory=list)
    verification: VerificationSpec
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)

    @model_validator(mode="after")
    def _validate_change_contract(self) -> "EvaluationCase":
        if self.intent != "implement" and self.allowed_changed_files:
            raise ValueError("non-implement cases cannot allow repository changes")
        return self


class VerificationResult(BaseModel):
    argv: list[str]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    start_error: str = ""
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.timed_out and not self.start_error and self.exit_code == 0


class FindingMatch(BaseModel):
    bug_id: str
    matched: bool
    source: str = ""


class EvaluationResult(BaseModel):
    run_id: str
    case_id: str
    case_version: int
    passed: bool
    score: int = Field(ge=0, le=100)
    agent_status: str = ""
    fixture_hash: str
    config_fingerprint: str
    findings_found: int = 0
    findings_total: int = 0
    finding_matches: list[FindingMatch] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    unauthorized_changes: list[str] = Field(default_factory=list)
    verification: VerificationResult
    loops: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
