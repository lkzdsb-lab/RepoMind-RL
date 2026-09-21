"""Internal task protocol; no model-controlled host paths or permissions."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Criterion(Contract):
    criterion_id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=2000)
    check: Literal["observation", "http_status", "exit_code", "body_contains", "output_contains"] = "observation"
    expected: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def valid_check(self):
        if self.check != "observation" and not self.expected:
            raise ValueError("A deterministic check requires expected")
        if self.check in {"http_status", "exit_code"}:
            int(self.expected)
        return self


class ExecutionTaskInput(Contract):
    objective: str = Field(min_length=1, max_length=4000)
    acceptance_criteria: list[Criterion] = Field(min_length=1, max_length=16)
    context: str = Field(default="", max_length=12000)

    @model_validator(mode="after")
    def unique_criteria(self):
        ids = [item.criterion_id for item in self.acceptance_criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion_id must be unique")
        return self


class Limits(Contract):
    timeout: int = Field(default=300, ge=10, le=1800)
    model_calls: int = Field(default=16, ge=1, le=64)
    tool_calls: int = Field(default=32, ge=1, le=128)
    retries: int = Field(default=8, ge=0, le=32)
    repeated_errors: int = Field(default=3, ge=1, le=8)
    startup_timeout: int = Field(default=30, ge=1, le=120)
    operation_timeout: int = Field(default=30, ge=1, le=300)
    cleanup_timeout: int = Field(default=5, ge=1, le=30)


class ResourceLimits(Contract):
    memory_limit_mb: int = Field(default=2048, ge=64, le=1048576)
    cpu_limit_percent: int = Field(default=50, ge=1, le=100)
    max_processes: int = Field(default=64, ge=8, le=4096)
    mode: Literal["strict", "best_effort"] = "strict"


class ResourceUsage(Contract):
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    capabilities: dict[str, str] = Field(default_factory=dict)
    peak_memory_bytes: int = 0
    peak_processes: int = 0
    cpu_percent: float = 0
    peak_cpu_percent: float = 0
    limit_event: str = ""


class ExecutionError(Contract):
    code: str
    category: Literal["validation", "transient", "permission", "budget", "runtime"]
    message: str
    retryable: bool = False


class CriterionResult(Contract):
    criterion_id: str
    verdict: Literal["pass", "fail", "inconclusive"]
    explanation: str = Field(max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=32)


class ExecutionResult(Contract):
    schema_version: int = 1
    task_id: str
    attempt_id: str
    execution_status: Literal["completed", "blocked", "failed", "timed_out", "cancelled"]
    verdict: Literal["pass", "fail", "inconclusive"] = "inconclusive"
    summary: str = Field(default="", max_length=4000)
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    cleanup_status: Literal["pending", "completed", "failed"] = "pending"
    source_fingerprint: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    resources: ResourceUsage = Field(default_factory=ResourceUsage)


class ProcessInput(Contract):
    action: Literal["start", "status", "read_output", "write_stdin", "stop", "list"]
    argv: list[str] = Field(default_factory=list, max_length=128)
    cwd: str = Field(default=".", max_length=1000)
    process_id: str = ""
    cursor: int = Field(default=0, ge=0)
    text: str = Field(default="", max_length=8000)


class RequestInput(Contract):
    process_id: str = Field(min_length=1, max_length=80)
    port: int = Field(ge=1, le=65535)
    path: str = Field(default="/", max_length=4000)
    method: Literal["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    body: str = Field(default="", max_length=16000)
    content_type: str = "application/json"
    timeout: int = Field(default=5, ge=1, le=20)


class ReadInput(Contract):
    path: str = Field(max_length=1000)
    start_line: int = Field(default=1, ge=1)


class EvidenceInput(Contract):
    evidence_id: str = Field(min_length=1, max_length=80)


class ListInput(Contract):
    pass


class Decision(Contract):
    action: Literal["managed_process", "local_request", "read_file", "list_files", "read_evidence", "finish", "blocked"]
    arguments_json: str = Field(default="{}", max_length=24000)
    summary: str = Field(default="", max_length=2000)
    criteria_results: list[CriterionResult] = Field(default_factory=list, max_length=16)
