"""Evidence coverage and deterministic acceptance checks shared by worker and Host."""
from .artifacts import EvidenceStore
from .contracts import Decision, ExecutionTaskInput


def validate_results(task: ExecutionTaskInput, decision: Decision, evidence: EvidenceStore) -> None:
    """ 来验证 subagent 最终提交的验收结论是否有合法证据支持"""
    expected = {item.criterion_id for item in task.acceptance_criteria}
    actual = [item.criterion_id for item in decision.criteria_results]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Return exactly one result for each acceptance criterion")
    # 1. 验收条件是否完整覆盖
    for item in decision.criteria_results:
        # 2. 证据 ID 是否真实存在
        if any(key not in evidence.records for key in item.evidence_ids):
            raise ValueError("Unknown evidence reference")
        # 3. pass/fail 必须引用运行证据
        if item.verdict != "inconclusive":
            records = [evidence.records[key] for key in item.evidence_ids]
            runtime = [record for record in records if record["result"].get("ok") and (
                (record["operation"] == "local_request" and "status_code" in record["result"])
                or (record["operation"] == "managed_process"
                    and record["arguments"].get("action") in {"start", "status", "read_output"}
                    and record["result"].get("exit_code") is not None))]
            if not runtime:
                raise ValueError("Each pass/fail requires a completed command or HTTP observation")
            runtime.sort(key=lambda record: record["created_at"])
            criterion = next(value for value in task.acceptance_criteria if value.criterion_id == item.criterion_id)
            # 4. 确定性条件会自动计算结果
            if criterion.check != "observation":
                field = {"http_status": "status_code", "exit_code": "exit_code",
                         "body_contains": "body", "output_contains": "output"}[criterion.check]
                observations = [record["result"] for record in runtime if record["result"].get(field) is not None]
                if not observations:
                    raise ValueError(f"Criterion {item.criterion_id} needs evidence containing {field}")
                latest = observations[-1]
                if criterion.check.endswith("contains"):
                    if latest.get("truncated") or latest.get("partial"):
                        raise ValueError("A substring check needs untruncated output")
                    passed = criterion.expected in str(latest[field])
                else:
                    passed = int(latest[field]) == int(criterion.expected)
                if item.verdict != ("pass" if passed else "fail"):
                    raise ValueError(f"Criterion {item.criterion_id} contradicts its deterministic check")


def aggregate_verdict(results):
    verdicts = {item.verdict for item in results}
    return "fail" if "fail" in verdicts else ("inconclusive" if "inconclusive" in verdicts else "pass")
