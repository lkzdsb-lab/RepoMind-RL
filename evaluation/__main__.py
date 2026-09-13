from __future__ import annotations

import argparse

from evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Lee-Agent conversation evaluation.")
    parser.add_argument("--case", required=True, help="Path to an evaluation case JSON file.")
    parser.add_argument("--config", default="config.json", help="Agent config JSON file.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep a successful workspace.")
    parser.add_argument("--timeout", type=int, default=None, help="Override the case timeout in seconds.")
    args = parser.parse_args()
    result = run_evaluation(
        args.case,
        config_path=args.config,
        keep_workspace=args.keep_workspace,
        timeout_override=args.timeout,
    )
    if isinstance(result, dict):
        print(f"{result['metric']}: {result['recall']}")
        print(f"passed: {result['passed']}")
        print(f"report: {result['artifacts']['report']}")
        raise SystemExit(0 if result["passed"] else 1)
    print(f"case_id: {result.case_id}")
    print(f"passed: {result.passed}")
    print(f"score: {result.score}/100")
    print(f"findings: {result.findings_found}/{result.findings_total}")
    print(f"verification_exit_code: {result.verification.exit_code}")
    print(f"changed_files: {', '.join(result.changed_files) or '(none)'}")
    print(f"loops/tools/llm_calls: {result.loops}/{result.tool_calls}/{result.llm_calls}")
    print(f"total_tokens: {result.total_tokens}")
    print(f"report: {result.artifacts.get('report', '')}")
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
