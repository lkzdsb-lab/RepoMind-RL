"""Evaluator-owned black-box acceptance runner; no new Go unit tests or LLM calls."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from evaluation.bug_baseline import load_baseline


def go_binary() -> str:
    configured = os.environ.get("EVALUATION_GO", "")
    found = configured or shutil.which("go")
    if not found:
        raise RuntimeError("Go unavailable; put go on PATH or set EVALUATION_GO")
    return found


def remaining(deadline: float, maximum: float = 120) -> float:
    seconds = min(maximum, deadline - time.monotonic())
    if seconds <= 0:
        raise TimeoutError("acceptance deadline exceeded")
    return seconds


def command(argv: list[str], cwd: Path, deadline: float) -> dict:
    cache_root = Path(__file__).resolve().parent.parent / ".repomind" / "evaluation" / "go-cache"
    for name in ("build", "tmp", "modules"):
        (cache_root / name).mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GOCACHE": str(cache_root / "build"),
                   "GOTMPDIR": str(cache_root / "tmp"), "GOMODCACHE": str(cache_root / "modules"),
                   "TEMP": str(cache_root / "tmp"), "TMP": str(cache_root / "tmp")}
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                            env=environment,
                            encoding="utf-8", errors="replace", timeout=remaining(deadline),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"argv": argv, "exit_code": result.returncode,
            "stdout": result.stdout[-16000:], "stderr": result.stderr[-16000:]}


@contextmanager
def running_server(binary: Path, workspace: Path, log_path: Path, deadline: float):
    environment = {**os.environ, "AGENT_TEST_ADDR": "127.0.0.1:0"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(binary)], cwd=workspace, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            startup = time.monotonic() + min(10, remaining(deadline))
            address = ""
            while time.monotonic() < startup:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                match = re.search(r"listening on (http://127\.0\.0\.1:\d+)", text)
                if match:
                    address = match.group(1)
                    break
                if process.poll() is not None:
                    raise RuntimeError("server exited at startup: " + text[-1500:])
                time.sleep(0.025)
            if not address:
                raise TimeoutError("server startup did not advertise loopback port")
            yield address
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def request(address: str, step: dict, deadline: float) -> dict:
    body = json.dumps(step["body"]).encode() if "body" in step else None
    headers = {"Content-Type": "application/json", **step.get("headers", {})}
    req = urllib.request.Request(address + step["path"], data=body,
                                 method=step["method"], headers=headers)
    # Environment HTTP proxies must not receive local benchmark traffic.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(req, timeout=remaining(deadline, 3))
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        text = response.read(1 << 20).decode("utf-8", errors="replace")
        status = response.code
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    return {"status": status, "json": payload, "text": text[:2000]}


def check_steps(address: str, steps: list[dict], deadline: float) -> tuple[list, list]:
    records, failures = [], []
    for index, step in enumerate(steps):
        remaining(deadline)
        try:
            actual = request(address, step, deadline)
            expected = step["expect"]
            errors = []
            if actual["status"] != expected["status"]:
                errors.append(f"status: expected {expected['status']}, got {actual['status']}")
            for field, value in expected.get("fields", {}).items():
                current = actual["json"]
                try:
                    for part in field.split("."):
                        current = current[int(part)] if isinstance(current, list) else current[part]
                    if current != value:
                        errors.append(f"{field}: expected {value!r}, got {current!r}")
                except (KeyError, IndexError, TypeError, ValueError):
                    errors.append(f"missing JSON field {field}")
            records.append({"step": index + 1, "actual": actual, "errors": errors})
            failures.extend(f"step {index + 1}: {error}" for error in errors)
        except Exception as exc:
            failures.append(f"step {index + 1}: {type(exc).__name__}: {exc}")
            records.append({"step": index + 1, "error": str(exc)})
    return records, failures


def race_probe(address: str, deadline: float) -> None:
    def exercise(index: int) -> None:
        try:
            if index % 2:
                step = {"method": "POST", "path": "/todos", "body": {
                    "title": f"parallel-{index}", "user_id": 1, "project_id": 1, "priority": 2}}
            else:
                step = {"method": "GET", "path": "/todos?project_id=1&page=1&limit=2"}
            request(address, step, deadline)
        except Exception:
            pass  # Capture race diagnostics from server stderr, not transport symptoms.
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(exercise, range(120)))


def verify(workspace: Path, baseline_path: Path, *, timeout: int = 180,
           expect_bugs: bool = False, difficulties: list[str] | None = None,
           skip_race: bool = False) -> dict:
    deadline = time.monotonic() + timeout
    baseline = load_baseline(baseline_path)
    bugs = [bug for bug in baseline["bugs"] if not difficulties or bug["difficulty"] in difficulties]
    if not bugs:
        raise ValueError("empty Bug selection")
    results, builds, failures = [], [], []
    go = go_binary()
    temporary_root = Path(__file__).resolve().parent.parent / ".repomind" / "evaluation" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="debug-http-", dir=temporary_root) as temporary:
        root = Path(temporary)
        binary = root / "server.exe"
        build = command([go, "build", "-o", str(binary), "."], workspace, deadline)
        builds.append(build)
        if build["exit_code"]:
            return {"passed": False, "setup_error": "build failed", "builds": builds, "bugs": []}
        with running_server(binary, workspace, root / "smoke.log", deadline) as address:
            smoke_records, smoke_failures = check_steps(address, baseline["smoke_steps"], deadline)
        failures.extend(smoke_failures)
        race_binary = root / "server-race.exe"
        for bug in bugs:
            if bug["verification"]["kind"] == "race" and skip_race:
                results.append({"bug_id": bug["bug_id"], "status": "skipped",
                                "reason": "explicitly excluded; not verified", "contract_passed": None})
                continue
            log_path = root / f"{bug['bug_id']}.log"
            steps, errors, status = [], [], "checked"
            executable = binary
            if bug["verification"]["kind"] == "race":
                race_build = command([go, "build", "-race", "-o", str(race_binary), "."], workspace, deadline)
                builds.append(race_build)
                if race_build["exit_code"]:
                    results.append({"bug_id": bug["bug_id"], "status": "unsupported",
                                    "errors": ["race build unavailable; not counted as verified"]})
                    failures.append("race build unavailable")
                    continue
                executable = race_binary
            with running_server(executable, workspace, log_path, deadline) as address:
                if bug["verification"]["kind"] == "race":
                    race_probe(address, deadline)
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")
                    if "WARNING: DATA RACE" in log_text:
                        errors.append("race detector reported shared-memory race")
                    else:
                        # A dead server without a race diagnostic is an infrastructure error,
                        # not proof of either a repaired race or a reproduced race.
                        try:
                            health = request(address, {"method": "GET", "path": "/health"}, deadline)
                            if health["status"] != 200:
                                status = "error"
                        except Exception:
                            status = "error"
                else:
                    steps, errors = check_steps(address, bug["verification"]["steps"], deadline)
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            reproduced = bool(errors)
            ok = status == "checked" and (reproduced if expect_bugs else not reproduced)
            if not ok:
                failures.append(f"{bug['bug_id']}: {'not reproduced' if expect_bugs else 'contract failed'} ({status})")
            results.append({"bug_id": bug["bug_id"], "difficulty": bug["difficulty"],
                            "status": status, "contract_passed": not errors and status == "checked",
                            "bug_reproduced": reproduced, "steps": steps,
                            "errors": errors, "server_log": log_text[-16000:]})
        existing = None
        if not expect_bugs and not difficulties:
            existing = command([go, "test", "./..."], workspace, deadline)
            if existing["exit_code"]:
                failures.append("existing Go checks failed")
    return {"passed": not failures, "mode": "reproduce" if expect_bugs else "acceptance",
            "coverage_complete": not any(row["status"] in {"skipped", "unsupported"} for row in results),
            "bugs": results, "smoke": smoke_records, "failures": failures,
            "builds": builds, "existing_checks": existing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--expect-bugs", action="store_true")
    parser.add_argument("--skip-race", action="store_true", help="Exclude race probe and report incomplete coverage.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], action="append")
    args = parser.parse_args()
    try:
        result = verify(Path(args.workspace).resolve(), Path(args.baseline).resolve(),
                        timeout=args.timeout, expect_bugs=args.expect_bugs, difficulties=args.difficulty,
                        skip_race=args.skip_race)
    except Exception as exc:
        result = {"passed": False, "setup_error": f"{type(exc).__name__}: {exc}"}
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "report": str(report),
                      "coverage_complete": result.get("coverage_complete", False),
                      "failures": result.get("failures", []), "setup_error": result.get("setup_error", "")}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
