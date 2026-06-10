"""Lightweight verification command inference."""

from __future__ import annotations

import shlex
from pathlib import Path
from utils import _clean_string_list


def infer_lightweight_verification_command(
    repo_path: str,
    *,
    configured: str = "",
    changed_files: list[str] | None = None,
    candidate_files: list[str] | None = None,
) -> str:
    """Infer lightweight verification command."""
    configured = str(configured or "").strip()
    if configured:
        return configured
    repo = Path(repo_path or ".").resolve()
    files = _clean_string_list(changed_files or [], -1, None) or _clean_string_list(candidate_files or [], -1, None)
    if not files:
        return _repo_default_command(repo)

    suffixes = {Path(path).suffix for path in files}
    if suffixes and suffixes.issubset({".py"}):
        return _python_compile_command(files)
    if ".go" in suffixes:
        return _go_test_command(repo, files)
    if suffixes.intersection({".ts", ".tsx", ".js", ".jsx"}):
        return _node_command(repo, files)
    if ".java" in suffixes:
        return _java_command(repo)
    return _repo_default_command(repo)


def _python_compile_command(files: list[str]) -> str:
    targets = [shlex.quote(path) for path in files if path.endswith(".py")][:20]
    if not targets:
        return "python -m compileall ."
    return "python -m py_compile " + " ".join(targets)


def _go_test_command(repo: Path, files: list[str]) -> str:
    if not (repo / "go.mod").exists():
        return "go test ./..."
    packages = []
    for path in files:
        if not path.endswith(".go"):
            continue
        parent = Path(path).parent.as_posix()
        package = "./..." if parent == "." else f"./{parent}"
        if package not in packages:
            packages.append(package)
    if not packages or len(packages) > 3:
        return "go test ./..."
    return "go test " + " ".join(shlex.quote(item) for item in packages)


def _node_command(repo: Path, files: list[str]) -> str:
    package_json = repo / "package.json"
    if not package_json.exists():
        for path in files:
            if path.endswith((".js", ".jsx")):
                return f"node --check {shlex.quote(path)}"
        return "node --version"
    text = package_json.read_text(encoding="utf-8", errors="ignore")
    for script in ("typecheck", "lint", "test"):
        if f'"{script}"' in text:
            return f"npm run {script}"
    return "npm test"


def _java_command(repo: Path) -> str:
    if (repo / "pom.xml").exists():
        return "mvn test"
    if (repo / "gradlew").exists():
        return "./gradlew test"
    if (repo / "build.gradle").exists() or (repo / "build.gradle.kts").exists():
        return "gradle test"
    return "javac"


def _repo_default_command(repo: Path) -> str:
    if (repo / "go.mod").exists():
        return "go test ./..."
    if (repo / "pyproject.toml").exists() or (repo / "setup.py").exists():
        return "python -m compileall ."
    if (repo / "package.json").exists():
        return _node_command(repo, [])
    if (repo / "pom.xml").exists() or (repo / "build.gradle").exists() or (repo / "build.gradle.kts").exists():
        return _java_command(repo)
    return "python -m compileall ."
