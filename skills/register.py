from pathlib import Path

from baseregister import BaseRegistry
from model.skill import SkillSpec

class SkillRegistry(BaseRegistry):
    def __init__(self, include_defaults: bool = True) -> None:
        super().__init__()
        if include_defaults:
            self.register_defaults()

    def register(self, spec: SkillSpec) -> None:
        super().register(spec)

    def register_defaults(self) -> None:
        skill_root = Path(__file__).resolve().parent
        default_specs = [
            SkillSpec(
                name="go_backend_debug",
                description=(
                    "Debug Go backend issues by following route, handler, service, "
                    "repository, model, and test relationships."
                ),
                version="0.1.0",
                triggers=[
                    "go",
                    "golang",
                    "handler",
                    "service",
                    "repository",
                    "controller",
                    "route",
                    "middleware",
                    "gorm",
                    "sql",
                    "go test",
                ],
                resources=[(skill_root / "go_backend_debug.md").as_posix()],
                entrypoints={"workflow": "go_backend_debug"},
                metadata={"priority": 1, "category": "debugging"},
            ),
            SkillSpec(
                name="codebase_context_workflow",
                description=(
                    "Use the structured codebase context index to search routes, symbols, "
                    "functions, DB models, call graph hints, and test mappings."
                ),
                version="0.1.0",
                triggers=[
                    "codebase_context",
                    "search_code_context",
                    "symbol",
                    "function",
                    "route",
                    "db_model",
                    "call_graph",
                    "test_mapping",
                    "index",
                ],
                resources=[(skill_root / "codebase_context_workflow.md").as_posix()],
                entrypoints={"workflow": "codebase_context_workflow"},
                metadata={"priority": 2, "category": "code_context"},
            ),
            SkillSpec(
                name="memory_consolidation",
                description=(
                    "Promote useful memories across episodic, semantic, procedural, "
                    "anti-pattern, and skill memory using reward-gated criteria."
                ),
                version="0.1.0",
                triggers=[
                    "memory",
                    "episodic",
                    "semantic",
                    "procedural",
                    "anti_pattern",
                    "reward",
                    "promotion",
                    "consolidation",
                    "skill memory",
                ],
                resources=[(skill_root / "memory_consolidation.md").as_posix()],
                entrypoints={"workflow": "memory_consolidation"},
                metadata={"priority": 3, "category": "memory"},
            ),
            SkillSpec(
                name="registry_extension",
                description=(
                    "Extend runtime registry resources with tool, node, prompt, and skill "
                    "manifests while preserving snapshot behavior."
                ),
                version="0.1.0",
                triggers=[
                    "registry",
                    "manifest",
                    "tool",
                    "node",
                    "prompt",
                    "skill",
                    "ToolSpec",
                    "ManifestLoader",
                    "Snapshot",
                ],
                resources=[(skill_root / "registry_extension.md").as_posix()],
                entrypoints={"workflow": "registry_extension"},
                metadata={"priority": 4, "category": "registry"},
            ),
            SkillSpec(
                name="test_failure_triage",
                description=(
                    "Triage failing pytest, go test, import, timeout, flaky, and assertion "
                    "failures before patching."
                ),
                version="0.1.0",
                triggers=[
                    "test",
                    "pytest",
                    "go test",
                    "assertion",
                    "timeout",
                    "flaky",
                    "import error",
                    "panic",
                    "traceback",
                    "failure",
                ],
                resources=[(skill_root / "test_failure_triage.md").as_posix()],
                entrypoints={"workflow": "test_failure_triage"},
                metadata={"priority": 5, "category": "testing"},
            ),
        ]
        for spec in default_specs:
            self.register(spec)
