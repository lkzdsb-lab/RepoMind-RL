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
                name="repo_engineering_workflows",
                description=(
                    "Handle repository engineering tasks including code review, code editing, "
                    "structured codebase search, test failure triage, and Go backend debugging."
                ),
                version="0.1.0",
                triggers=[
                    "review",
                    "code review",
                    "代码审查",
                    "审查",
                    "检查代码",
                    "看代码",
                    "有没有问题",
                    "是否正确",
                    "找 bug",
                    "找风险",
                    "find bugs",
                    "risk",
                    "correctness",
                    "implementation quality",
                    "fix",
                    "patch",
                    "modify",
                    "implement",
                    "change code",
                    "edit code",
                    "apply_code_patch",
                    "改代码",
                    "修改",
                    "修复",
                    "实现",
                    "codebase_context",
                    "search_code_context",
                    "symbol",
                    "function",
                    "route",
                    "db_model",
                    "call_graph",
                    "test_mapping",
                    "index",
                    "test",
                    "pytest",
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
                    "assertion",
                    "timeout",
                    "flaky",
                    "import error",
                    "panic",
                    "traceback",
                    "failure",
                ],
                resources=[(skill_root / "repo_engineering_workflows" / "SKILL.md").as_posix()],
                entrypoints={"workflow": "repo_engineering_workflows"},
                metadata={"priority": 1, "category": "engineering"},
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
                resources=[(skill_root / "memory_consolidation" / "SKILL.md").as_posix()],
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
                resources=[(skill_root / "registry_extension" / "SKILL.md").as_posix()],
                entrypoints={"workflow": "registry_extension"},
                metadata={"priority": 5, "category": "testing"},
            ),
        ]
        for spec in default_specs:
            self.register(spec)
