# Lee-Agent
RepoMind-RL 是一个能在真实代码仓库中自动定位 Bug、生成补丁、运行测试并沉淀经验的 Coding Agent。它通过独创的“奖励门控因果记忆策略”管理长期经验，并用 Agentic RL 学习何时检索、写入、更新、遗忘记忆，从而让 Agent 在多轮任务中越修越聪明。

## 第一版 Agent

当前版本提供一个可运行的 Debug Agent Harness：

- `agent_runtime/executor.py`：负责编排任务生命周期。
- `agent_runtime/policy.py`：第一版启发式 action policy，后续可替换为 LLM/RL policy。
- `agent_runtime/tool_registry.py`：统一封装 `list_files`、`search_code`、`read_file`、`run_tests`、`git_diff`。
- `agent_runtime/trajectory.py`：记录可回放 trajectory，并落盘到 `.repomind/traces/`。
- `agent_runtime/memory/`：第一版 JSONL memory card 存储，后续可替换为 SQLite/向量库。

运行示例：

```bash
python3 main.py "订单状态不会从 pending 更新到 paid" \
  --description "请定位并修复支付回调后订单状态偶发不更新的问题" \
  --repo /path/to/target/repo \
  --verify "pytest"
```

## 运行时 Registry

运行时资源由 `RegistryManager` 管理，包含 `tools`、`nodes`、`prompts`、`skills` 四类 registry。每次 agent run 开始时会创建一份不可变 `RegistrySnapshot`，本轮执行只使用这份 snapshot；后续 manifest reload 或服务端发布不会影响已经开始的任务。

可以通过 `--manifest-dir` 加载 JSON/TOML manifest：

```bash
python3 main.py "定位订单状态问题" \
  --repo /path/to/repo \
  --manifest-dir .agent/registry
```

Tool manifest 示例：

```toml
kind = "tool"
name = "my_tool"
description = "Run a custom repository tool."
runner = "my_package.my_tools:run_my_tool"
reducer = "my_package.my_tools:reduce_my_tool_output"
permissions = ["repo:read"]

[input_schema]
query = "string"
```

Node manifest 示例：

```toml
kind = "node"
name = "triage"
description = "Add a custom triage node."
handler = "my_package.nodes:triage_node"

[[metadata.edges]]
from = "make_plan"
to = "triage"

[[metadata.edges]]
from = "triage"
to = "select_action"
```

Prompt manifest 示例：

```toml
kind = "prompt"
name = "debug_agent"
version = "1.0.0"
template_file = "../../prompts/system/debug_agent.md"
variables = ["title", "description"]
```

Skill manifest 示例：

```toml
kind = "skill"
name = "go_bug_localization"
version = "0.1.0"
description = "Localize Go bugs using code search and tests."
triggers = ["go", "bug", "test"]
resources = ["../../skills/go_bug_localization.md"]
```
