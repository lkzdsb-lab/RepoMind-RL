# Lee-Agent
RepoMind-RL 是一个能在真实代码仓库中自动定位 Bug、生成补丁、运行测试并沉淀经验的 Coding Agent。它通过独创的“奖励门控因果记忆策略”管理长期经验，并用 Agentic RL 学习何时检索、写入、更新、遗忘记忆，从而让 Agent 在多轮任务中越修越聪明。

## 第一版 Agent

当前版本提供一个可运行的 Debug Agent Harness：

- `agent_runtime/executor.py`：负责编排任务生命周期。
- `agent_runtime/policy.py`：第一版启发式 action policy，后续可替换为 LLM/RL policy。
- `agent_runtime/tool_registry.py`：统一封装 `search_code_context`、`search_code`、`read_file`、`run_tests`、`git_diff`。
- `agent_runtime/trajectory.py`：记录可回放 trajectory，并落盘到 `.repomind/traces/`。
- `agent_runtime/memory/`：第一版 JSONL memory card 存储，后续可替换为 SQLite/向量库。

运行示例：

```bash
python3 main.py "订单状态不会从 pending 更新到 paid"
```

启动时会自动读取仓库根目录下的 `config.json`；如果文件不存在，会先生成一份默认模板再加载。常用运行参数、LLM API、mode、memory、code context、RL 和日志配置都可以放在这个文件里；`config.schema.json` 提供可选值校验，完整说明见 `docs/config.md`。命令行参数仍然保留，且只在显式传入时覆盖配置文件。

是否运行验证命令不是配置开关。启用 `modes.task_analyzer = "llm"` 后，LLM 会根据任务意图输出 `verification_required`；只有该值为 `true` 时，后续 policy 才会选择 `run_tests`。

根 `llm` 只作为默认模型配置，不代表所有 LLM 模块都会启用。是否调用 LLM 由 `modes` 单独控制；比如 `modes.memory_reranker = "disabled"` 时，即使根 `llm` 已配置 key 和 model，memory reranker 也不会调用 LLM。

运行产物按目标项目隔离。传入 `--repo /path/to/project-a` 时，trace、memory、log、code index、RL 数据都会写入 `/path/to/project-a/.repomind/`；调试另一个项目会写入另一个项目自己的 `.repomind/`。

LLM key 不写入 `config.json`。默认会从同目录 `.env` 读取：

```bash
cp .env.example .env
# edit .env: LLM_API_KEY=...
```

`config.json` 里只保留 key 的环境变量名：

```json
{
  "env_file": ".env",
  "llm": {
    "provider": "openai_compatible",
    "model": "<model-name>",
    "api_base": "https://<host>/v1",
    "api_key_env": "LLM_API_KEY"
  }
}
```

也可以指定其他配置文件或禁用配置文件：

```bash
python3 main.py "定位订单状态问题" --config config.local.json
python3 main.py "定位订单状态问题" --no-config --repo /path/to/repo
```

如果把任务也写进配置文件：

```json
{
  "task": {
    "title": "订单状态不会从 pending 更新到 paid",
    "description": "请定位并修复支付回调后订单状态偶发不更新的问题"
  }
}
```

就可以直接运行：

```bash
python3 main.py
```

## 运行时 Registry

运行时资源由 `RegistryManager` 管理，包含 `tools`、`nodes`、`prompts`、`skills` 四类 registry。每次 agent run 开始时会创建一份不可变 `RegistrySnapshot`，本轮执行只使用这份 snapshot；后续 manifest reload 或服务端发布不会影响已经开始的任务。

可以通过 `config.json` 的 `manifest_dir` 加载 JSON/TOML manifest，也可以临时使用 `--manifest-dir` 覆盖：

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

默认内置的 workflow skills 会随 `SkillRegistry` 自动注册，不需要额外 manifest：

- `go_backend_debug`
- `codebase_context_workflow`
- `code_review_workflow`
- `memory_consolidation`
- `registry_extension`
- `test_failure_triage`

这些 skill 不替代工具和存储层；它们描述如何组合 tools、memory、registry 和测试反馈完成一类任务。检索命中后会进入 `skill_context` 和 `selected_skills`，再参与 memory/context/LLM prompt。

## 分层 Memory

当前 memory 层实现了四种记忆类型：

- `episodic`：某次任务的经历、候选文件、验证结果和补丁摘要。
- `semantic`：从高 reward 任务中提炼出的稳定事实。
- `procedural`：可复用的调试流程或操作步骤。
- `anti_pattern`：失败路径、错误假设或不应重复的操作。

存储层分三层：

- 短期记忆存在 `AgentState.short_term_memories`，只服务当前任务上下文。
- 中期记忆默认写入 `.repomind/memory_mid.jsonl`；传入 `--memory-redis-url` 后使用 Redis adapter。
- 长期记忆默认写入 `.repomind/memory_long.jsonl`，通过 `LocalVectorMemoryStore` 模拟向量库边界，后续可替换为真实向量数据库。

执行流程：

1. `retrieve_memory` 会从短期、中期、长期和 skill 四层检索相关记忆。
2. 检索结果会拼成 `AgentState.memory_context`，可直接传给 LLM prompt。
3. `write_memory` 写入 episodic 或 anti-pattern memory。
4. reward 达到阈值后自动 promotion 到 semantic/procedural long-term memory。
5. reward 更高的长期记忆会 consolidation 到 `.repomind/skills/{skill}.md`。

默认阈值在 `DebugAgentConfig` 中：

```python
semantic_promotion_threshold = 0.7
procedural_promotion_threshold = 1.2
skill_consolidation_threshold = 1.5
```

## Context Compression 与 LLM 接入

Agent 每轮 action 前会调用 `ContextCompressionManager`。当估算 token 超过 `context_max_tokens * context_compression_threshold` 时，会把旧上下文压缩成结构化 `ContextDigest`，写入：

- `AgentState.context_digest`
- `AgentState.compressed_context`
- `AgentState.context_items`

默认使用 rule-based compressor，不需要外部 API。要启用 LLM 压缩，在 `config.json` 中提供 OpenAI-compatible 配置；`ContextCompressionManager` 会复用根 `llm` 配置：

```json
{
  "llm": {
    "provider": "openai_compatible",
    "model": "<model-name>",
    "api_base": "https://<host>/v1",
    "api_key_env": "LLM_API_KEY"
  },
  "modes": {
    "context_compressor": "llm"
  },
  "context": {
    "enabled": true,
    "max_tokens": 32000,
    "compression_threshold": 0.75
  }
}
```

如果 LLM 未配置、请求失败或返回无法解析的 JSON，会自动降级到 rule-based compression，并把 fallback 原因写入 digest constraints。

## Codebase Context Layer

默认代码搜索已经从原始 grep 切到结构化索引工具 `search_code_context`。首次查询会在目标仓库生成：

```text
.repomind/codebase_context/index.json
```

索引包含：

- `Repo Tree Index`：文件、语言、行数、包名、架构层。
- `Symbol Index`：Go `type`、`struct`、`interface`、`var`、`const`。
- `Function / Method Index`：函数、方法、receiver、起止行、调用列表。
- `API Route Index`：Gin/Echo 风格 `GET/POST/...`、`HandleFunc`、`http.HandleFunc`。
- `DB Model Index`：Go struct、gorm/db/json tags、`TableName()` 映射。
- `Call Graph Lite`：基于函数体调用表达式的轻量调用边。
- `Test File Mapping`：`foo.go -> foo_test.go` 和同 package 测试映射。
- `Embedding Index`：本地 token-vector 索引，后续可替换真实 embedding/vector DB。

可手动构建或刷新：

```python
from tools.code_tools.context import build_codebase_context, search_code_context

build_codebase_context("/path/to/repo", force_rebuild=True)
search_code_context("/path/to/repo", "pay order status", limit=10)
```

Go 项目会额外输出 `flow` 提示，例如：

```text
handler -> service -> repository -> model
route -> middleware -> controller
接口 -> 数据表 -> 测试文件
```

CLI 可指定索引位置：

```bash
python3 main.py "订单支付状态不更新" \
  --repo /path/to/repo \
  --code-context-index-path .repomind/codebase_context/index.json
```

## RL Policy

默认仍使用 `HeuristicDebugPolicy`。开启 `--rl-enabled` 后，Agent 会切换到 epsilon-greedy Q-learning policy，并在每个 action 后记录 transition、计算 reward、在线更新 Q-table。

RL 模块位于：

```text
agent_runtime/rl/
  state_encoder.py   # AgentState -> 离散 state key/features
  action_space.py    # 合法 action 生成和 action args 构造
  reward.py          # reward shaping
  replay_buffer.py   # JSONL replay buffer
  policy.py          # epsilon-greedy Q policy
  trainer.py         # Q-learning update 和 Q-table 持久化
```

运行示例：

```bash
python3 main.py "订单支付状态不更新" \
  --repo /path/to/repo \
  --verify "go test ./..." \
  --rl-enabled \
  --rl-epsilon 0.15
```

默认产物：

```text
.repomind/rl/q_table.json
.repomind/rl/replay.jsonl
```

当前 reward 由工具结果和任务进展构成：

- 搜索到候选文件、route、DB model：正向 reward。
- 读到文件内容：正向 reward。
- 测试通过：大正向 reward。
- 生成 diff 摘要、写入 memory、promotion 到 skill：正向 reward。
- 工具报错、过早 finish、测试失败：负向 reward。

这是最小可运行 RL 闭环：

```text
AgentState -> StateEncoder -> QPolicy -> Action
Action -> ToolRegistry -> next AgentState
prev_state/action/next_state -> RewardFunction
Transition -> ReplayBuffer -> QLearningTrainer -> Q-table
```

后续如果要换成 DQN/Policy Gradient，只需要替换 `QLearningDebugPolicy` 和 `QLearningTrainer`，executor 的 transition 接入点可以保持不变。
