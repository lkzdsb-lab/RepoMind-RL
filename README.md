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
