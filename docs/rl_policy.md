# RepoMind-RL Policy

## 定位

**RepoMind-RL 是一个面向真实代码仓库的 Debug/Coding Agent Harness，不是用来训练大模型的。**

- **LLM** 负责语义推理和复杂参数生成（如 patch 内容、shell 命令）。
- **RL** 负责 action type 决策、流程控制、reward shaping、replay buffer 管理和 Q-table 更新。

RL 策略在 Q-table 中学习「在什么 state 下该选什么 action」，LLM 负责填充该 action 的具体参数。

## 当前 RL 数据流

```
AgentState  →  StateEncoder.encode(state)  →  encoded state key
                                                    ↓
                          ActionSpace.legal_specs(state)  →  legal actions
                                                    ↓
                          QLearningDebugPolicy.next_action(state)
                             ├─ 探索 (ε-greedy): 随机选 legal action
                             └─ 利用: 选 Q 值最高的 legal action
                                                    ↓
                          executor 执行 action  →  next state + output
                                                    ↓
                          RewardFunction.compute(prev, action, next, output)
                                                    ↓
                          Transition (s, a, r, s', done, next_legal_actions)
                                                    ↓
                          ReplayBuffer.append()  +  QLearningTrainer.update()
                                                    ↓
                          QTableStore.save()  (envelope format)
```

## Action 分类

| 分类 | Actions | 说明 |
|---|---|---|
| **Pure RL** | `search_code_context`, `read_file`, `run_tests`, `git_diff`, `finish` | 始终可用；RL 能自主选择并生成参数 |
| **LLM-assisted** | `search_text`, `run_shell_command`, `EnterPlanMode`, `ExitPlanMode`, `apply_code_patch`, `request_user_input` | 仅在 `llm_action_inputs_enabled=true` 时可用；LLM 负责参数生成 |
| **System** | `write_memory` | 不由 policy 选择，executor finalize 时自动触发 |

### 约束

- `list_files` — 已从默认 action space 移除（ToolRegistry 中注册被注释）。
- `apply_code_patch` — 需要全部四项条件：`editing_enabled=true`, `llm_action_inputs_enabled=true`, `plan_mode_approved=true`, 已有 read file 记录。
- `run_shell_command` — 仅在 `llm_action_inputs_enabled=true` 时合法；纯 RL 模式使用 `run_tests`。

## Reward v1 规则

版本: `reward-v1`

| 条件 | Reward |
|---|---|
| 每步基础 cost | -0.02 |
| Fatal / error | -1.0 |
| `search_code_context` — 新候选文件 | min(0.4, 0.1 × 新文件数) |
| `search_code_context` — 无候选 | -0.15 |
| `read_file` — 首次读取 | +0.25 |
| `read_file` — 重复读取 | -0.12 |
| `read_file` — 失败/空内容 | -0.10 |
| `run_tests` / `run_shell_command(verification)` — 通过 | +1.0 |
| `run_tests` / `run_shell_command(verification)` — 失败 | -0.25 |
| `run_tests` — 清除 `verification_stale`（编辑后首次验证通过） | +0.5 |
| `apply_code_patch` — applied | +0.3 |
| `apply_code_patch` — needs_user_input | +0.05 |
| `apply_code_patch` — guard 拒绝 / needs_more_context | -0.08 |
| `git_diff` — 有真实 diff | +0.15 |
| `finish` — 测试通过后 | +1.0 |
| `finish` — verification_stale 时 | -1.2 |
| `finish` — 无候选/无测试/无 diff（过早 finish） | -0.8 |

**Terminal reward** 仅用于 final report，不写入 replay buffer（避免重复计分）。

## 如何运行

### 运行 RL 单元测试

```bash
pytest tests/rl -q
```

### 运行集成测试

```bash
pytest tests/rl tests/integration -q
```

### 离线评测

```bash
python -m agent_runtime.rl.evaluator \
    --replay .repomind/rl/replay.jsonl \
    --q-table .repomind/rl/q_table.json \
    --format text
```

支持 `--format json` 输出机器可读结果。

### Q-table 工具

```bash
# 查看 Q-table 状态
python -m agent_runtime.rl.qtable_tools inspect --q-table .repomind/rl/q_table.json

# 包装旧版 Q-table（需要手动验证兼容性）
python -m agent_runtime.rl.qtable_tools wrap-legacy \
    --input old_q.json \
    --output .repomind/rl/q_table.json \
    --trust-legacy
```

## 已知限制

1. **没有真实 benchmark** — 当前 RL 策略在单元测试级别验证，缺少大规模代码修复的端到端评测（如 SWE-bench）。
2. **没有 DQN / 深度学习** — 当前纯 Q-table 是表格化学习，状态空间离散化后维度有限；未引入神经网络函数近似。
3. **memory retrieval 尚不是 RL action** — memory 检索由启发式/LLM 驱动，未纳入 RL action space。
4. **端到端自动修复能力仍需评测证明** — 当前没有完整的修复成功率指标；evaluator 只统计 replay 统计数据。
5. **LLM 依赖 lazily imported** — 默认 LLM disabled 时不需要 `openai`；启用 LLM 时需要 `pip install openai`。
6. **Q-table 自动迁移有限** — 旧格式 Q-table 被识别并安全忽略；`qtable_tools wrap-legacy --trust-legacy` 提供显式手动包装路径，不做语义兼容性保证。
