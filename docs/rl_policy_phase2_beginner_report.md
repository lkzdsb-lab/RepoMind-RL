# RepoMind-RL RL Policy Phase 2 小白版详细报告

这份文档解释本轮 `feat/rl-policy-v2` 分支到底做了什么、为什么要做、每个文件有什么作用、指标怎么理解，以及这些内容未来如何写进简历或项目展示。

重点先说清楚：

- RepoMind-RL 不是训练一个新的大模型。
- 它是在做一个 Debug/Coding Agent Harness，也就是让 Agent 围绕真实代码仓库执行任务的运行框架。
- LLM 负责理解代码、生成复杂内容，比如 patch、搜索关键词、shell 命令。
- RL 负责决策流程，比如下一步应该搜索代码、读文件、跑测试、看 diff，还是结束任务。
- 本轮改动把 RL 决策层从“有雏形”推进到了“能记录、能训练、能诊断、能测试、能比较”的工程闭环。

---

## 1. 这个项目到底是什么

你可以把 RepoMind-RL 想成一个“自动调试助手的工作台”。

普通聊天机器人只是回答问题，比如：

> 这个 bug 可能在哪里？

但 RepoMind-RL 想做的是更接近真实 Coding Agent 的流程：

1. 读用户给的 issue。
2. 判断这个任务需要看哪些代码。
3. 搜索仓库。
4. 读取候选文件。
5. 必要时生成修改方案。
6. 执行测试命令。
7. 检查 git diff。
8. 判断是否可以结束。
9. 把过程记录下来，未来可以复盘和学习。

所以它不是只做聊天，而是在做一个 Agent Runtime，也就是 Agent 的运行系统。

---

## 2. LLM 和 RL 分别负责什么

这点非常重要，也是你负责的 RL Policy 部分最核心的技术路线。

### LLM 负责什么

LLM 指 Large Language Model，也就是大语言模型，比如 GPT、Claude、DeepSeek。

在这个项目里，LLM 主要负责“语义理解”和“复杂内容生成”。

例如：

- 看懂 issue 在说什么。
- 判断哪些代码文件可能相关。
- 生成搜索关键词。
- 生成 patch 内容。
- 解释测试失败原因。
- 写最终报告。

LLM 擅长处理自然语言、代码语义和复杂文本。

### RL 负责什么

RL 指 Reinforcement Learning，强化学习。

在这个项目里，RL 不负责训练大模型，也不负责直接写代码。RL 负责 Harness 决策，也就是：

> 当前 Agent 处在这个状态，下一个动作应该做什么？

例如：

- 现在还没有候选文件，应该先 `search_code_context`。
- 已经找到候选文件了，应该 `read_file`。
- 已经改完代码了，应该 `run_tests`。
- 测试通过之后，应该 `git_diff` 或 `finish`。
- 如果验证过期了，不应该直接 finish。

所以可以总结成一句话：

> LLM 负责“想明白复杂语义”，RL 负责“选择下一步行动”。

---

## 3. 用一个例子理解整个流程

假设用户给了一个 issue：

> 修复登录接口在密码为空时没有报错的问题。

Agent 的理想流程大概是：

1. `search_code_context`
   - 搜索 login、password、auth 相关代码。
   - 找到 `auth.py`、`test_auth.py`。

2. `read_file`
   - 读取 `auth.py`。
   - 发现密码为空时没有校验。

3. `read_file`
   - 读取 `test_auth.py`。
   - 了解现有测试风格。

4. `apply_code_patch`
   - 由 LLM 生成具体修改内容。
   - RL 不直接写 patch，RL 只决定“现在应该尝试修改代码”。

5. `run_tests`
   - 运行测试，确认是否修复成功。

6. `git_diff`
   - 查看改了哪些文件。

7. `finish`
   - 如果测试通过，且 diff 合理，就结束任务。

每一步之后，系统都会记录：

- 当前状态是什么。
- 执行了什么 action。
- 得到了什么结果。
- reward 是多少。
- 下一个状态是什么。

这就是 RL 的最小闭环。

---

## 4. 这次整体技术路线

本轮路线可以拆成五层。

### 第 1 层：Action Space 清理

Action Space 的意思是“Agent 允许选择的动作集合”。

以前 action 边界比较混乱，有些 action 已经过时，有些 action 其实不适合纯 RL 自己选。

现在拆成三类：

| 分类 | 含义 | 例子 |
|---|---|---|
| Pure RL actions | RL 可以直接选择的动作 | `search_code_context`, `read_file`, `run_tests`, `git_diff`, `finish` |
| LLM-assisted actions | 需要 LLM 生成复杂参数的动作 | `apply_code_patch`, `run_shell_command`, `search_text` |
| System actions | 系统内部动作，不给 policy 直接选择 | `write_memory` |

这样做的好处是：

- RL 不假装自己能写复杂 patch。
- LLM 不负责整个流程控制。
- 两者职责边界更清楚。

### 第 2 层：Reward 升级

Reward 的意思是“奖励分数”。

RL 学习时需要知道一个动作做得好不好，所以每一步都要给 reward。

例如：

| 行为 | Reward 倾向 |
|---|---|
| 搜索到了新候选文件 | 加分 |
| 重复读取同一个文件 | 扣分 |
| 测试通过 | 大幅加分 |
| 测试失败 | 扣分 |
| 有代码 diff | 加分 |
| 验证过期还 finish | 大幅扣分 |

这样 RL 才能逐渐学到：

> 哪些状态下做哪些动作更有价值。

### 第 3 层：Transition 和 Replay 记录

Transition 指“一步经验”。

它记录的是：

```text
state -> action -> reward -> next_state
```

中文理解：

```text
当前情况 -> 做了什么 -> 得了多少分 -> 变成什么情况
```

Replay Buffer 指“经验回放池”。

它把很多条 transition 保存成 JSONL 文件，后续可以：

- 重新训练 Q-table。
- 做离线分析。
- 看 agent 做了哪些动作。
- 统计行为指标。

### 第 4 层：Q-table 训练和版本管理

Q-table 是表格化强化学习里的一个表。

它大概长这样：

```json
{
  "state_A": {
    "read_file": 0.5,
    "run_tests": 0.1
  }
}
```

意思是：

> 在 state_A 这个状态下，选择 read_file 的价值是 0.5，选择 run_tests 的价值是 0.1。

本轮把 Q-table 从裸字典升级成 envelope 格式：

```json
{
  "metadata": {
    "encoder_version": "state-encoder-v1",
    "action_space_version": "action-space-v1",
    "reward_version": "reward-v1"
  },
  "q_values": {
    "state_A": {
      "read_file": 0.5
    }
  }
}
```

好处是：

- 能知道这个 Q-table 是用哪一版 state encoder 训练出来的。
- 能知道它对应哪一版 action space。
- 能知道 reward 规则是哪一版。
- 如果版本不匹配，可以拒绝加载，避免旧数据污染新策略。

### 第 5 层：Evaluator 和 Benchmark

Evaluator 是“离线评估器”。

它读取 replay 和 Q-table，然后输出：

- action 分布。
- 平均 reward。
- Q-table 状态数量。
- finish 是否发生在测试之后。
- replay 版本覆盖率。
- 是否重复读文件。
- 搜索命中率。
- 验证通过率。

Benchmark 是“对比评测器”。

它比较两组 replay：

```text
heuristic policy replay vs RL policy replay
```

然后输出：

- 任务成功率谁更高。
- 成功任务平均步数谁更少。
- 谁重复读文件更少。
- 谁更少在验证过期时 finish。

---

## 5. 谁添加了哪些东西

这里按时间线整理。

### 第一轮：Claude/DeepSeek-v4-pro 主要完成

这一轮主要完成 RL v2 的基础工程。

| 模块 | 文件 | 做了什么 |
|---|---|---|
| Action Space | `agent_runtime/rl/action_space.py` | 清理 action space，区分 Pure RL / LLM-assisted / System |
| Reward | `agent_runtime/rl/reward.py` | 升级 reward shaping，让 reward 看状态变化 |
| State Encoder | `agent_runtime/rl/state_encoder.py` | 增加 `ENCODER_VERSION` |
| Transition | `model/agent/transition.py` | 增加版本字段和 `next_legal_actions` |
| Trainer | `agent_runtime/rl/trainer.py` | Q-table 改成 envelope 格式 |
| Executor | `agent_runtime/executor.py` | 主流程记录版本化 transition |
| Evaluator | `agent_runtime/rl/evaluator.py` | 新增离线 evaluator |
| Tests | `tests/rl/*` | 新增 RL 单元测试 |

### 第二轮：Claude/DeepSeek-v4-pro 继续完成

这一轮主要补工程完整性。

| 模块 | 文件 | 做了什么 |
|---|---|---|
| LLM lazy import | `agent_runtime/llm/llm.py` | 没装 `openai` 时 executor 也能 import |
| Integration test | `tests/integration/test_executor_rl_smoke.py` | 验证 `DebugAgent` 可以在 RL 模式初始化 |
| Q-table tools | `agent_runtime/rl/qtable_tools.py` | 新增 inspect 和 wrap-legacy 工具 |
| Docs | `docs/rl_policy.md` | 新增 RL Policy 文档 |

### 第三轮：Codex 后续补强

这一轮是我在 code review 之后继续补的。

| 模块 | 文件 | 做了什么 |
|---|---|---|
| Tool output summary | `model/agent/transition.py`, `agent_runtime/executor.py` | transition 里保存小型工具输出摘要 |
| Q-table guard | `agent_runtime/rl/trainer.py` | Q-table 版本不匹配时默认不加载 |
| Evaluator behavior metrics | `agent_runtime/rl/evaluator.py` | 增加行为质量指标 |
| Benchmark | `agent_runtime/rl/benchmark.py` | 新增 heuristic vs RL replay 对比器 |
| Benchmark fixtures | `benchmarks/rl_policy_smoke/*` | 新增 smoke 示例任务和 replay |
| Tests | `tests/rl/test_benchmark.py`, `tests/rl/test_benchmark_fixtures.py` | 新增 benchmark 测试 |
| Docs | `docs/rl_policy.md` | 补充指标和 benchmark 使用说明 |

---

## 6. 每个关键文件到底有什么用

### `agent_runtime/rl/action_space.py`

作用：

> 决定当前状态下有哪些合法 action 可以选。

例子：

如果还没有候选文件：

```text
合法 action 可能是 search_code_context
```

如果已经有候选文件但还没读：

```text
合法 action 可能是 read_file
```

如果已经测试通过：

```text
合法 action 可能是 git_diff 或 finish
```

它的意义：

- 防止 RL 乱选动作。
- 把不适合当前状态的动作过滤掉。
- 让 Q-learning 只在合法动作里选择。

### `agent_runtime/rl/reward.py`

作用：

> 给每一步动作打分。

例子：

```text
搜索到了候选文件：加分
重复读同一个文件：扣分
测试通过：加很多分
验证过期还结束：扣很多分
```

它的意义：

- reward 是 RL 学习的核心信号。
- reward 设计得好，RL 才有可能学到合理策略。

### `agent_runtime/rl/state_encoder.py`

作用：

> 把复杂的 AgentState 压缩成 Q-table 可以使用的 state key。

AgentState 里面可能有很多信息，比如：

- 当前是否有候选文件。
- 是否已经读过文件。
- 测试是否通过。
- 是否有 patch。
- 当前循环次数。

StateEncoder 会把这些信息变成一个离散 key。

例如：

```text
has_candidates=True|has_tests=False|status=running|unread_candidates=2
```

它的意义：

- Q-table 不能直接理解复杂 Python dict。
- 需要把状态变成字符串 key。

### `agent_runtime/rl/policy.py`

作用：

> 根据 Q-table 和探索率选择下一步 action。

这里使用的是 epsilon-greedy。

epsilon-greedy 的意思是：

- 大部分时候选择 Q 值最高的动作。
- 少部分时候随机探索其他动作。

举例：

```text
epsilon = 0.1
```

意思是：

- 90% 情况选当前认为最好的动作。
- 10% 情况随机探索。

### `agent_runtime/rl/trainer.py`

作用：

> 用 transition 更新 Q-table，并负责保存/加载 Q-table。

这次重要改动：

- 支持 envelope 格式。
- 支持 metadata。
- 支持版本不匹配时拒绝加载。

它的意义：

- Q-table 是 RL 的学习结果。
- 版本保护可以避免旧数据污染新策略。

### `agent_runtime/rl/replay_buffer.py`

作用：

> 把每一步 transition 存成 JSONL。

JSONL 的意思是：

> 每一行都是一个 JSON。

好处：

- 可以一行一行追加。
- 方便后续离线分析。
- 不需要一次读入所有数据。

### `model/agent/transition.py`

作用：

> 定义“一步经验”的数据结构。

现在 transition 包含：

| 字段 | 含义 |
|---|---|
| `state_key` | 当前状态 |
| `action` | 做了什么动作 |
| `reward` | 得分 |
| `next_state_key` | 下一个状态 |
| `action_args` | action 参数 |
| `tool_output_summary` | 工具输出摘要 |
| `encoder_version` | state encoder 版本 |
| `action_space_version` | action space 版本 |
| `reward_version` | reward 版本 |
| `next_legal_actions` | 下一个状态可选动作 |

这次新增的 `tool_output_summary` 很重要。

它不会保存大段 stdout 或文件内容，只保存小型摘要，例如：

```json
{
  "exit_code": 0,
  "command": "pytest tests/rl -q"
}
```

这样 evaluator 才能可靠统计验证是否通过。

### `agent_runtime/executor.py`

作用：

> Agent 主流程执行器。

它负责：

- 初始化 DebugAgent。
- 选择 policy。
- 执行动作。
- 更新状态。
- 记录 RL transition。
- 保存 replay 和 Q-table。

这次改动让 executor 真正把 RL 数据写进主流程。

### `agent_runtime/llm/llm.py`

作用：

> LLM 客户端封装。

这次改动是 lazy import `openai`。

以前：

```text
只要 import executor，就会 import openai
```

如果没安装 openai，就直接报错。

现在：

```text
只有真的启用 OpenAI-compatible LLM 时，才 import openai
```

好处：

- RL 测试不需要安装 openai。
- 无 LLM 环境也能跑 executor import。

### `agent_runtime/rl/evaluator.py`

作用：

> 离线分析 replay 和 Q-table。

它能输出：

- 多少条 transition。
- 多少个 episode。
- 平均 reward。
- action 分布。
- Q-table 有多少 state。
- 版本是否匹配。
- 行为质量指标。

### `agent_runtime/rl/qtable_tools.py`

作用：

> Q-table 工具。

支持两个命令：

```bash
python -m agent_runtime.rl.qtable_tools inspect --q-table path
```

查看 Q-table 状态。

```bash
python -m agent_runtime.rl.qtable_tools wrap-legacy --input old.json --output wrapped.json --trust-legacy
```

把旧 Q-table 包装成新 envelope 格式。

### `agent_runtime/rl/benchmark.py`

作用：

> 比较两组 replay 的效果。

典型用法：

```text
heuristic replay vs RL replay
```

输出：

- `task_success_rate`
- `avg_steps_to_success`
- `verification_pass_rate`
- `duplicate_read_ratio`
- `stale_finish_count`

### `benchmarks/rl_policy_smoke/`

作用：

> 提供一个 smoke benchmark 示例。

里面有：

| 文件 | 作用 |
|---|---|
| `tasks.json` | 固定任务列表 |
| `heuristic_replay.jsonl` | 模拟 heuristic policy 的 replay |
| `rl_replay.jsonl` | 模拟 RL policy 的 replay |

注意：

这只是 smoke fixture，用于展示 benchmark 怎么跑，不代表真实项目修复效果。

---

## 7. 专业术语解释

### Agent

Agent 就是“会自己按步骤做任务的程序”。

在这个项目里，Agent 会：

- 搜索代码。
- 读文件。
- 调工具。
- 跑测试。
- 判断是否结束。

### Harness

Harness 可以理解为“测试台”或“运行框架”。

它负责把 Agent 的所有工具、状态、日志、测试、记忆、评估连接起来。

### Policy

Policy 是“决策策略”。

它回答一个问题：

> 下一步该做什么？

### State

State 是“当前状态”。

例如：

- 有没有候选文件。
- 有没有读过文件。
- 测试是否通过。
- 是否已经有 patch。

### Action

Action 是“动作”。

例如：

- `search_code_context`
- `read_file`
- `run_tests`
- `git_diff`
- `finish`

### Reward

Reward 是“奖励分数”。

动作做得好就加分，做得差就扣分。

### Transition

Transition 是“一步经验”。

格式是：

```text
state -> action -> reward -> next_state
```

### Replay Buffer

Replay Buffer 是“经验池”。

它保存很多 transition，方便以后训练或分析。

### Q-table

Q-table 是“状态-动作价值表”。

它记录：

> 在某个状态下，某个动作大概值多少分。

### Evaluator

Evaluator 是“离线评估器”。

它不执行 Agent，只读取已经产生的 replay 和 Q-table，然后输出统计结果。

### Benchmark

Benchmark 是“固定评测集”。

它用一组固定任务比较两个策略谁表现更好。

### Smoke test

Smoke test 是“冒烟测试”。

意思是：

> 不证明系统非常强，只证明最基本流程能跑通。

### Unit test

Unit test 是“单元测试”。

测试一个小模块，比如 reward 或 trainer。

### Integration test

Integration test 是“集成测试”。

测试多个模块能不能接起来，比如 DebugAgent + RL 初始化。

---

## 8. 指标解释

### `task_success_rate`

中文：任务成功率。

意思是：

> 固定任务集中，有多少比例最终成功完成。

例子：

3 个任务里成功 2 个：

```text
task_success_rate = 2 / 3 = 0.6667
```

这个指标越高越好。

### `avg_steps_to_success`

中文：成功任务平均用了多少步。

意思是：

> 成功完成的任务，平均执行了多少个 action。

例如：

- 任务 A 用了 5 步成功。
- 任务 B 用了 3 步成功。

```text
avg_steps_to_success = (5 + 3) / 2 = 4
```

这个指标越低越好。

因为步数越少，说明 Agent 更高效。

### `verification_pass_rate`

中文：验证通过率。

意思是：

> 跑测试或验证命令时，有多少比例通过了。

例如：

跑了 4 次测试，3 次通过：

```text
verification_pass_rate = 3 / 4 = 0.75
```

这个指标越高越好。

### `duplicate_read_ratio`

中文：重复读取比例。

意思是：

> Agent 有没有反复读同一个文件。

例如：

读文件 4 次，其中 1 次是重复读：

```text
duplicate_read_ratio = 1 / 4 = 0.25
```

这个指标越低越好。

### `stale_finish_count`

中文：验证过期时结束任务的次数。

什么叫 verification stale？

例如：

1. Agent 改了代码。
2. 但是还没重新跑测试。
3. 这时测试结果已经过期。
4. 如果 Agent 直接 finish，就是不安全的。

所以 stale finish count 越低越好，最好是 0。

### `search_hit_rate`

中文：搜索命中率。

意思是：

> `search_code_context` 有多少次真的找到了候选代码。

例如：

搜索 5 次，4 次找到候选文件：

```text
search_hit_rate = 4 / 5 = 0.8
```

这个指标越高越好。

### `finish_after_tests_ratio`

中文：测试后结束比例。

意思是：

> Agent finish 之前是否已经有成功测试。

这个指标越接近 1 越好。

---

## 9. 当前实际测试结果

目前本地验证结果：

```text
python -m pytest tests/rl tests/integration -q
89 passed in 0.49s
```

说明：

- RL 单元测试通过。
- evaluator 测试通过。
- qtable tools 测试通过。
- benchmark 测试通过。
- executor + RL 最小集成测试通过。

导入验证：

```text
imports ok
```

说明：

- `DebugAgent` 可以 import。
- `benchmark` 可以 import。
- 没有因为 openai 缺失导致基础 import 崩掉。

benchmark smoke fixture 输出：

```text
task_success_rate        0.6667 -> 1.0000   delta +0.3333
avg_steps_to_success     5.0000 -> 4.0000   delta -1.0000
verification_pass_rate   1.0000 -> 1.0000   delta +0.0000
duplicate_read_ratio     0.5000 -> 0.0000   delta -0.5000
stale_finish_count       1      -> 0        delta -1
```

解释：

- 示例 RL replay 比 heuristic replay 多成功了 1 个任务。
- 示例 RL replay 平均成功步数更少。
- 示例 RL replay 没有重复读文件。
- 示例 RL replay 没有 stale finish。

再次强调：

> 这是 smoke fixture，不是真实 benchmark 结果。

它证明的是：

> 项目现在具备比较 heuristic 和 RL replay 的评测接口。

---

## 10. 现在可以怎么展示这个项目

### 可以真实写进简历的内容

可以写：

- 设计并实现 Coding Agent Harness 中的 RL Policy 决策层。
- 构建 state encoder、action space、reward shaping、replay buffer、Q-table trainer。
- 引入 transition 和 Q-table 版本管理，避免旧策略数据污染新策略。
- 实现离线 evaluator，用于分析 action 分布、reward、Q-table 和行为质量指标。
- 实现 heuristic vs RL replay benchmark 对比工具，支持任务成功率、平均成功步数、验证通过率等指标。
- 增加 RL 单元测试和 DebugAgent 集成测试。

### 暂时不要写的内容

不要写：

- 已经能自动修复真实 GitHub bug。
- 已经在 SWE-bench 上取得效果。
- 已经训练了 DQN 或深度强化学习模型。
- RL 已经完全控制 memory retrieval。
- RL 已经比 heuristic 在真实任务上效果更好。

更准确的说法是：

> 项目已经完成 RL Harness 的最小闭环和离线评估接口，下一步需要收集真实 benchmark replay 来验证 RL policy 相比 heuristic policy 的效果提升。

---

## 11. 后续还可以继续做什么

### 必须做，如果想证明真实效果

1. 准备固定 benchmark 任务集。
2. 分别用 heuristic policy 和 RL policy 跑同一批任务。
3. 保存两组 replay。
4. 用 `agent_runtime.rl.benchmark` 比较。
5. 输出真实指标表。

### 加分项

1. 增加自动运行 benchmark 的脚本。
2. 把 benchmark 输出保存成 Markdown 报告。
3. 增加更多任务类型，比如 bug fix、test failure、small feature。
4. 增加 CI 检查，保证 evaluator 和 benchmark 不坏。

### 更远期方向

1. 让 memory retrieval 也进入 RL action space。
2. 增加更细的 state features。
3. 从 Q-table 过渡到函数近似方法，比如 DQN。
4. 接入真实 SWE-bench 风格任务。

---

## 12. 一句话总结

这轮工作不是“训练大模型”，也不是“宣称已经能自动修复真实 bug”。

它完成的是：

> 给 RepoMind-RL 增加一个可测试、可记录、可诊断、可比较的 RL Policy 决策闭环。

这个闭环包括：

- action space
- reward
- state encoder
- transition
- replay buffer
- Q-table trainer
- evaluator
- qtable tools
- benchmark compare
- tests
- docs

这就是你在项目里负责的 RL Policy / Agent Harness 决策层的核心价值。
