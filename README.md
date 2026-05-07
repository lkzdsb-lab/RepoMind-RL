# Lee-Agent
RepoMind-RL 是一个能在真实代码仓库中自动定位 Bug、生成补丁、运行测试并沉淀经验的 Coding Agent。它通过独创的“奖励门控因果记忆策略”管理长期经验，并用 Agentic RL 学习何时检索、写入、更新、遗忘记忆，从而让 Agent 在多轮任务中越修越聪明。

### 架构设计
```yaml
RepoMind-RL
├── agent_runtime/
│   ├── graph_adapter.py          # 适配 LangGraph SDK
│   ├── state.py                  # 自定义 AgentState
│   ├── trajectory.py             # 自定义轨迹记录格式
│   └── executor.py               # 统一执行入口
│
├── tools/
│   ├── search_code.py
│   ├── read_file.py
│   ├── run_tests.py
│   ├── edit_file.py
│   ├── git_diff.py
│   └── sandbox.py
│
├── memory/
│   ├── memory_card.py            # 你的 Memory Card schema
│   ├── rgcm_policy.py            # Reward-Gated Causal Memory
│   ├── retriever.py
│   ├── consolidator.py
│   └── anti_pattern.py
│
├── rl/
│   ├── state_encoder.py
│   ├── reward_model.py
│   ├── dqn_policy.py
│   └── trainer.py
│
├── eval_harness/
│   ├── task_loader.py
│   ├── swebench_adapter.py
│   ├── backend_bug_bench.py
│   ├── metrics.py
│   └── replay.py
│
└── dashboard/
    ├── trajectory_viewer
    ├── memory_viewer
    └── reward_curve
```

用 LangGraph 做：
状态图、工具调用、trace、checkpoint

自己实现：
RGCM 记忆层
RL controller
debug sandbox
trajectory replay
eval harness
dashboard
消融实验

###  Prompt Layer 和 Skill Layer 的关系

设计：

```
Base System Prompt
    ↓
Node Prompt
    ↓
Skill Prompt
    ↓
Memory Context
    ↓
Tool Schema
    ↓
Output Schema
```

### 处理架构

```
┌────────────────────────────────────────────┐
│ LangGraph Agent Runtime                    │
│ state / node / edge / checkpoint / replay  │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│ Policy Layer                               │
│ rule policy → RL skill-tool policy          │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│ Skill Layer                                │
│ GoBugLocalization / DBConsistency / Auth   │
│ Memory-to-Skill Consolidation              │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│ Prompt Layer                               │
│ system prompt / node prompt / skill prompt │
│ output schema / few-shot examples          │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│ Tool Harness                               │
│ search_code / read_file / go_test / diff   │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│ Go Community Backend Environment           │
│ real repo / API / DB / tests / logs         │
└────────────────────────────────────────────┘
```
