
# Agent 运行时状态机与文件读取重构计划

## 1. 背景

当前运行时把任务语义拆成以下链路：

```text
goal contract
-> criterion
-> obligation
-> execution queue
-> progress ledger
-> evidence evaluation
-> completion transition
```

这条链路试图用确定性代码判断诊断、实现和验证是否完成，并进一步限制 LLM 的合法动作。实际运行已经出现两个结构性问题：

1. 多个 implement criterion 绑定同一个文件时，一次局部 patch 会因为文件路径相同而错误完成其他 criterion。
2. verification 失败后，即使 LLM 和 observer 已确认仍需修改代码，当前 obligation 仍可能锁定在 verification，导致 `apply_code_patch` 不再是合法动作，形成 `run command -> observe -> run command` 循环。

文件读取层也存在类似问题。运行时根据 execution target、query 关键词重合度和固定阈值推导 `full_read_requirements`，再用它阻挡 planning、patch 和 evidence。这种规则不能真正判断上下文是否充分，并可能强制读取大型文件全文，增加 token 消耗和注意力噪声。

## 2. 重构目标

采用“LLM 负责语义决策，运行时负责客观事实、安全和协议”的边界：

```text
current user intent
-> task brief
-> LLM-maintained work plan
-> LLM selects action
-> tool execution
-> runtime facts
-> LLM completion judgement
-> deterministic completion gate
```

具体目标：

- 当前用户消息决定本轮任务意图和修改权限，历史记忆只补充事实与上下文。
- LLM 自己决定下一步工作、文件关联、读取范围、验证命令和语义完成情况。
- legal actions 只表达工具能力、安全限制和必要的客观前置条件，不表达业务执行顺序。
- 运行时不再根据“某个文件被修改”推断一个或多个语义目标已经完成。
- 修改代码后必须验证，失败的验证不能完成任务，但也不能阻止 LLM 返回实现阶段继续修复。
- 文件不必全文读取；patch 只能使用当前轮实际读取且仍然有效的代码作为修改依据。

## 3. 非目标

- 不重写工具实现、代码索引、上下文压缩、MEM 持久化或 RL 数据记录。
- 不让 LLM 绕过路径、命令、文件大小和 patch 大小等安全限制。
- 不依赖观察文本判断工具是否成功，工具结构化输出仍是唯一事实来源。
- 不在本次重构中设计长期记忆晋升或 skill 沉淀。

## 4. 新状态模型

### 4.1 Task Brief

Task Analyzer 生成稳定、精简的本轮任务描述：

```json
{
  "intent": "diagnose | implement | explain | review",
  "objective": "当前用户要求的结果",
  "constraints": [],
  "acceptance": [],
  "historical_context": []
}
```

约束：

- `intent` 只能由当前用户消息决定。
- session memory 可以填充 `historical_context`，但不能将 diagnose 自动升级为 implement。
- 历史 playbook 的 `next_steps` 只能标记为历史建议，不能视为当前授权。

### 4.2 Work Plan

工作计划由 LLM 维护，不由运行时从 criterion 派生：

```json
{
  "steps": [
    {
      "id": "inspect_failure",
      "description": "定位失败原因",
      "status": "pending | in_progress | done | blocked"
    }
  ],
  "current_focus": "当前关注的问题",
  "open_questions": []
}
```

`next_action` 的结构化响应同时携带可选 `plan_update`，避免为更新计划增加一次额外 LLM 调用。

### 4.3 Runtime Facts

运行时只维护可确定的事实：

```json
{
  "last_tool_error": null,
  "edited_files": [],
  "edit_revision": 0,
  "verified_revision": 0,
  "last_verification": {},
  "pending_user_input": false,
  "plan_mode_active": false
}
```

禁止在 `runtime_facts` 中存放“某个业务目标已经完成”这类语义判断。

## 5. 执行队列状态机重构

### 5.1 删除语义执行队列

移除以下在线控制结构：

- `goal_contract`
- `progress_ledger`
- `next_obligation`
- `execution_queue`
- criterion dependency
- criterion evidence policy
- criterion 与文件、命令、capability 的绑定

对应删除以下行为：

- 根据 obligation 强制选择 diagnose、patch 或 verification。
- 根据 queue item 裁剪 LLM 的语义动作。
- 根据修改文件交集推进多个 criterion。
- 根据 repository evidence 或 observation 自动完成 criterion。
- verification 失败后仍将执行阶段锁死在 verify。

### 5.2 Legal Actions 新边界

`legal_specs` 只根据以下条件过滤动作：

- 工具是否在 registry 中注册。
- 参数 schema 是否有效。
- 路径是否位于仓库范围内。
- 命令是否属于允许的验证命令族。
- 当前是否处于 Plan Mode。
- editing 是否启用。
- patch 是否满足 read-before-write 和大小限制。
- 是否正在等待用户输入或步骤审批。

以下条件不能再用于移除动作：

- 当前工作计划处于 diagnose、implement 或 verify。
- LLM 是否已经完成某个语义步骤。
- verification 是否失败。
- 某个文件是否属于旧 execution queue 的目标。

verification 失败后，`read_file`、`search_text`、`search_code_context`、`apply_code_patch` 和安全的 `run_shell_command` 应根据各自客观前置条件继续可用。

### 5.3 工具执行后的状态更新

Reducer 只更新事实：

- `read_file` 更新读取快照和已读区间。
- `apply_code_patch` 记录真实 diff、修改文件并增加 `edit_revision`。
- patch 成功后使 `verified_revision < edit_revision`。
- `run_shell_command` 保存命令、退出码、stdout 和 stderr。
- 成功验证最新代码后设置 `verified_revision = edit_revision`。
- 工具错误写入 `last_tool_error`；后续成功恢复可以清除对应错误。

Observer 可以解释工具结果并给出建议，但不能直接修改任务完成状态或限制动作集合。

### 5.4 Completion Gate

完成流程改为：

```text
LLM selects finish
-> LLM completion judge checks task semantics
-> runtime checks deterministic blockers
-> finish or continue
```

运行时只检查：

- Plan Mode 已退出。
- 没有待处理的用户输入或审批。
- 没有尚未恢复的工具协议错误。
- 如果发生代码修改，`verified_revision == edit_revision`。
- 最新验证成功且对应当前代码版本。
- 未超过循环和连续失败限制。

completion judge 负责判断当前结果是否满足 `task_brief.acceptance`。运行时不再通过文件、action name 或 observation 推断语义完成。

### 5.5 轻量循环保护

保留通用保护，不建立新的业务状态机：

- 最大 loop 数。
- 连续相同 action 和相同参数的重复限制。
- 连续同类工具错误限制。
- 临近 loop 上限时执行一次正式 completion judgement。
- 达到上限仍无法完成时返回明确失败原因和最后未解决问题。

重复检测只用于阻止无变化的完全相同调用，不对 phase、文件组合或问题类型建立复杂签名。

## 6. 文件读取与修改重构

### 6.1 删除 Full Read 硬门禁

删除以下规则：

- execution target 必须全文读取。
- 根据关键词 relevance 和固定阈值强制全文读取。
- 自动将 `read_file.max_chars` 提升到固定大值。
- `full_read` 未满足时阻止 planning、patch、evidence 或 completion。
- patch 后必须重新全文读取文件才能验证。

小文件全文读取可以继续作为 prompt 建议，但不能成为 legal action 或完成条件。

### 6.2 由 LLM 决定读取策略

LLM 根据当前问题选择：

- 搜索 symbol 或文本。
- 读取指定行范围。
- 读取目标函数及相邻上下文。
- 追踪调用方或被调用方。
- 对确实需要整体理解的小文件执行全文读取。

Prompt 中提供：

- 文件总行数和大小。
- 已读取行区间。
- 是否截断。
- 当前读取快照的 revision/hash。
- 搜索命中和相关 symbol。

### 6.3 范围级 Read-Before-Write

运行时保留修改安全约束，但从“文件必须全文读过”改为“修改锚点必须实际读过”：

```text
LLM submits old_text/new_text
-> locate old_text in current file
-> verify matched lines are covered by a valid read span
-> verify file revision has not changed since that read
-> apply patch
```

读取快照建议记录：

```json
{
  "file_path": "server.go",
  "file_revision": "hash-or-version",
  "file_size": 12345,
  "total_lines": 220,
  "spans": [
    {
      "start_line": 80,
      "end_line": 130,
      "content": "..."
    }
  ]
}
```

同一 revision 的重叠区间可以合并。文件被 patch 或外部修改后，旧 revision 的 span 不能继续作为修改依据。

### 6.4 缺少上下文时的工具反馈

patch 涉及未读区域时不应改变全局 phase，只返回结构化结果：

```json
{
  "error": "patch_anchor_not_grounded",
  "needs_more_context": true,
  "file_path": "server.go",
  "suggested_range": {
    "start_line": 140,
    "end_line": 190
  }
}
```

LLM 可以选择补读建议区间、重新搜索或采用其他修改方案。legal actions 不强制下一步只能读取文件。

### 6.5 修改后的读取状态

patch 成功后：

- 更新目标文件 revision。
- 保存工具返回的新内容或 diff 对应区间。
- 旧 revision 的读取 span 标记失效。
- 不强制重新读取整个文件。
- 如果后续 patch 的锚点位于新版本中尚未读取或无法由 patch 结果可靠推导的区域，再要求局部补读。
- 验证命令可以直接执行，不要求先重读修改文件。

## 7. 受影响文件

### 7.1 预计删除

- `agent_runtime/lifecycle/goal_contract.py`
- `agent_runtime/lifecycle/obligations.py`
- `agent_runtime/lifecycle/progress_ledger.py`
- `agent_runtime/lifecycle/evidence.py`
- `agent_runtime/lifecycle/execution_queue.py`
- `ext/file_requirements.py` 中的 full-read 推导逻辑

是否删除整个 `ext/file_requirements.py` 取决于范围级读取元数据是否放在新模块中；不得保留旧 relevance/threshold 逻辑作为隐式控制。

### 7.2 预计重写或显著收缩

- `agent_runtime/lifecycle/completion.py`
- `agent_runtime/executor.py`
- `agent_runtime/rl/action_space.py`
- `agent_runtime/actions/factory.py`
- `agent_runtime/tool_registry.py`
- `agent_runtime/llm/task_analysis.py`
- `agent_runtime/llm/llm_policy.py`
- `agent_runtime/llm/completion_judge.py`
- `agent_runtime/context/attention.py`
- `agent_runtime/llm/observation.py`
- `ext/focus_files.py`
- `tools/code_tools/edit.py`
- `model/agent/graph.py`
- `prompts/system/task_analyzer.md`
- `prompts/user/task_analyzer.md`
- `prompts/system/action_policy.md`
- `prompts/user/action_policy.md`
- `prompts/system/completion_judge.md`
- `prompts/user/completion_judge.md`

### 7.3 保留但需要核对接口

- 工具 registry 和具体工具实现。
- verification command guard。
- context compressor 和 codebase retrieval。
- session memory SQLite 表结构。
- RL transition 记录；输入 state encoder 需要移除旧字段。
- change event、真实 diff 和 CLI `Code Changes` 展示。

## 8. 实施顺序

### 阶段一：建立新状态协议

1. 定义 `task_brief`、`work_plan`、`runtime_facts` 和读取 span 类型。
2. Task Analyzer 改为输出当前意图和验收要求。
3. 明确 session memory 只能提供历史事实，不能提供当前修改授权。
4. LLM action response 增加可选 `plan_update`。

完成标准：新字段可以贯穿一轮运行，但暂时不承担 completion 权限。

### 阶段二：替换动作选择

1. 重写 `legal_specs`，只保留能力、安全和协议检查。
2. 移除 obligation、queue 和 full-read 对动作集合的裁剪。
3. verification 失败后确保读取、搜索、patch 和重新验证仍可选择。

完成标准：Go demo 中只修复一个 bug 后测试失败，LLM 仍能继续选择 patch 修复其他问题。

### 阶段三：替换文件读取门禁

1. `read_file_cache` 改为 revision + spans。
2. edit guard 检查 patch anchor 的读取覆盖范围。
3. 删除 full-read blocker 和强制重读逻辑。
4. patch 冲突返回建议补读范围。

完成标准：只读取目标函数区间即可安全 patch；未读取的锚点仍会被拒绝。

### 阶段四：替换完成判断

1. completion judge 使用 `task_brief`、`work_plan` 和真实工具结果做语义判断。
2. runtime gate 使用 revision、验证和错误事实做确定性检查。
3. 删除 criterion evidence 和 goal contract completion。

完成标准：诊断任务无需 patch 即可完成；代码修改任务未经最新验证不能完成。

### 阶段五：删除旧链路

1. 删除 lifecycle 旧模块及 imports。
2. 从 `AgentState`、prompt、attention、observation 和 RL encoder 删除旧字段。
3. 删除旧 queue reducer、reconcile、advance 和 migration fallback。
4. 全仓搜索确认不存在旧字段的控制性引用。

完成标准：运行日志中不再出现 goal contract、criterion、obligation、progress ledger 或 execution queue。

## 9. 验收场景

使用现有 Go Agent Test demo 做端到端验证，不为旧状态机补单元测试。

1. 用户说“看看项目有什么 bug”时只诊断，不因历史 playbook 自动修改代码。
2. 下一轮说“修复刚才三个 bug”时能引用上一轮事实并进入修改。
3. 多个问题位于同一个文件时，一次局部 patch 不会让运行时推断其他问题已解决。
4. 第一次验证失败后，LLM 可以继续读取和 patch，而不是被锁在 verification。
5. LLM 只读取目标函数区间后可以修改该区间。
6. LLM 修改未读取代码时，工具拒绝 patch 并返回建议读取范围。
7. patch 后无需全文重读即可运行验证。
8. 修改后未验证或验证失败时不能 finish。
9. 纯分析任务可以在没有 patch 和测试命令的情况下完成。
10. 连续重复相同命令会触发轻量循环保护并返回清晰原因。

## 10. 主要风险与控制

- 风险：LLM 遗漏用户要求。
  控制：稳定保留 `task_brief.acceptance`，每次 action 和 completion judgement 都可见。
- 风险：LLM 过早 finish。
  控制：completion judge 加确定性的最新修改验证 gate。
- 风险：LLM 基于局部读取误改其他区域。
  控制：范围级 read-before-write、revision 校验和 patch 大小限制。
- 风险：移除 queue 后 LLM 重复操作。
  控制：保留完全相同 action/参数的轻量重复检测，不恢复业务状态机。
- 风险：多轮记忆再次改变当前权限。
  控制：Task Analyzer 将当前 intent 与 historical context 分字段处理，当前消息始终优先。

## 11. 设计原则

后续实现必须遵守以下原则：

```text
语义决策交给 LLM。
客观事实、安全和协议交给运行时。
历史记忆补充上下文，但不授予当前操作权限。
读取多少由 LLM 决定，修改依据是否真实由运行时校验。
验证失败是新信息，不是只能继续验证的状态锁。
```
