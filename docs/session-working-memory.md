# 会话工作记忆

## 候选更新与进展判定

Policy 的 `draft_findings` 和 Observer 的 `finding_candidates` 是更新请求，
其元素使用 `FindingUpdateResponse`：`operation` 为 create/enrich/revise，
另外携带 `evidence_refs`（当前 task 的 tool:N）和 `revision_reason`。
create 使用空 ID；enrich 保留原 claim；revise 明确修订结论并解释原因。
已有候选通过 ID 和源码锚点校验，新 ID 由运行时分配后不再重算。
只提供相同文件或函数不能证明两个不同根因属于同一个候选。

`agent_runtime/llm/finding_updates.py` 是 Policy 和 Observer 共用的更新入口。
Executor 不再重复应用 Policy 请求。更新入口验证引用、拒绝陈旧源码引用、
合并集合，并记录类型化的 `finding_update_events`；非法更新以错误反馈返回，
同一批中的合法更新仍保留。Observer 错误交给下一次 Policy 决策处理。
引用合法只证明来源存在，结论正确性仍由完成复核判断。

接受后的 `draft_findings`、SQLite 工作项仍使用原字段；操作字段不进入
FindingContent。旧数据库/trace 不要求补出过去的更新请求或证据字段。
新模型调用必须遵循新请求协议。候选省略时通过 read_session_memory 取详情。

checkpoint 比较规范化内容：重复和集合换序不产生版本；元数据变化可产生
版本，但不撤销确认。结论、位置/测试范围或有效证据变化会清除旧复核状态。
已有源码指纹由复核刷新，普通补充不能把陈旧证据包装成当前证据。

`agent_runtime/context/progress.py` 统一记录有效增量。分析覆盖按文件版本、
固定 dimension 和范围并集累计；总结改写、拆分已有范围、反复切换 partial/
complete 不重复计分。候选进展按工具证据内容判重，不按 claim 或 confidence。
分析维度为 correctness/boundary/concurrency/security/resource/api_contract/test_gap/other。

完成复核分别比较严格输入指纹与进展指纹：版本或 claim 改变使旧缓存失效，
但不会因此清零连续无进展 finish 计数。当前结论改变时清除当前 judgement，
避免把旧 verdict 应用到新 claim。有效的工作项终态复核按事项及结果只计一次进展。

日志包括候选更新来源、操作、变化类型、拒绝原因、工作项新旧版本、复核
失效原因和新增进展数量。历史版本清理和压缩事件身份优化不在本次改动内。

## 字段契约与规则入口

`model/session_work.py` 定义 `RequirementItem`、`FindingItem`、`ActionItem`，
通过 kind 区分类型。来源、审查、要求、候选和计划字段保持原有平铺 JSON 格式。
`WorkKind`、`WorkStatus`、`EditScope`、`ReviewOutcome`、`EvidenceValidity` 和
`ProjectionAudience` 集中列出固定值；工作项校验还限制类型与状态的合法组合。
其中 `RequirementUpdate`、`WorkItemReview` 保持原有 LLM 返回字段和枚举字符串。

| 工作项 | 合法 status |
| --- | --- |
| requirement | pending、blocked、completed、revoked、superseded |
| finding | pending、blocked、confirmed、rejected |
| action | pending、blocked、completed、rejected |

`agent_runtime/memory/work_policy.py` 集中管理本轮复核、展示、授权和报告的规则。
`items_requiring_review()` 表示“本轮仍需审查”，不是简单筛选 pending 状态。
已完成要求继续提供其授权；unknown 证据保持现有处理方式。
审查 outcome satisfied 映射到存储状态 completed，其余合法 outcome 使用同名状态。
最终未解决报告的筛选与审查筛选分别定义，以保留原有展示语义。

`WorkItemView` 单独定义 historical、evidence_validity、source_user_message。
外部输出仍为平铺字典，存储时明确去掉这些运行时字段。
序列化使用 exclude_unset，避免补齐可选默认字段制造新的工作项版本。
`SessionProjection`、`WorkItemPreview`、`WorkItemPage`、`WorkItemBaseline` 约束展示和分页结构。

数据库读写、恢复 state、trace 输出和 evaluation worker 都校验工作项契约。
非法字段、非法状态及类型组合明确报错，不能默认转换成 pending 或静默丢弃。
旧合法记录无需重写；表结构、ID 算法、字段位置、schema version 和已有归档均保持不变。

## 改动边界

沿用现有 AgentSession → DebugAgent → Policy/Observer → CompletionJudge 流程。
不引入全局 evidence ledger，不改变 task_id、归档目录、consolidation replay key，
也不改变长期记忆 Catalog、embedding 或召回算法。

- session_id：对话会话；reset 创建新会话。
- topic_id：会话内的任务话题；新一轮分析可继续、新建或切换已有话题。
- task_id：仍由运行时按原逻辑生成；awaiting_user_input 后 resume 保持同一个 task_id 和 topic_id。
- item_id / version：工作项的稳定标识及修订版本，不替代 task_id。

## 原逻辑与新逻辑

| 环节 | 原逻辑 | 新逻辑 |
| --- | --- | --- |
| 跨轮信息 | 最近轮次、摘要、playbook | 这些信息仍保留，另有独立的工作项及版本历史 |
| 用户要求 | 容易依赖摘要或当前“继续”输入 | 保存用户原文；原文整体与提取出的条件要求均进入审查 |
| 候选发现 | 当前 state，部分列表存在 20 条截断 | 不按条数丢弃已知候选；独立保存来源、状态和源码指纹 |
| 各节点上下文 | 各自截取字符串 | 共用 JSON 投影，标明省略项数，可按页读取完整记录 |
| 历史确认 | 容易混入当前结论 | 区分历史来源、源码有效性和本轮审查状态 |
| 完成判断 | 当前计划及 finding 审查 | 增加工作项 ID、版本、证据引用及覆盖检查 |

## 保存与读取

SessionMemoryService 在原 SQLite 数据库中增量创建四张表：

- `session_messages`：不可变用户原文，包括同一运行中的后续输入。
- `session_runs`：task_id、话题、轮初快照、checkpoint 与提交标记。
- `session_work_items`：requirement、finding、action 的最新版本。
- `session_work_versions`：每次有效变更的历史版本及写入 task_id。

用户输入在调用模型前保存。任务分析后绑定话题并提取要求，引用的 quote 必须确实
出现在当前运行的用户消息中；替换或撤销必须指向同一话题已有的要求。
分析前中断的未绑定输入会在后续分析中提示，继续时保留其原始来源。
Policy/Observer 产生候选、完成审查以及终止时均 checkpoint；重复 checkpoint 不制造版本。
失败运行中的未完成计划步骤和显式阻塞步骤也会保留，不因历史窗口淘汰而消失。
最终工作记录与 turn/playbook 在同一事务提交，重复提交同一已登记运行不会插入重复 turn。

`session_memory_max_turns` 仅约束叙事历史，不删除工作项。
`session_memory_max_chars` 约束默认提示投影，不是数据库保存上限。
任务分析、规划、搜索、技能选择、Policy、Observer、完成审查及报告共用投影实现；
部分节点使用更小的展示预算。投影是预览，不是完整证据。
压缩器仅保存工作记忆引用提示，不再把整份工作记忆重复压入历史摘要。

`read_session_memory` 是普通注册工具，运行时注入会话服务，模型不能自行指定其他会话。
支持 item_ids 或 offset/limit；返回原始要求、完整 finding、来源消息、版本、next_offset。
可传 revision 检查翻页期间记录是否变化。工具摘要保留这条显式详情读取路径。
完成审查优先展示最近读取的工作页，允许逐批保留有效审查结果。

## 可信边界与完成审查

程序检查工作项身份、版本、状态转换和证据引用。已知项未审查、仍 pending/blocked，
或内存持久化出错时，不能仅凭模型返回 complete 就结束。
`read_session_memory`、计划及用户问答等不能作为本轮验证证据。
判定 finding 的语义是否正确仍由模型结合当前仓库/工具证据完成；引用存在不等于结论必然真实。

源码 hash 用于标记历史证据 current/stale/unknown，不因普通 checkpoint 自动刷新。
历史 confirmed finding 在新运行中仍需复核，不能仅因旧状态 confirmed 就列为本轮发现。
失败报告保留未解决项；不把待复核的候选偷偷算作确认发现。

用户要求的 completed 表示该次义务已审查，不等于撤销同一话题中的明确约束/授权。
例如仅授权补测试时，patch 工具只能修改解析后的测试路径，不能借此改业务文件。
测试路径识别是保守约定；特殊目录结构应由用户澄清权限，不推断扩大范围。
这不是操作系统沙箱，不为允许执行的项目命令提供文件系统隔离。

## 对长期记忆及评估的影响

归档仍在最终 turn 提交之前执行。归档使用运行开始时冻结的 snapshot，
而非已经被本轮 checkpoint 改写的会话状态。SessionMemorySnapshot 增加 work_items，
其 schema_version 为 2；这与原来的 memory_candidates.json schema 2 是两个独立契约。
最终 state 同时带本轮工作记录，便于审计；历史工作项不自动成为 consolidation 的新证据。
现有 evidence collector、candidate builder 和长期记忆 mutation 流程保持原语义。

Evaluation worker 附带话题、工作项基线、结束版本及持久化错误。
评分增加 `session_work_retention`：本轮继承的活跃工作项，在结束时仍保留相同 ID 且版本未回退的比例。
没有继承项时不输出该指标。它独立于原有 finding_recall、评分和按 task_id 去重的成本统计。
它只验证结构保留，不能衡量要求是否提取正确、bug 是否属实；真实场景仍需要人工标注。

## 验证与已知限制

不新增单测文件，不调用付费模型。离线检查覆盖临时 SQLite 多轮运行、25 条候选、
失败恢复、窗口淘汰、话题隔离、工具真实注册调用、提示模板、版本校验、源码失效，
以及临时归档与 evidence collector 的接口衔接。

已有数据库会自动增量建表，但不会凭摘要重新编造旧候选，亦不自动扫描旧 trace。
升级前已经丢失的细节需要原始 trace 或用户补充；新增记录也不能保证模型发现所有 bug。
新结构的真实模型提取质量、执行成本及完整 debug 效果仍需实际场景验证。
