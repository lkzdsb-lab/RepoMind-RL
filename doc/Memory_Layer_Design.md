#MemoryLayer晋升整体逻辑#

`memoryLayer` 的晋升逻辑主要在 `manager.py` 里，核心方法是：

```python
record_task_memory() 
_build_task_card() 
_reward_credit() 
_promote() 
_consolidate_to_skills()
```

它的流程可以理解成：

> 任务结束  
>
> -> 生成一张基础 MemoryCard  
>
> -> 写入 mid-term memory  
>
> -> 根据 reward/promotion_score 判断是否晋升到 long-term  
>
> -> 如果分数更高，再沉淀成 skill memory

## 1. 先生成基础记忆
任务执行到 write_memory action 时，会调用：

```
record_task_memory(state, registry)
```

这里第一步是：

```
base = self._build_task_card(state)
```

_build_task_card() 会根据当前 AgentState 生成一张 MemoryCard。

如果测试通过：

```
memory_type = "episodic" 
status = "verified"
```

如果测试失败：

```
memory_type = "anti_pattern" 
status = "draft"
```

也就是说：

> 测试通过 -> 正向经验 episodic 
>
> 测试失败 -> 失败经验 anti_pattern

## 2. 计算 reward_credit
基础记忆会带一个 reward_credit，来自：

```
_reward_credit(state)
```

当前规则是：

```python
reward = 0.0 
if latest_exit == 0: 
	reward += 1.0 
elif latest_exit is not None:
	reward -= 0.2 
if state.get("patch"):    
	reward += 0.35 
if state.get("error"):   
	reward -= 0.4
```

所以大概是：

> 测试通过        +1.0 
>
> 测试失败        -0.2 
>
> 有代码 diff     +0.35 
>
> 执行中有 error  -0.4

举例：

> 测试通过 + 有 patch 
>
> ​	reward_credit = 1.0 + 0.35 = 1.35
>
> 测试失败 + 有 error 
>
> ​	reward_credit = -0.2 - 0.4 = -0.6

## 3. promotion_score 晋升分
真正决定晋升的是 MemoryCard.promotion_score()。

位置在 cards.py。

当前公式是：

```python
reuse_score = 0.15 * reuse_success - 0.25 * reuse_failure 
age_penalty = min(max(conflict_score, 0.0), 1.0)
promotion_score = reward_credit + reuse_score - age_penalty
```

也就是：

> promotion_score =  reward_credit  + 0.15 * 复用成功次数  - 0.25 * 复用失败次数  - conflict_score

含义：

> 任务本身做得好 -> 分数高 
>
> 这个 memory 后续被成功复用 -> 分数继续涨 
>
> 这个 memory 后续误导 Agent -> 分数下降 
>
> 冲突越高 -> 越不容易晋升

## 4. 写入 mid-term
无论是否晋升，基础记忆都会先写入中期记忆：

```
written = [self.mid_store.append_card(base)]
```

默认路径是：

.repomind/memory_mid.jsonl

这一层保存的是任务级经验，主要是 episodic 或 anti-pattern。

## 5. 晋升到 long-term
然后进入：

```
promoted = self._promote(base, state)
```

晋升阈值在 config.py：

```python
semantic_promotion_threshold = 0.7 
procedural_promotion_threshold = 1.2 
skill_consolidation_threshold = 1.5
```

对应：

> \>= 0.7  -> semantic memory 
>
> \>= 1.2  -> procedural memory 
>
> \>= 1.5  -> skill memory

## 6. semantic 晋升
如果基础 memory 是成功经验，并且：

```
score >= semantic_threshold
```

就会生成一张 semantic long-term memory：

```
type="semantic" 
tier="long_term" 
content=self._semantic_content(state)
```

semantic 记忆更像“稳定事实”。

例如：

> When debugging `订单状态不会更新`, relevant code tends to be in: orders/service.py, payment/callback.py. Verified outcome: 0.

它回答的是：

> 类似问题通常和哪些文件/模块有关？

## 7. procedural 晋升
如果分数更高：

```
score >= procedural_threshold
```

就会再生成一张 procedural long-term memory：

```
type="procedural" tier="long_term" content=self._procedural_content(state)
```

procedural 记忆更像“操作流程”。

例如：

> Procedure for similar tasks: 
>
> search using task-specific keywords, 
>
> read the top candidate files, 
>
> run `pytest`, inspect `git_diff`, and only promote the memory when verification evidence is present.

它回答的是：

> 下次遇到类似问题应该按什么步骤调试？

## 8. anti-pattern 的特殊逻辑
如果基础 memory 是：

```
type == "anti_pattern"
```

当前逻辑会直接把它晋升到 long-term：

```
content="Avoid repeating this failing path. " + base.content
```

也就是说，失败经验不会等正向阈值，而是直接进入长期反例库。

原因是：

> 失败路径也很有价值。 
>
> 它可以提醒 Agent：不要重复错误搜索路径、错误假设、无效操作。

不过anti-pattern 现在不会污染正向搜索 query。它可以出现在 prompt 里提醒 Agent 避坑，但不会被当成“我要搜索的代码关键词”。

## 9. 写入 long-term
所有晋升出来的 memory 会写入：

```python
persisted_promotions = [    
	self.long_store.append_card(card)    
	for card in promoted 
]
```

默认路径：

> .repomind/memory_long.jsonl

虽然当前叫 LocalVectorMemoryStore，但本质还是 JSONL，只是用 token cosine 模拟向量检索边界，方便以后替换成真正的向量数据库。

## 10. skill consolidation*
最后调用：

```python
consolidated = self._consolidate_to_skills(
    persisted_promotions,
    state,
    registry,
)
```

如果某张 long-term memory 的：

```
card.promotion_score() >= skill_threshold
```

也就是默认：

> \>= 1.5

它会被追加写入：

> .repomind/skills/{skill_name}.md

这一步叫：

> consolidation to skill

也就是把高价值经验沉淀成可读的 skill 文档。

### 完整例子
假设一次任务：

> 测试通过
>
> 有 patch 
>
> 没有 error

那么：

> reward_credit = 1.0 + 0.35 = 1.35 
>
> reuse_success = 0 
>
> reuse_failure = 0 
>
> conflict_score = 0 
>
> promotion_score = 1.35

结果：

> 写入 mid-term episodic 
>
> 晋升 semantic，因为 1.35 >= 0.7 
>
> 晋升 procedural，因为 1.35 >= 1.2 
>
> 不会沉淀 skill，因为 1.35 < 1.5

如果这条 memory 后续又成功复用 2 次：

> promotion_score = 1.35 + 0.15 * 2 = 1.65

那么下一次它就可能达到：

> skill_consolidation_threshold = 1.5

最终进入：

> .repomind/skills/*.md