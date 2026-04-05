---
name: plan-worker-orchestrator
description: 先由记忆继承 subagent 产出 `plan.md` 和 `verify.md`，再按文件 ownership 串并行派发 worker，最后由主 agent 亲自复核验收门禁的多阶段执行技能。适用于用户明确要求“先计划、后门禁、再由 workers 执行”的任务。
---

# Plan Worker Orchestrator

用于把一次复杂任务拆成：

1. `plan agent` 先落任务编排文件
2. `worker agents` 按 ownership 执行
3. 主 agent 做最终集成和门禁复核

## 什么时候用

- 用户明确要求：
  - 先派一个 subagent 写 `plan.md`
  - 再写 `verify.md`
  - 之后再串行或并行派 worker
- 用户希望任务过程沉淀到 `.agents/cache/<task_name>/`
- 用户要求只有通过验收门禁后才能算完成
- 任务天然能拆成多个写入范围互不重叠的子任务

## 默认目录

- 任务缓存目录：`.agents/cache/<task-name>/`
- 最终至少包含：
  - `plan.md`
  - `verify.md`
  - `log4human.md`

`<task-name>` 用短横线或下划线命名，要求能表达这轮任务主题。

## 执行顺序

### 1. 串行前置

- 先检查仓库现状，确认任务边界、目标文件、运行环境。
- 再创建 `.agents/cache/<task-name>/`。
- 然后派一个记忆继承的 `plan agent`，它只允许写：
  - `plan.md`
  - `verify.md`

不要在 `plan.md` 和 `verify.md` 出来之前抢先派实现 worker。
此时不要让 `plan agent` 预写 `log4human.md`。

### 2. `plan agent` 的要求

`plan.md` 必须：

- 用 checkbox 列出任务
- 显式区分：
  - 串行前置
  - 可并行实施
  - 串行集成
- 假设后续执行完全由 subagents 完成，而不是默认主 agent 亲自做所有细活
- 提前写清文件 ownership，避免 worker 冲突

`verify.md` 必须：

- 是验收门禁，而不是随手 checklist
- 至少包含：
  - file gate
  - content gate
  - runtime gate
  - build/test gate（如适用）
- 末尾有 `task_complete` checkbox
- 明确写出：只有所有门禁通过，才允许勾选 `task_complete`

`log4human.md` 必须：

- 面向人类，不面向 agent
- 简洁，不写推理过程和废话
- 至少回答五件事：
  - 任务是什么
  - 具体完成了什么
  - 关键结果是什么
  - 成果文件在哪里
  - 如何快速验证
- 写法应接近“秘书向老板汇报”，而不是执行日志回放
- 必须包含一个 `Quick Verify` 区块，明确写：
  - 先看哪个文件
  - 再跑哪条命令
  - 预期会看到什么现象

但这个文件不属于 planner 阶段；它应在真实实现完成后，由 worker 或串行集成阶段产出。

### 2.1 Planner Isolation Contract

`plan agent` 只允许修改缓存目录中的：

- `plan.md`
- `verify.md`

除此之外的任何文件改动，都视为越权，不算成功完成计划阶段。

主 agent 在接受 `plan agent` 结果前，必须做一次越权检查：

- 查看 `git status --short`、文件列表或等价证据
- 确认 `plan agent` 没有改动缓存目录之外的目标文件

如果发现越权：

- 不接受该 planner 的“任务已完成”说法
- 不把其正文改动视为有效实现
- 重新派一个 planner，或由主 agent 亲自重写计划文件

这条规则的目的不是惩罚 subagent，而是防止“计划阶段偷偷实现任务”导致串并行编排失真。

### 2.2 Planner Prompt Requirements

给 `plan agent` 的提示词至少要包含四层约束：

1. 明确写出它的唯一写入范围
2. 明确声明“不要实现任务本体”
3. 明确声明“越权修改 = 失败”
4. 要求它在回报时列出修改文件，供主 agent 复核
5. 明确声明“不要预写 `log4human.md`”

如有命令占位符，例如 `<harder_demo_file>.py`，在主 agent 最终验收前要替换成真实路径。

## worker 派发规则

- 先读 `plan.md`，再按其中的串并行设计派 worker。
- `log4human.md` 默认分配给最后一个 worker 或 serial integration 阶段，而不是 planner。
- 每个 worker 必须有清晰 ownership，写入范围不能重叠。
- 如果任务可拆为多个互不冲突的文件面：
  - README / 文档说明
  - 新代码或 benchmark
  - 英文主文档
  - 中文翻译文档
  可一次并发派出。
- 若 agent 线程数达到上限，先关闭已完成使命的 agent，再补发剩余 worker。

## 主 agent 的职责

主 agent 不能把 subagent 的“已完成”“已验收”原样照单全收。主 agent 必须亲自做：

- 抽样或全文检查关键文件
- 先做 planner 越权检查，再进入 worker 阶段
- 运行门禁中的关键命令
- 复核 `verify.md` 的勾选是否真实
- 清理运行残留物，例如 `__pycache__`
- 做最后的统一口径和风格收敛

特别是：

- 不要仅因为 subagent 说 `task_complete` 已勾选，就直接结束
- 要以仓库真实状态和实跑结果为准

## 默认复核清单

- `plan.md` 是否真的覆盖了串行、并行、集成三层
- `verify.md` 是否包含客观门禁，而不是空泛描述
- `log4human.md` 是否让人类不翻聊天记录也能迅速看懂任务、结果、产物位置和快速验证方法
- `plan agent` 是否严格只改了缓存目录中的计划文件
- worker 是否遵守了 ownership
- 最终命令是否由主 agent 亲自跑过至少一轮
- 若生成了文档或 PDF，主 agent 要亲自确认构建成功

## 资源

- `references/plan-template.md`
- `references/verify-template.md`
- `references/log4human-template.md`
- `references/planner-prompt-template.md`

如果任务要用共享 Python 环境，额外读取：

- `../shared-python-env/SKILL.md`
