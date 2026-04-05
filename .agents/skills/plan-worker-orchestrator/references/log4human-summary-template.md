# Log4Human Summary Template

秘书 subagent 汇总所有 worker 日志后，写 `log4human/log4human.md`。

```md
# <Task Title>

## 任务

<一段话说明整体任务是什么>

## 完成情况

- <汇总完成项 1>
- <汇总完成项 2>
- <汇总完成项 3>

## 关键结果

- <一到两条最有决策价值的结论>

## 快速验证

- Open: <first file to inspect>
- Run: `<one command that demonstrates the overall result>`
- Expect: <one sentence about expected outcome>

## 产出清单

- <file or directory 1>: <why it matters>
- <file or directory 2>: <why it matters>
- <file or directory 3>: <why it matters>
```

规则：

- 面向人类老板，不面向 agent
- 不逐条搬运 worker 日志，要做信息聚合和提炼
- 不写推理过程、agent 协调细节、命令输出
- `快速验证` 要足够短：1 个文件 + 1 条命令 + 1 句预期
- 控制在一屏内
- 必须至少回答五件事：
  - 任务是什么
  - 具体完成了什么
  - 关键结果是什么
  - 成果文件在哪里
  - 如何快速验证
