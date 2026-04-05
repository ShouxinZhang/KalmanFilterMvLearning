# Log4Human Worker Template

每个 worker 在完成自己的子任务后，写一份 `log4human/log4human_<worker_task>.md`。

```md
# <Worker Task Title>

## 我负责什么

<一句话说明这个 worker 的 ownership>

## 完成了什么

- <具体完成项 1>
- <具体完成项 2>

## 产出文件

- <file path 1>: <一句话说明>
- <file path 2>: <一句话说明>

## 快速验证

- Run: `<one command>`
- Expect: <expected outcome>

## 遇到的问题（可选）

- <如有值得记录的问题或决策，写在这里；没有则删掉此节>
```

规则：

- 只写自己负责的部分，不管其他 worker
- 不写推理过程和冗长日志
- 控制在半屏以内
- `产出文件` 必须用真实路径，不用占位符
- 如果子任务没有可运行的命令，`快速验证` 写"目视检查 <file path>"
