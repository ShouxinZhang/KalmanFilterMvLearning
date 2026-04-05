# Plan Template

```md
# <Task Title> Plan

Task scope: <one-paragraph scope statement>

## Serial prerequisites

- [ ] Inspect the current files, constraints, and environment.
- [ ] Freeze target outputs and file ownership before implementation.
- [ ] Lock environment assumptions, build commands, and run commands.

## Parallel implementation

- [ ] Worker A: <write scope + concrete task>
- [ ] Worker B: <write scope + concrete task>
- [ ] Worker C: <write scope + concrete task>

## Serial integration

- [ ] Reconcile naming, assumptions, and claims across outputs.
- [ ] Run the required commands and capture observed behavior.
- [ ] Rebuild artifacts if needed.
- [ ] Each worker writes its own `log4human/log4human_<worker_task>.md` upon completion.
- [ ] Spawn secretary subagent to read all worker logs and write `log4human/log4human.md`.
- [ ] Execute every item in `verify.md`. Do not mark completion until the acceptance gate passes.
```

Notes:

- 每个 worker 条目都要写清“负责哪些文件”和“产出什么”。
- 若有中英文双文档，默认拆成两个 worker。
- 若有 benchmark / code / docs 三块，优先拆成三个 ownership 面。
