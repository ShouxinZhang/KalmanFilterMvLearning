# Verify Template

```md
# <Task Title> Verification Gate

Acceptance rule: all items below must pass before completion. If any item fails, leave `task_complete` unchecked.

## File gate

- [ ] <required file 1> exists.
- [ ] <required file 2> exists.
- [ ] <new artifact or benchmark> exists and is runnable.
- [ ] `log4human/` directory exists.
- [ ] Each worker has written `log4human/log4human_<worker_task>.md`.
- [ ] `log4human/log4human.md` summary exists (written by secretary subagent).

## Content gate

- [ ] The new text makes the intended claim explicitly.
- [ ] The text does not overclaim or contradict existing material.
- [ ] Parallel outputs make the same scientific or technical claim at a high level.
- [ ] `log4human/log4human.md` clearly states task, completed work, key result, and artifact locations without unnecessary detail.

## Runtime gate

- [ ] `<command 1>` runs successfully.
- [ ] `<command 2>` runs successfully.
- [ ] The observed output includes a measurable result, not only prose.

## Build/Test gate

- [ ] `<build-or-test-command>` succeeds.
- [ ] Expected artifacts exist at `<paths>`.

## Completion

- [ ] task_complete
```

Notes:

- 把命令里的占位符在最终验收前改成真实路径。
- `task_complete` 必须最后再勾。
- 若任务没有 build 步，可把 `Build/Test gate` 改成更合适的 `Validation gate`。
