# RNN/GRU Demo Gap Plan

Task scope: update the RNN/GRU note and code so they explicitly state that the current toy demo is too easy to separate plain RNN and GRU, then add a harder demo that better exposes the difference under longer dependency or length-generalization pressure.

## Serial prerequisites

- [x] Inspect the current files under `notes/RNN_GRU_基础/code` and `notes/RNN_GRU_基础/{main.tex,main_zh-cn.tex}` to confirm the present claim gap.
- [x] Freeze the target outputs and file ownership before implementation:
  - code/demo worker owns new or updated files under `notes/RNN_GRU_基础/code`
  - note-text worker owns `notes/RNN_GRU_基础/main.tex`
  - zh-cn note worker owns `notes/RNN_GRU_基础/main_zh-cn.tex`
- [x] Keep the shared Python environment contract fixed:
  - attach or reuse `.venv` via `.agents/skills/shared-python-env`
  - use the existing shared env `py312-torch-cu130` if `py314` is unavailable

## Parallel implementation

- [x] Worker A: update `notes/RNN_GRU_基础/code/README.md` and the existing toy-demo comments so they explicitly say the current demo mainly shows common mechanism, not GRU's practical advantage.
- [x] Worker B: add a harder benchmark under `notes/RNN_GRU_基础/code` that stresses longer-range dependence or length generalization, with a runnable plain-RNN baseline and GRU baseline plus a short result summary.
- [x] Worker C: revise `notes/RNN_GRU_基础/main.tex` to add a concise English note that the toy demo is intentionally easy and that a harder benchmark is needed to reveal the engineering difference.
- [x] Worker D: revise `notes/RNN_GRU_基础/main_zh-cn.tex` with the aligned Chinese explanation.

## Serial integration

- [x] Reconcile naming, assumptions, and claims across README, code comments, English note, and zh-cn note.
- [x] Run the required code commands in `.venv` and capture the observed behavior for both the easy toy demo and the harder benchmark.
- [x] Rebuild the note PDFs if the TeX files changed.
- [x] Write `log4human.md` so a human can see task, result, and artifact locations at a glance.
- [x] Execute every item in `verify.md`. Do not mark completion until the acceptance gate passes.
