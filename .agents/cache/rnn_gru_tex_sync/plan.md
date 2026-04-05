# RNN/GRU TeX Sync Plan

Task scope: update the English and zh-cn TeX notes so they concretely reflect the new code results, especially the harder long-memory benchmark that now separates plain RNN and GRU more honestly than the easy toy sentiment demo.

## Serial prerequisites

- [x] Inspect the current TeX sections on companion code and inspect the new benchmark code/results.
- [x] Freeze file ownership for this task:
  - English note worker owns `notes/RNN_GRU_基础/main.tex`
  - zh-cn note worker owns `notes/RNN_GRU_基础/main_zh-cn.tex`
  - integration layer owns `.agents/cache/rnn_gru_tex_sync/{plan.md,verify.md,log4human.md}`
- [x] Keep verification commands fixed:
  - `source .venv/bin/activate && python notes/RNN_GRU_基础/code/long_memory_benchmark.py`
  - `bash notes/RNN_GRU_基础/compile.sh`

## Parallel implementation

- [x] Update `notes/RNN_GRU_基础/main.tex` so the companion-code section includes the harder benchmark setup, observed comparison, and a modest claim about what the benchmark does and does not prove.
- [x] Update `notes/RNN_GRU_基础/main_zh-cn.tex` with the aligned Chinese explanation and result summary.

## Serial integration

- [x] Reconcile English and zh-cn claims so they make the same scientific point.
- [x] Rebuild the PDFs after the TeX changes.
- [x] Write `log4human.md` so a human can quickly understand the task, result, and quick verification route.
- [x] Execute every item in `verify.md`. Do not mark completion until the acceptance gate passes.
