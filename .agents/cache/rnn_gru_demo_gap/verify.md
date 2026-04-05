# RNN/GRU Demo Gap Verification Gate

Acceptance rule: all items below must pass before completion. If any item fails, leave `task_complete` unchecked.

## File gate

- [x] `notes/RNN_GRU_基础/code/plain_rnn_demo.py` exists.
- [x] `notes/RNN_GRU_基础/code/gru_demo.py` exists.
- [x] `notes/RNN_GRU_基础/code/README.md` exists and documents both the easy toy demo and the harder comparison demo.
- [x] At least one harder benchmark file exists under `notes/RNN_GRU_基础/code` and is runnable.
- [x] `notes/RNN_GRU_基础/main.tex` contains an explicit statement that the current toy demo is too easy to cleanly separate RNN and GRU.
- [x] `notes/RNN_GRU_基础/main_zh-cn.tex` contains the aligned Chinese statement.
- [x] `.agents/cache/rnn_gru_demo_gap/log4human.md` exists.

## Content gate

- [x] README or code comments explicitly state that small-sample short-sequence results can make plain RNN and GRU look similar.
- [x] README or code comments explicitly state why the harder benchmark is more discriminative.
- [x] The new note text does not falsely claim that GRU must always outperform plain RNN.
- [x] English and zh-cn note text make the same scientific claim at a high level.
- [x] `log4human.md` clearly states the task, completed work, key result, and artifact locations.

## Runtime gate

- [x] `source .venv/bin/activate && python notes/RNN_GRU_基础/code/plain_rnn_demo.py` runs successfully.
- [x] `source .venv/bin/activate && python notes/RNN_GRU_基础/code/gru_demo.py` runs successfully.
- [x] `source .venv/bin/activate && python notes/RNN_GRU_基础/code/long_memory_benchmark.py` runs successfully.
- [x] The observed harder-demo output includes a measurable comparison between plain RNN and GRU rather than only qualitative prose.

## Build gate

- [x] If `main.tex` or `main_zh-cn.tex` changed, `bash notes/RNN_GRU_基础/compile.sh` succeeds.
- [x] The updated PDFs exist at `notes/RNN_GRU_基础/pdf/main.pdf` and `notes/RNN_GRU_基础/pdf/main_zh-cn.pdf`.

## Completion

- [x] task_complete
