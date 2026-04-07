# RNN/GRU TeX Diagram Verification Gate

Acceptance rule: all items below must pass before completion. If any item fails, leave `task_complete` unchecked.

## File gate

- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex` exists and contains the English RNN and GRU architecture diagrams.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex` exists and contains the Chinese RNN and GRU architecture diagrams.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble.tex` exists and includes the shared TikZ support required by the diagrams.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble_zh-cn.tex` exists and includes the shared TikZ support required by the diagrams.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main.pdf` exists after rebuild.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf` exists after rebuild.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/` exists.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human_main_zh-cn.md` exists.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human_main_en.md` exists.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human_shared_preamble.md` exists.
- [x] `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human.md` exists and was written by the secretary subagent.

## Content gate

- [x] The English note shows a clear RNN architecture diagram with an input-to-hidden-to-output flow.
- [x] The English note shows a clear GRU architecture diagram with the update/reset gate structure visible.
- [x] The Chinese note shows a clear RNN architecture diagram with the same structural content as the English one.
- [x] The Chinese note shows a clear GRU architecture diagram with the same structural content as the English one.
- [x] The diagrams match the surrounding explanatory style and do not introduce claims that contradict the existing text.
- [x] `log4human/log4human.md` clearly states the task, completed work, key result, and artifact locations without unnecessary detail.

## Runtime gate

- [x] `grep -n 'tikzpicture' /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex` runs successfully and finds the inserted diagram environments.
- [x] `bash /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/compile.sh` runs successfully.
- [x] `test -s /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main.pdf && test -s /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf` runs successfully.
- [x] The rebuild produces visible diagram changes in both PDFs, not just untouched text.

## Build/Test gate

- [x] `bash /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/compile.sh` succeeds without errors.
- [x] The expected artifacts exist at `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main.pdf` and `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf`.

## Completion

- [x] task_complete
