# RNN/GRU TeX Diagram Plan

Task scope: add clear TeX-drawn architecture diagrams for both RNN and GRU inside the existing bilingual note directory `notes/RNN_GRU_基础`, then rebuild the English and Chinese PDFs so the diagrams appear in the rendered documents.

## Serial prerequisites

- [ ] Inspect the current TeX layout and decide where the architecture diagrams should sit relative to the existing RNN and GRU sections.
- [ ] Freeze file ownership before implementation so workers do not touch overlapping files.
- [ ] Use one shared TikZ style for both languages so the diagrams look structurally identical across the two notes.
- [ ] Lock the build command to `bash /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/compile.sh`.

## Parallel implementation

- [ ] Worker A owns `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex` and will add the Chinese RNN and GRU architecture diagrams plus any local explanatory text needed to introduce them; after finishing, write `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human_main_zh-cn.md`.
- [ ] Worker B owns `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex` and will add the English RNN and GRU architecture diagrams plus any local explanatory text needed to introduce them; after finishing, write `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human_main_en.md`.
- [ ] Worker C owns `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble_zh-cn.tex` and `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble.tex` and will add the shared TikZ-related package support needed by both diagrams; after finishing, write `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human_shared_preamble.md`.

## Serial integration

- [ ] Reconcile naming, spacing, arrow conventions, and caption style so the English and Chinese diagrams communicate the same structure at a high level.
- [ ] Spawn a secretary subagent to read all worker logs and write `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/.agents/cache/rnn-gru-tex-diagrams/log4human/log4human.md`.
- [ ] Run `bash /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/compile.sh` and confirm both PDFs are regenerated.
- [ ] Inspect the rendered PDFs for the final placement and readability of both diagrams.
- [ ] Execute every item in `verify.md` before marking the task complete.
