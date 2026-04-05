# RNN/GRU TeX Sync

## Task

Update the English and zh-cn TeX notes so they reflect the newly added code results, especially the harder long-memory benchmark that makes the RNN/GRU difference much more visible than the original easy toy demo.

## Completed

- Added a concrete benchmark summary to the English note.
- Added the aligned Chinese summary to the zh-cn note.
- Rebuilt both PDFs after the TeX update.

## Key Result

- The notes now say not only that a harder benchmark is needed, but also what that benchmark does and what one run showed: plain RNN reached about `0.502` mean accuracy across test lengths, while GRU reached about `1.000`.

## Quick Verify

- Open: [main.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex)
- Run: `source .venv/bin/activate && python notes/RNN_GRU_基础/code/long_memory_benchmark.py`
- Expect: the output prints a length-by-length comparison where `gru` stays strong while `plain_rnn` degrades, matching the new TeX summary.

## Artifacts

- [main.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex): updated English note.
- [main_zh-cn.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex): updated Chinese note.
- [main.pdf](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main.pdf): rebuilt English PDF.
- [main_zh-cn.pdf](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf): rebuilt Chinese PDF.
