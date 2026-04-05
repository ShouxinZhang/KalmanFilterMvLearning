# RNN/GRU Demo Gap

## Task

Clarify that the existing toy RNN/GRU code only shows that both models can solve a short easy small-sample task, then add a harder benchmark that more honestly exposes the engineering difference between plain RNN and GRU.

## Completed

- Updated the code README and toy demo wording so the easy sentiment demo is explicitly described as a mechanism demo, not a decisive benchmark.
- Added a harder benchmark that tests long-memory retention and length generalization.
- Updated both the English and zh-cn notes to explain why the toy demo makes RNN and GRU look similar and why a harder benchmark is needed.
- Re-ran the code and rebuilt the PDFs.

## Key Result

- On the easy toy sentiment demo, plain RNN and GRU both perform very well, which matches the original observation.
- On the harder long-memory benchmark, plain RNN dropped to about `0.502` mean accuracy across test lengths, while GRU stayed at about `1.000`, so the harder benchmark cleanly separates them.

## Quick Verify

- Open: [long_memory_benchmark.py](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/code/long_memory_benchmark.py)
- Run: `source .venv/bin/activate && python notes/RNN_GRU_基础/code/long_memory_benchmark.py`
- Expect: the output prints an accuracy table where `gru` stays strong across test lengths while `plain_rnn` degrades, with mean accuracy near `1.000` vs `0.502`.

## Artifacts

- [README.md](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/code/README.md): human-readable explanation of the easy demo vs harder benchmark.
- [long_memory_benchmark.py](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/code/long_memory_benchmark.py): new runnable benchmark that exposes the gap.
- [main.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex): English note updated with the new interpretation.
- [main_zh-cn.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex): Chinese note updated with the same claim.
- [main.pdf](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main.pdf): rebuilt English PDF.
- [main_zh-cn.pdf](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf): rebuilt Chinese PDF.
