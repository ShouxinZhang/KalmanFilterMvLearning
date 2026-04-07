# Shared Preamble Update

Task: add shared TikZ support in both preambles so the English and Chinese RNN/GRU notes can draw matching architecture diagrams.

Completed work: added `tikz` plus the minimal shared TikZ libraries, then defined reusable styles and helper macros for input, hidden/state, gate, output, arrow, recurrent, skip, and background-box elements in both `preamble.tex` and `preamble_zh-cn.tex`.

Key result: both notes now expose the same diagram vocabulary from the preamble, so the two workers can build visually consistent RNN and GRU figures without duplicating style setup.

Artifact locations:
- `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble.tex`
- `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble_zh-cn.tex`

Quick Verify:
1. Open both preamble files and confirm `\usepackage{tikz}` and the shared `\tikzset{...}` block are present.
2. Run `rg -n "tikz|rnninput|rnnstate|rnngate|rnnoutput|rnnblock" /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/preamble*.tex`.
3. Expected: both files contain the same shared style definitions.
