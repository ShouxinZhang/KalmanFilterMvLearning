# RNN/GRU Architecture Detail

## Task

Expand the RNN/GRU TeX note so it explains the neural-network architecture itself in more detail, especially how embeddings, recurrent state updates, readout layers, and gradient-based learning work together.

## Completed

- Expanded the English note with a more detailed architecture-learning section.
- Expanded the zh-cn note with the aligned Chinese explanation.
- Recompiled both PDFs after the update.

## Key Result

- The note now explains not only that RNN/GRU are trained by BPTT, but also what each architectural block learns and why GRU's gated parameterization is usually easier to optimize.

## Quick Verify

- Open: [main.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex)
- Run: `bash notes/RNN_GRU_基础/compile.sh`
- Expect: the note contains a longer architecture-learning section around the training part, and both PDFs rebuild successfully.

## Artifacts

- [main.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main.tex): updated English note.
- [main_zh-cn.tex](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex): updated Chinese note.
- [main.pdf](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main.pdf): rebuilt English PDF.
- [main_zh-cn.pdf](/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf): rebuilt Chinese PDF.
