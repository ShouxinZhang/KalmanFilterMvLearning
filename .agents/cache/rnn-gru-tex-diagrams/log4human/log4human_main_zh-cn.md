# 中文笔记 RNN/GRU 架构图

任务：在 `notes/RNN_GRU_基础/main_zh-cn.tex` 里补上 RNN 和 GRU 的 TeX 架构图，并保持正文风格简洁一致。

已完成：在朴素 RNN 公式后插入了一张流程图，在 GRU 门控公式后插入了一张门控结构图；同时补了中文文档所需的 TikZ 依赖和局部样式定义。

关键结果：中文笔记现在能直接在正文里看到 RNN 的输入-状态-输出流，以及 GRU 的更新门、重置门、候选状态和门控混合流程。

成果文件：
- `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/main_zh-cn.tex`
- `/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/pdf/main_zh-cn.pdf`

Quick Verify：
先看 `main_zh-cn.tex` 里 `\begin{tikzpicture}` 的两处插图，然后运行
`bash /home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/notes/RNN_GRU_基础/compile.sh`。
预期现象是 `pdf/main_zh-cn.pdf` 重新生成，且正文中可看到两张新架构图。
