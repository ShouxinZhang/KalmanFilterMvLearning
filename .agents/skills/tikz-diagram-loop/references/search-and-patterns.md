# 搜索与模式

## 搜索顺序

1. 先搜索当前仓库。
   - `rg -n "tikzpicture|tikzset|usetikzlibrary|node\\[" <dir>`
   - 如果本地已经存在箭头约定、颜色方案和标签样式，就优先复用。
2. 如果仓库内模式不够，再去查官方 PGF/TikZ 文档。
   - 推荐查询词：`pgf tikz manual fit background layer`、`pgf tikz positioning nodes`、`pgf tikz arrows.meta`。
3. 如果确实需要去更广泛的网页上找灵感，不要盲信摘下来的代码片段。
   - 先编译它。
   - 再栅格化它。
   - 最后检查图像效果。

## 模块化结构

- 共享样式应放在 preamble 或专门的样式文件里。
- 可复用辅助逻辑应放进宏里，而不是复制粘贴 `\tikzset{...}` 代码块。
- 调试界面应是 scratch 图文件，而不是主文档。
- 每张图都应当有四个视觉层次：
  - styles
  - nodes
  - edges
  - captions / surrounding prose

## 隔离规则

- 设计循环：
  - scratch `figure.tex`
  - scratch PDF/PNG
  - visual inspection
  - revise
- 集成循环：
  - 把最终 `tikzpicture` 贴回真实文档
  - 跑一次编译
  - 只检查摆放效果

不要把这两个循环混在一起。

## 默认好做法

- 能用命名节点时，就不要直接写裸坐标。
- 节点内部标签尽量短。
- 如果普通换行不稳定，用 `\shortstack{...}` 处理双行节点文本。
- 只有在节点布局已经可读后，才引入背景 `fit` 块。
- 较长的语义解释放到图外。
