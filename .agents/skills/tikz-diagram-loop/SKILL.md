---
name: tikz-diagram-loop
description: 在 TeX 项目中创建或修复 TikZ 图，当任务需要复用样式、在隔离的 scratch 文件中渲染、基于单张图片做可视化检查，并在集成回主文档前反复迭代修正时使用。
---

# TikZ 图迭代流程

当你需要创建或修复一个 TikZ 图，而且“渲染出来的图像本身”才是真正的验收标准时，使用这个 skill。

## 不可违背的规则

在图形设计和调试阶段，不要编译主 PDF。

- 不要直接在 `main.tex`、`main_zh-cn.tex`、幻灯片源文件或论文主文件里反复迭代。
- 不要把集成后的 PDF 当作主要审阅界面。
- 先在隔离的 scratch 工作区里把图本身调正确。
- 只有当隔离图在视觉上已经干净、清晰后，才把它贴回真实文档，并做一次集成编译。

## 工作流

### 1. 先搜索现有模式

- 先在仓库里搜索：`rg -n "tikzpicture|tikzset|usetikzlibrary|node\\[" <target-dir>`。
- 如果本地已经有质量不错的样式，优先复用。
- 如果没有合适的本地模式，阅读 [references/search-and-patterns.md](references/search-and-patterns.md)。
- 如果需要查外部资料，优先查官方 PGF/TikZ 手册。

### 2. 将样式与布局拆开

- 共用的颜色、箭头、节点样式和辅助宏，放到 preamble 或样式文件里。
- 单个图文件本身应主要只包含：
  - 命名好的节点，
  - 连线，
  - 简短标签。
- 不要把 caption 和长段解释塞进 scratch 图文件。

### 3. 创建 scratch 工作区

- 使用 [scripts/init_tikz_scratch.py](scripts/init_tikz_scratch.py) 创建隔离的图形工作区。
- scratch 目录是唯一允许进行图形调试的地方。
- 如果图依赖项目里的本地样式，就通过 `--preamble-file` 让 scratch 工作区指向真实 preamble。

典型结果：

```text
scratch/
└── gru-figure/
    ├── figure.tex
    ├── build/
    ├── artifacts/
    └── NOTES.md
```

  ### 4. 只渲染单张图

  - 对 scratch 里的 `figure.tex` 使用 [scripts/render_tikz.py](scripts/render_tikz.py)。
  - 真正要关注的输出是“单图 PNG”，不是整个项目 PDF。
  - 用图像查看工具检查这个 PNG。

  ### 5. 基于视觉结果迭代

  - 如果标签重叠，就缩短标签，或把标签移到箭头外侧。
  - 如果箭头交叉到难以阅读，就改图的拓扑结构，而不是无休止地微调标签位置。
  - 如果图仍然过于拥挤，就拆成多张图。
  - 遇到具体视觉问题时，阅读 [references/visual-debugging.md](references/visual-debugging.md) 获取按症状定位的修复建议。

  重复这个过程，直到隔离生成的 PNG 在不放大的情况下也清晰可读。

  ### 6. 只做一次集成

  - 把最终的 `tikzpicture` 拷回真实的 TeX 源文件。
  - 只有在隔离图已经稳定后，才把可复用样式迁移到共享 preamble。
  - 做一次集成编译，确认版面位置和 caption 间距。
  - 如果集成后的页面效果不对，只做少量上下文调整；不要回到主 PDF 里重启整个设计迭代流程。

  ## 快速命令

  创建 scratch 工作区：

```bash
python3 .agents/skills/tikz-diagram-loop/scripts/init_tikz_scratch.py \
  --dir /tmp/gru-figure \
  --title "GRU Figure" \
  --preamble-file /abs/path/preamble.tex
```

渲染 scratch 图：

```bash
python3 .agents/skills/tikz-diagram-loop/scripts/render_tikz.py \
  --input /tmp/gru-figure/figure.tex \
  --output-dir /tmp/gru-figure/artifacts
```

可选：对最终集成后的某一页 PDF 做检查：

```bash
python3 .agents/skills/tikz-diagram-loop/scripts/render_pdf_page.py \
  --pdf /abs/path/main.pdf \
  --page 3 \
  --output /tmp/page-3.png
```

## 资源

- [scripts/init_tikz_scratch.py](scripts/init_tikz_scratch.py)：为单张图创建隔离的 scratch 工作区。
- [scripts/render_tikz.py](scripts/render_tikz.py)：把单张 scratch 图编译为 PDF/PNG。
- [scripts/render_pdf_page.py](scripts/render_pdf_page.py)：仅用于最终摆放检查，将某一页已集成 PDF 栅格化为图片。
- [references/search-and-patterns.md](references/search-and-patterns.md)：仓库内搜索顺序与模块化规则。
- [references/visual-debugging.md](references/visual-debugging.md)：根据症状定位糟糕布局修复方法的启发式说明。
