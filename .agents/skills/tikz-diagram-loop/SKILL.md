---
name: tikz-diagram-loop
description: Create or repair TikZ diagrams in TeX projects when the task needs reusable styles, isolated scratch-file rendering, single-image visual inspection, and iterative correction before any integration into the main document.
---

# TikZ Diagram Loop

Use this skill when a TikZ figure must be created or repaired and the rendered picture is the real source of truth.

## Non-Negotiable Rule

During figure design and debugging, do not compile the main PDF.

- Do not iterate inside `main.tex`, `main_zh-cn.tex`, slides, or thesis sources.
- Do not use the integrated PDF as the primary review surface.
- First make the figure correct in an isolated scratch workspace.
- Only after the isolated figure is visually clean may you paste it back into the real document and run one integration build.

## Workflow

### 1. Search for patterns

- Search the repo first: `rg -n "tikzpicture|tikzset|usetikzlibrary|node\\[" <target-dir>`.
- Reuse local styles when they are already good.
- If no good local pattern exists, read [references/search-and-patterns.md](references/search-and-patterns.md).
- If browsing is needed, prefer the official PGF/TikZ manual first.

### 2. Split style from layout

- Shared colors, arrows, node styles, and helper macros go in a preamble or style file.
- One figure file should mostly contain:
  - named nodes,
  - edges,
  - short labels.
- Keep captions and long explanation out of the scratch figure.

### 3. Create a scratch workspace

- Use [scripts/init_tikz_scratch.py](scripts/init_tikz_scratch.py) to create an isolated figure workspace.
- The scratch directory is the only place where figure debugging happens.
- If the figure depends on project-local styles, point the scratch workspace at the real preamble with `--preamble-file`.

Typical result:

```text
scratch/
└── gru-figure/
    ├── figure.tex
    ├── build/
    ├── artifacts/
    └── NOTES.md
```

### 4. Render the single figure only

- Use [scripts/render_tikz.py](scripts/render_tikz.py) on the scratch `figure.tex`.
- The output of interest is the single-figure PNG, not the project PDF.
- Review that PNG with image-reading tools.

### 5. Iterate visually

- If labels overlap, shorten them or move them outside the arrow.
- If arrows cross unreadably, change the topology rather than nudging labels forever.
- If the diagram is still crowded, split it into multiple figures.
- Read [references/visual-debugging.md](references/visual-debugging.md) for symptom-driven fixes.

Repeat until the isolated PNG is readable without zooming.

### 6. Integrate once

- Copy the final `tikzpicture` back into the real TeX source.
- Move reusable styles into the shared preamble only after the isolated figure is stable.
- Run one integration build to confirm placement and caption spacing.
- If the integrated page looks wrong, adjust the integration context sparingly; do not restart the whole design loop inside the main PDF.

## Quick Commands

Create a scratch workspace:

```bash
python3 .agents/skills/tikz-diagram-loop/scripts/init_tikz_scratch.py \
  --dir /tmp/gru-figure \
  --title "GRU Figure" \
  --preamble-file /abs/path/preamble.tex
```

Render the scratch figure:

```bash
python3 .agents/skills/tikz-diagram-loop/scripts/render_tikz.py \
  --input /tmp/gru-figure/figure.tex \
  --output-dir /tmp/gru-figure/artifacts
```

Optional final integration check on one PDF page:

```bash
python3 .agents/skills/tikz-diagram-loop/scripts/render_pdf_page.py \
  --pdf /abs/path/main.pdf \
  --page 3 \
  --output /tmp/page-3.png
```

## Resources

- [scripts/init_tikz_scratch.py](scripts/init_tikz_scratch.py): create an isolated scratch workspace for one figure.
- [scripts/render_tikz.py](scripts/render_tikz.py): compile a single scratch figure to PDF/PNG.
- [scripts/render_pdf_page.py](scripts/render_pdf_page.py): rasterize one integrated PDF page for final placement QA only.
- [references/search-and-patterns.md](references/search-and-patterns.md): repo search order and modularization rules.
- [references/visual-debugging.md](references/visual-debugging.md): symptom-to-fix heuristics for bad layouts.
