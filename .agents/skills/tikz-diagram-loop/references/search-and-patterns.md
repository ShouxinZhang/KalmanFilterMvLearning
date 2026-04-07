# Search And Patterns

## Search order

1. Search the current repo first.
   - `rg -n "tikzpicture|tikzset|usetikzlibrary|node\\[" <dir>`
   - Reuse local arrow conventions, colors, and label style when they already exist.
2. If repo patterns are insufficient, search official PGF/TikZ documentation.
   - Good queries: `pgf tikz manual fit background layer`, `pgf tikz positioning nodes`, `pgf tikz arrows.meta`.
3. If you must search the broader web for inspiration, do not trust the snippet blindly.
   - Compile it.
   - Rasterize it.
   - Inspect the image.

## Modular structure

- Shared styles belong in preamble or a dedicated style file.
- Reusable helpers belong in macros, not copied `\tikzset{...}` blocks.
- The debug surface belongs in a scratch figure file, not in the main document.
- Each figure should have four visual layers:
  - styles
  - nodes
  - edges
  - captions / surrounding prose

## Isolation rule

- Design loop:
  - scratch `figure.tex`
  - scratch PDF/PNG
  - visual inspection
  - revise
- Integration loop:
  - paste final `tikzpicture` into the real document
  - run one build
  - check placement only

Do not merge these two loops.

## Good defaults

- Use named nodes instead of raw coordinates whenever possible.
- Use short labels inside nodes.
- Use `\shortstack{...}` for two-line node text when ordinary line breaks are unstable.
- Use background `fit` blocks only after the node layout is already readable.
- Keep long semantic explanations outside the figure.
