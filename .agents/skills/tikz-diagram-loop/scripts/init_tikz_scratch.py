#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib


DEFAULT_PREAMBLE = r"""
\usepackage{amsmath,amssymb,bm,mathtools}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
""".strip()


FIGURE_TEMPLATE = r"""
\documentclass[tikz,border=10pt]{{standalone}}
% BEGIN_SHARED_PREAMBLE
{shared_preamble}
% END_SHARED_PREAMBLE

\begin{{document}}
\begin{{tikzpicture}}[>=Latex]
  % Replace this with the real figure.
  \node[draw, rounded corners, minimum width=24mm, minimum height=10mm] (box) at (0,0) {{{title}}};
\end{{tikzpicture}}
\end{{document}}
""".strip()


NOTES_TEMPLATE = """
# Scratch Notes

Purpose: isolated debug workspace for one TikZ figure.

Rules:
- Do not debug this figure in the main TeX document.
- Render only `figure.tex` until the PNG is visually correct.
- Keep labels short and move long explanation outside the figure.
- After the isolated render is good, copy the final `tikzpicture` back into the real document.
""".strip()


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an isolated scratch workspace for one TikZ figure.")
    parser.add_argument("--dir", required=True, help="Scratch directory to create.")
    parser.add_argument("--title", default="Figure", help="Placeholder title for the starter node.")
    parser.add_argument("--preamble-file", help="Optional TeX preamble file to embed into the scratch template.")
    args = parser.parse_args()

    root = pathlib.Path(args.dir).expanduser().resolve()
    (root / "build").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)

    shared_preamble = DEFAULT_PREAMBLE
    if args.preamble_file:
        shared_preamble = read_text(pathlib.Path(args.preamble_file).expanduser().resolve())

    figure_text = FIGURE_TEMPLATE.format(shared_preamble=shared_preamble, title=args.title)
    (root / "figure.tex").write_text(figure_text + "\n", encoding="utf-8")
    (root / "NOTES.md").write_text(NOTES_TEMPLATE + "\n", encoding="utf-8")

    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
