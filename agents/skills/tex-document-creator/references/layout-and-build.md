# TeX Layout And Build Reference

## Default layout

For a standalone document folder, prefer:

```text
doc/
├── name.tex
├── name.pdf
└── build/
    ├── compile.sh
    ├── name.aux
    ├── name.log
    ├── name.out
    ├── name.toc
    ├── name.fls
    ├── name.fdb_latexmk
    ├── name.synctex.gz
    └── name.xdv
```

Why this layout works:

- the root stays human-readable,
- Git diffs stay cleaner,
- users can find the final PDF immediately,
- and repeated compilation does not litter the directory.

## Class selection

- `ctexart`: default for Chinese notes, tutorials, and short writeups.
- `ctexrep`: Chinese long-form reports.
- `ctexbeamer`: Chinese slides.
- `article`: English notes when Chinese support is unnecessary.
- `report`: English long-form reports.
- `beamer`: English slides.

## Compile commands

Preferred:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir="build" name.tex
```

Fallback:

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory="build" name.tex
```

## Common cleanup targets

If intermediate files escape into the root, move or remove:

- `*.aux`
- `*.log`
- `*.out`
- `*.toc`
- `*.fls`
- `*.fdb_latexmk`
- `*.synctex.gz`
- `*.xdv`

## Practical reminders

- `hyperref` plus math-heavy headings can create bookmark errors; avoid raw inline math in section titles.
- `xelatex` is the safest default for Chinese documents.
- If the repo already uses a working TeX toolchain, preserve it instead of forcing a new one.
