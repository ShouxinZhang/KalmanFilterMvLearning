#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import textwrap


DEFAULT_PREAMBLE = r"""
\usepackage{amsmath,amssymb,bm,mathtools}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
""".strip()


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def build_wrapper(body: str, extra_preamble: str) -> str:
    preamble = DEFAULT_PREAMBLE
    if extra_preamble.strip():
        preamble = f"{preamble}\n{extra_preamble.strip()}"
    return textwrap.dedent(
        f"""\
        \\documentclass[tikz,border=4pt]{{standalone}}
        {preamble}
        \\begin{{document}}
        {body}
        \\end{{document}}
        """
    )


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def run(cmd: list[str], cwd: pathlib.Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile one isolated TikZ figure or TeX file and export PDF/PNG."
    )
    parser.add_argument("--input", required=True, help="Path to a TeX file or bare tikzpicture snippet.")
    parser.add_argument("--output-dir", required=True, help="Directory for build artifacts.")
    parser.add_argument("--job-name", help="Base name for generated files. Defaults to input stem.")
    parser.add_argument("--preamble-file", help="Optional TeX file whose contents are injected into the wrapper preamble.")
    parser.add_argument("--engine", default="xelatex", help="TeX engine passed to latexmk. Default: xelatex.")
    parser.add_argument("--density", type=int, default=200, help="PNG rasterization DPI. Default: 200.")
    args = parser.parse_args()

    ensure_command("latexmk")
    ensure_command("pdftocairo")

    input_path = pathlib.Path(args.input).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    job_name = args.job_name or input_path.stem
    source_text = read_text(input_path)
    extra_preamble = ""
    if args.preamble_file:
        extra_preamble = read_text(pathlib.Path(args.preamble_file).expanduser().resolve())

    if r"\documentclass" in source_text:
        tex_text = source_text
    else:
        tex_text = build_wrapper(source_text, extra_preamble)

    if input_path.name.startswith("main") and "scratch" not in str(input_path.parent).lower():
        sys.stderr.write(
            "Refusing likely integrated document input. Render figures from an isolated scratch file instead.\n"
        )
        raise SystemExit(2)

    tex_path = output_dir / f"{job_name}.tex"
    tex_path.write_text(tex_text, encoding="utf-8")

    run(
        [
            "latexmk",
            f"-{args.engine}",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=.",
            tex_path.name,
        ],
        cwd=output_dir,
    )

    pdf_path = output_dir / f"{job_name}.pdf"
    png_base = output_dir / job_name
    run(
        [
            "pdftocairo",
            "-png",
            "-singlefile",
            "-r",
            str(args.density),
            str(pdf_path),
            str(png_base),
        ],
        cwd=output_dir,
    )

    png_path = output_dir / f"{job_name}.png"
    print(f"tex={tex_path}")
    print(f"pdf={pdf_path}")
    print(f"png={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
