#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rasterize one page from a PDF to PNG.")
    parser.add_argument("--pdf", required=True, help="Input PDF path.")
    parser.add_argument("--page", required=True, type=int, help="1-based page number.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--density", type=int, default=200, help="PNG rasterization DPI. Default: 200.")
    args = parser.parse_args()

    ensure_command("pdftocairo")

    pdf_path = pathlib.Path(args.pdf).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.with_suffix("")

    proc = subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-singlefile",
            "-f",
            str(args.page),
            "-l",
            str(args.page),
            "-r",
            str(args.density),
            str(pdf_path),
            str(base),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
