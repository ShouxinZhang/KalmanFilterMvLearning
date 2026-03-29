#!/usr/bin/env bash

set -euo pipefail

note_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pdf_dir="${note_dir}/pdf"
build_dir="${note_dir}/build"

mkdir -p "${pdf_dir}" "${build_dir}"

(
  cd "${note_dir}"
  latexmk \
    -xelatex \
    -silent \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -outdir="${build_dir}" \
    "main.tex" > /dev/null 2>&1
)

cp "${build_dir}/main.pdf" "${pdf_dir}/main.pdf"
