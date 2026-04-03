#!/usr/bin/env bash

set -euo pipefail

note_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pdf_dir="${note_dir}/pdf"
build_dir="${note_dir}/build"

compile_tex() {
  local src_file="$1"
  local out_dir="$2"
  local out_pdf="$3"
  local job_name

  job_name="$(basename "${src_file%.tex}")"
  mkdir -p "${out_dir}"

  (
    cd "${note_dir}"
    latexmk \
      -xelatex \
      -silent \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -outdir="${out_dir}" \
      "${src_file}" > /dev/null 2>&1
  )

  cp "${out_dir}/${job_name}.pdf" "${out_pdf}"
}

mkdir -p "${pdf_dir}" "${build_dir}/main"
compile_tex "main.tex" "${build_dir}/main" "${pdf_dir}/main.pdf"
mkdir -p "${build_dir}/main_zh-cn"
compile_tex "main_zh-cn.tex" "${build_dir}/main_zh-cn" "${pdf_dir}/main_zh-cn.pdf"
