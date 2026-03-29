#!/usr/bin/env bash

set -euo pipefail

note_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pdf_dir="${note_dir}/pdf"
build_dir="${note_dir}/build"

compile_tex() {
  local src_file="$1"
  local out_dir="$2"
  local out_pdf="$3"
  local src_dir="$4"
  local job_name

  job_name="$(basename "${src_file%.tex}")"
  mkdir -p "${out_dir}"

  (
    cd "${src_dir}"
    latexmk \
      -xelatex \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -outdir="${out_dir}" \
      "${src_file}"
  )

  cp "${out_dir}/${job_name}.pdf" "${out_pdf}"
  rm -f "${out_dir}/${job_name}.pdf"
}

rm -rf "${pdf_dir}" "${build_dir}"
mkdir -p "${pdf_dir}/derivations" "${build_dir}/main" "${build_dir}/derivations"

compile_tex "main.tex" "${build_dir}/main" "${pdf_dir}/main.pdf" "${note_dir}"

for src_file in derivations/*.tex; do
  job_name="$(basename "${src_file%.tex}")"
  compile_tex "${job_name}.tex" "${build_dir}/derivations/${job_name}" "${pdf_dir}/derivations/${job_name}.pdf" "${note_dir}/derivations"
done
