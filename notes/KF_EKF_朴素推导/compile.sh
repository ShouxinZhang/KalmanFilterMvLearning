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
      -silent \
      -interaction=nonstopmode \
      -halt-on-error \
      -file-line-error \
      -outdir="${out_dir}" \
      "${src_file}" > /dev/null 2>&1
  )

  cp "${out_dir}/${job_name}.pdf" "${out_pdf}"
  rm -f "${out_dir}/${job_name}.pdf"
}

mkdir -p "${pdf_dir}/derivations" "${build_dir}/main" "${build_dir}/derivations"

# 并行编译所有 tex 文件（main + derivations）
pids=()

compile_tex "main.tex" "${build_dir}/main" "${pdf_dir}/main.pdf" "${note_dir}" &
pids+=($!)

for src_file in derivations/*.tex; do
  job_name="$(basename "${src_file%.tex}")"
  mkdir -p "${build_dir}/derivations/${job_name}"
  compile_tex "${job_name}.tex" "${build_dir}/derivations/${job_name}" "${pdf_dir}/derivations/${job_name}.pdf" "${note_dir}/derivations" &
  pids+=($!)
done

# 等待全部完成，任一失败则报错退出
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "ERROR: one or more compilations failed" >&2
  exit 1
fi
