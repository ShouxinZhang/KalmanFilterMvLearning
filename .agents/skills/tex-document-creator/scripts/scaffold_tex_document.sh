#!/usr/bin/env bash

set -euo pipefail

target_dir=""
doc_name=""
title=""
doc_class="ctexart"
build_dir_name="build"

usage() {
cat <<'EOF'
Usage:
  scaffold_tex_document.sh --dir <target_dir> --name <doc_name> [--title <title>] [--class <doc_class>] [--build-dir <dir_name>]

Examples:
  scaffold_tex_document.sh --dir notes/kf-note --name kf_note --title "KF 笔记"
  scaffold_tex_document.sh --dir slides/intro --name intro --class ctexbeamer --title "课程简介"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      target_dir="$2"
      shift 2
      ;;
    --name)
      doc_name="$2"
      shift 2
      ;;
    --title)
      title="$2"
      shift 2
      ;;
    --class)
      doc_class="$2"
      shift 2
      ;;
    --build-dir)
      build_dir_name="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$target_dir" || -z "$doc_name" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$title" ]]; then
  title="$doc_name"
fi

mkdir -p "$target_dir/$build_dir_name"

tex_path="$target_dir/$doc_name.tex"
compile_path="$target_dir/$build_dir_name/compile.sh"

if [[ -e "$tex_path" ]]; then
  echo "Refusing to overwrite existing file: $tex_path" >&2
  exit 1
fi

cat > "$tex_path" <<EOF
\documentclass[12pt,a4paper]{$doc_class}

\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath, amssymb, mathtools}
\usepackage{booktabs}
\usepackage{hyperref}

\title{$title}
\author{}
\date{\today}

\begin{document}

\maketitle

\section{背景}

在这里写文档背景。

\section{主体}

在这里写核心内容。

\section{结论}

在这里写总结。

\end{document}
EOF

cat > "$compile_path" <<EOF
#!/usr/bin/env bash

set -euo pipefail

build_dir="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
doc_dir="\$(cd -- "\${build_dir}/.." && pwd)"
doc_name="$doc_name"

rm -f \\
  "\${doc_dir}/\${doc_name}.aux" \\
  "\${doc_dir}/\${doc_name}.log" \\
  "\${doc_dir}/\${doc_name}.out" \\
  "\${doc_dir}/\${doc_name}.toc" \\
  "\${doc_dir}/\${doc_name}.fls" \\
  "\${doc_dir}/\${doc_name}.fdb_latexmk" \\
  "\${doc_dir}/\${doc_name}.synctex.gz" \\
  "\${doc_dir}/\${doc_name}.xdv"

cd "\${doc_dir}"

latexmk \\
  -xelatex \\
  -interaction=nonstopmode \\
  -halt-on-error \\
  -file-line-error \\
  -outdir="\${build_dir}" \\
  "\${doc_name}.tex"

cp "\${build_dir}/\${doc_name}.pdf" "\${doc_dir}/\${doc_name}.pdf"
rm -f "\${build_dir}/\${doc_name}.pdf"
EOF

chmod +x "$compile_path"

cat <<EOF
Created:
  $tex_path
  $compile_path

Expected top-level layout:
  $target_dir/$doc_name.tex
  $target_dir/$doc_name.pdf
  $target_dir/$build_dir_name/
EOF
