#!/usr/bin/env bash
# Workstation entrypoint for the separately validated MOFF executable.
set -euo pipefail
epic_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
epic_python="${EPIC_PYTHON:-/home/zdj/.conda/envs/epic310/bin/python}"
epic_binary="${EPIC_MOFF_BINARY:-${epic_repo}/build/epic-repaired++}"
if [[ ! -x "$epic_python" || ! -x "$epic_binary" ]]; then
  echo 'Set EPIC_PYTHON and EPIC_MOFF_BINARY to the Python environment and rebuilt executable.' >&2
  exit 1
fi
epic_python_prefix="$("$epic_python" -c 'import sys; print(sys.prefix)')"
# The existing dependency build uses system GCC 13. The epic310 environment
# ships an older libstdc++; use the compiler runtime for this process only.
epic_cpp_runtime="${EPIC_CPP_RUNTIME:-$(c++ -print-file-name=libstdc++.so.6)}"
if [[ ! -f "$epic_cpp_runtime" ]]; then
  echo 'Set EPIC_CPP_RUNTIME to the libstdc++.so.6 matching the build compiler.' >&2
  exit 1
fi
exec env PYTHONHOME="$epic_python_prefix" \
  PYTHONPATH="${epic_repo}/src/python${PYTHONPATH:+:${PYTHONPATH}}" \
  LD_PRELOAD="${epic_cpp_runtime}${LD_PRELOAD:+:${LD_PRELOAD}}" \
  "$epic_binary" --offline "$@"
