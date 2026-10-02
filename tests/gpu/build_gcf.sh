#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-${ROOT_DIR}/repair-build/gcf_regression}"
MATHDX_DIR="${EPIC_MATHDX_DIR:-${ROOT_DIR}/src/extern/nvidia/mathdx/22.11/include}"
mkdir -p "$(dirname "$OUTPUT")"
/usr/local/cuda-12.1/bin/nvcc -std=c++17 -arch=sm_89 -lineinfo -O2 \
  -I"${ROOT_DIR}/src/ex" -I"${MATHDX_DIR}" \
  "${ROOT_DIR}/tests/gpu/gcf_regression.cu" -o "$OUTPUT"
