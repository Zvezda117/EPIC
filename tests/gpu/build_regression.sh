#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-${ROOT_DIR}/repair-build/gpu_regression}"
AUTOCORR_FLAG="${EPIC_HAS_AUTOCORR_FLAG:-1}"
MATHDX_DIR="${EPIC_MATHDX_DIR:-${ROOT_DIR}/src/extern/nvidia/mathdx/22.11/include}"
mkdir -p "$(dirname "$OUTPUT")"
/usr/local/cuda-12.1/bin/nvcc -std=c++17 -arch=sm_89 -lineinfo -O2 \
  -DEPIC_HAS_AUTOCORR_FLAG="$AUTOCORR_FLAG" \
  -I"${ROOT_DIR}/src/ex" -I"${MATHDX_DIR}/include" -I"${MATHDX_DIR}" \
  "${ROOT_DIR}/tests/gpu/imager_regression.cu" -o "$OUTPUT"
