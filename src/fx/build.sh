#!/usr/bin/env bash
set -euo pipefail
fx_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${NVCC:-nvcc}" -O3 -std=c++17 -arch="${CUDA_ARCH:-sm_89}" -shared \
  -Xcompiler=-fPIC "$fx_root/lband_fx.cu" -lcublas \
  -Xlinker=-rpath -Xlinker="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}" \
  -o "$fx_root/libepic_fx.so"
