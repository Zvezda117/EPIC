#!/bin/bash
# 9197 多 .data 文件 → 每文件 1s 转换 → EPIC 成图 → 多图拼接
# 用法: ./run_9197_multi_epic_stitch.sh [文件索引...]
# 例:   ./run_9197_multi_epic_stitch.sh 0 1 2 3     # 4 张图，默认 2x2 拼接
#       ./run_9197_multi_epic_stitch.sh 0 1 2 3 4 5 --grid 2 3   # 6 张图，2 行 3 列
set -e
cd "$(dirname "$0")/.."
export MKL_THREADING_LAYER=GNU
source /opt/miniconda3/bin/activate epic310 2>/dev/null || true
python3 scripts/convert_9197_multi_epic_and_stitch.py "$@"
echo "结果目录: results/multi_9197/"
echo "拼接图:   results/multi_9197/stitched.png"
