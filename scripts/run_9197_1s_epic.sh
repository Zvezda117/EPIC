#!/bin/bash
# 9197 单帧 1s 积分 + CasA 校准 → EPIC 成图
set -e
cd "$(dirname "$0")/.."

echo "=== 1. 生成 1s 校准数据 (约 3–4 分钟) ==="
source /opt/miniconda3/bin/activate epic310
export EPIC_ARRAY_CONFIG="$PWD/antenna_mapping_9197.yaml"
export EPIC_ANTENNA_COORDS="$PWD/antenna_coordinates.csv"
export EPIC_CENTER_FREQ_MHZ=1355
python3 scripts/convert_9197_1s_epic.py data/epic_ready/9197_1s_calib.npz

echo ""
echo "=== 2. 复制到 EPIC 输入路径 ==="
mkdir -p build/data
cp -f data/epic_ready/9197_1s_calib.npz build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz

echo ""
echo "=== 3. 编译 EPIC (若改过 C++) ==="
cd build && make -j8 2>/dev/null || true && cd ..

echo ""
echo "=== 4. 运行 EPIC (1s 单帧成图) ==="
mkdir -p results/images_1s
cd build
export EPIC_GULP_NPZ="../data/epic_ready/9197_1s_calib.npz"
./epic++ --offline --out_dir ../results/images_1s --disable_metrics \
  --imagesize 64 --channels 128 --chan_nbin 4 --seq_accum 40 --nts 1000 \
  --imageres 0.05 --accum_16bit --nstreams 1 --support 2 --aeff 8 \
  --kernel_oversample 2 --ngpus 1 --gpu_ids=0

echo ""
echo "完成。图像: results/images_1s/"
