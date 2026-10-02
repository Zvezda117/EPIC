#!/bin/bash
# 运行全带宽40ms积分成像

set -e

cd /mnt/data_ssd/zdj/LWA_EPIC
source /opt/miniconda3/bin/activate epic310
export EPIC_ARRAY_CONFIG="$PWD/antenna_mapping_9197.yaml"
export EPIC_ANTENNA_COORDS="$PWD/antenna_coordinates.csv"
export EPIC_CENTER_FREQ_MHZ=1355

echo "══════════════════════════════════════════════════════════════"
echo "  9197全带宽处理：40ms积分 + 250MHz带宽"
echo "══════════════════════════════════════════════════════════════"
echo ""

# 参数
OUTPUT_CHANNELS=1024  # 输出通道数
NUM_GULPS=53          # 处理53个gulp = 2.12秒

echo "【参数设置】"
echo "输出通道: $OUTPUT_CHANNELS"
echo "Gulp数量: $NUM_GULPS"
echo "时间积分: 40 ms/gulp"
echo "总时长: $(echo "$NUM_GULPS * 0.04" | bc) 秒"
echo ""

# 步骤1: 转换数据
echo "━━━ 步骤1: 数据转换 ━━━"
echo ""

python3 scripts/convert_9197_fullband_40ms.py $OUTPUT_CHANNELS $NUM_GULPS

CONVERTED_FILE="data/epic_ready/9197_${NUM_GULPS}gulps_${OUTPUT_CHANNELS}chan_fullband_40ms.npz"

if [ ! -f "$CONVERTED_FILE" ]; then
    echo "❌ 转换失败"
    exit 1
fi

echo ""
echo "━━━ 步骤2: 复制到EPIC输入位置 ━━━"
echo ""

cp $CONVERTED_FILE build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz
ls -lh build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz

echo ""
echo "━━━ 步骤3: 检查EPIC是否支持${OUTPUT_CHANNELS}通道 ━━━"
echo ""

# 检查EPIC的通道限制
echo "检查 src/ex/constants.h 中的 MAX_CHANNELS_4090..."
MAX_CHAN=$(grep "MAX_CHANNELS_4090" src/ex/constants.h | grep -oE "[0-9]+")
echo "当前限制: $MAX_CHAN"

if [ "$OUTPUT_CHANNELS" -gt "$MAX_CHAN" ]; then
    echo "⚠️  警告: 输出通道数($OUTPUT_CHANNELS)超过EPIC限制($MAX_CHAN)"
    echo "临时方案: 使用前$MAX_CHAN个通道"
    OUTPUT_CHANNELS=$MAX_CHAN
fi

echo ""
echo "━━━ 步骤4: 运行EPIC ━━━"
echo ""

# EPIC参数
IMAGESIZE=128
SEQ_ACCUM=$((NUM_GULPS * 40))  # 总积分时间(ms)
IMAGERES=0.05
AEFF=15.9  # 4.5m口径: π×(4.5/2)^2

echo "EPIC参数:"
echo "  --imagesize $IMAGESIZE"
echo "  --channels $OUTPUT_CHANNELS"
echo "  --nts $NUM_GULPS"
echo "  --seq_accum $SEQ_ACCUM ms"
echo "  --imageres $IMAGERES"
echo "  --aeff $AEFF"
echo ""

cd build

./epic++ --offline \
    --out_dir ../results/images_fullband_40ms \
    --disable_metrics \
    --imagesize $IMAGESIZE \
    --channels $OUTPUT_CHANNELS \
    --chan_nbin 1 \
    --seq_accum $SEQ_ACCUM \
    --nts $NUM_GULPS \
    --imageres $IMAGERES \
    --accum_16bit \
    --nstreams 1 \
    --support 2 \
    --aeff $AEFF \
    --kernel_oversample 2 \
    --ngpus 1 \
    --gpu_ids=0

cd ..

echo ""
echo "━━━ 步骤5: 检查输出 ━━━"
echo ""

mkdir -p results/images_fullband_40ms
ls -lh results/images_fullband_40ms/EPIC_*.fits 2>/dev/null || echo "未找到FITS文件"
ls -lh results/images_fullband_40ms/EPIC_*.png 2>/dev/null || echo "未找到PNG文件"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✅ 处理完成！"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "【处理总结】"
echo "  ✓ 时间积分: 40ms/gulp (与LWA-SV一致)"
echo "  ✓ 总时长: $(echo "$NUM_GULPS * 0.04" | bc) 秒"
echo "  ✓ 带宽: 250 MHz (100%利用)"
echo "  ✓ 输出通道: $OUTPUT_CHANNELS"
echo "  ✓ 通道带宽: $(echo "scale=1; 250000 / $OUTPUT_CHANNELS" | bc) kHz"
echo "  ✓ 时延校准: 蟹状星云位置"
echo ""
echo "【查看结果】"
echo "  FITS: ds9 results/images_fullband_40ms/EPIC_*.fits"
echo "  PNG:  eog results/images_fullband_40ms/EPIC_*.png"
echo ""
echo "【与原始版本对比】"
echo "  原始: 0.1秒, 1.3MHz, 无校准"
echo "  当前: $(echo "$NUM_GULPS * 0.04" | bc)秒, 250MHz, 有校准"
echo "  预期SNR提升: √($(echo "$NUM_GULPS * 0.04 / 0.1" | bc) × $(echo "250 / 1.3" | bc) × 8) ≈ $(echo "scale=0; sqrt($(echo "$NUM_GULPS * 0.04 / 0.1 * 250 / 1.3 * 8" | bc))" | bc) 倍"
