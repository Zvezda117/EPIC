#!/bin/bash
# 使用改进转换的9197数据进行成像

set -e

cd /mnt/data_ssd/zdj/LWA_EPIC
source /opt/miniconda3/bin/activate epic310

echo "══════════════════════════════════════════════"
echo "  9197改进版数据处理和成像"
echo "══════════════════════════════════════════════"
echo ""

# 1. 转换数据（使用蟹状星云位置进行时延校准）
echo "━━━ 步骤1: 数据转换（含时延校准） ━━━"
echo ""

CRAB_RA=83.6331   # 蟹状星云赤经（度）
CRAB_DEC=22.0145  # 蟹状星云赤纬（度）

python3 scripts/convert_9197_improved.py $CRAB_RA $CRAB_DEC

echo ""
echo "━━━ 步骤2: 复制到EPIC输入位置 ━━━"
echo ""

# 检查生成的文件
CONVERTED_FILE="data/epic_ready/9197_214steps_512chan_improved.npz"

if [ ! -f "$CONVERTED_FILE" ]; then
    echo "❌ 转换文件未生成: $CONVERTED_FILE"
    exit 1
fi

cp $CONVERTED_FILE build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz
ls -lh build/data/40ms_128chan_600offset_gulp_c64_virtransit.npz

echo ""
echo "━━━ 步骤3: 运行EPIC（512通道） ━━━"
echo ""

# EPIC参数说明：
# --channels 512: 输出512个频率通道（需要修改过的EPIC）
# --nts 214: 214个时间步
# --seq_accum 2140: 积分时间2.14秒 = 214步 × 10ms
# --imageres 1.056: 像素分辨率
# --aeff 15.9: 4.5m口径天线有效面积 = π × (4.5/2)^2

cd build

# 注意：如果EPIC仍然有512通道限制，需要先修改源码并重新编译
# 临时方案：使用128通道，丢弃高频部分

echo "⚠️  注意: EPIC可能仍有128通道限制"
echo "当前使用128通道（512通道中的前128个）"
echo ""

./epic++ --offline \
    --out_dir ../results/images_improved \
    --disable_metrics \
    --imagesize 128 \
    --channels 128 \
    --chan_nbin 1 \
    --seq_accum 2140 \
    --nts 214 \
    --imageres 1.056 \
    --accum_16bit \
    --nstreams 8 \
    --support 5 \
    --aeff 15.9 \
    --kernel_oversample 4 \
    --ngpus 1 \
    --gpu_ids=0

cd ..

echo ""
echo "━━━ 步骤4: 检查输出 ━━━"
echo ""

ls -lh results/images_improved/EPIC_*.fits
ls -lh results/images_improved/EPIC_*.png

echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ 完成！"
echo "══════════════════════════════════════════════"
echo ""
echo "改进点："
echo "  ✓ 积分时间: 2.14秒 (vs 0.1秒，提升21x)"
echo "  ✓ 每步积分: 10ms (vs 0.1ms，提升100x)"
echo "  ✓ 时延校准: 基于蟹状星云位置"
echo "  ✓ 有效面积: 准确设置为15.9m²"
echo ""
echo "预期信噪比提升: √(21 × 100) ≈ 46倍"
echo ""
echo "查看图像："
echo "  ds9 results/images_improved/EPIC_*.fits"
echo "  eog results/images_improved/EPIC_*.png"
