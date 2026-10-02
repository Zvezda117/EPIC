#!/bin/bash
# 9197数据成图 - 最终版（绕过所有硬编码限制）

set -e

cd /mnt/data_ssd/zdj/LWA_EPIC
source /opt/miniconda3/bin/activate epic310

echo "═══════════════════════════════════════════════════════════"
echo "  9197数据EPIC成图 - 最终版本"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "【关键限制】"
echo "  1. RTX 4090: 最多128通道 (不是512)"
echo "  2. gulp_size: nts必须≤2500 (对于seq_accum=100)"
echo "  3. imagesize: 只能64或128"
echo ""
echo "【使用参数】"
echo "  --channels 128    # 1.95 MHz/通道"
echo "  --nts 1000        # 只用前0.1秒数据"
echo "  --seq_accum 100   # 100ms积分"
echo "  --imagesize 64    # 64×64像素"
echo ""
echo "【数据利用率】"
echo "  准备: 4.3秒 × 512通道"
echo "  使用: 0.1秒 × 128通道"
echo "  利用率: 0.6%"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# 检查128通道数据
echo "步骤1：检查128通道数据..."
if [ ! -f "data/epic_ready/9197_epic_128chan.npz" ]; then
    echo "  错误：128通道数据不存在！"
    echo "  正在转换..."
    python3 scripts/convert_to_epic_format.py \
        data/processed/9197_file0_complete.npz \
        -o data/epic_ready/9197_epic_128chan.npz \
        --nchan 128 \
        --ntime-samples 25000
    echo "  ✓ 转换完成"
else
    echo "  ✓ 128通道数据已存在"
    ls -lh data/epic_ready/9197_epic_128chan.npz
fi
echo ""

# 检查EPIC
echo "步骤2：检查EPIC编译..."
if [ ! -f "build/epic++" ]; then
    echo "  错误：EPIC未编译！"
    exit 1
fi
echo "  ✓ EPIC已编译"
echo ""

# 创建输出目录
mkdir -p results/images
echo "步骤3：运行EPIC成像..."
echo ""

# 进入build目录运行EPIC
cd build

# EPIC命令
./epic++ --offline \
    --npytbnfile ../data/epic_ready/9197_epic_128chan.npz \
    --out_dir ../results/images \
    --disable_metrics \
    --imagesize 64 \
    --imageres 5.0 \
    --channels 128 \
    --chan_nbin 2 \
    --seq_accum 100 \
    --nts 1000 \
    --support 2 \
    --aeff 10 \
    --kernel_oversample 2 \
    --nstreams 4 \
    --ngpus 1 \
    --gpu_ids=0

EPIC_EXIT=$?

cd ..

echo ""
echo "═══════════════════════════════════════════════════════════"

if [ $EPIC_EXIT -eq 0 ]; then
    echo "  ✅ 成像完成！"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "输出文件:"
    ls -lh results/images/EPIC_*.fits 2>/dev/null || echo "  (未生成FITS文件)"
    ls -lh results/images/EPIC_*.png 2>/dev/null || echo "  (未生成PNG文件)"
    echo ""
    echo "查看图像:"
    echo "  ds9 results/images/EPIC_*.fits"
    echo "  eog results/images/EPIC_*_QL.png"
else
    echo "  ❌ 成像失败 (退出码: $EPIC_EXIT)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "检查错误日志，或手动运行："
    echo "  cd build"
    echo "  ./epic++ --offline --npytbnfile ../data/epic_ready/9197_epic_128chan.npz ..."
fi

echo ""
