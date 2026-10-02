#!/bin/bash
# 9197数据成图完整流程

set -e

cd /mnt/data_ssd/zdj/LWA_EPIC
source /opt/miniconda3/bin/activate epic310

echo "==========================================="
echo "  9197 数据EPIC成图流程"
echo "==========================================="
echo ""

# 步骤1：验证处理后的数据
echo "步骤1：验证处理后的数据..."
if [ ! -f "data/processed/9197_file0_complete.npz" ]; then
    echo "错误：9197_file0_complete.npz 不存在！"
    echo "请先运行：python scripts/process_9197_single_channel.py --files 0"
    exit 1
fi

python << 'EOF'
import numpy as np
data = np.load('data/processed/9197_file0_complete.npz', allow_pickle=True)
v = data['voltages']
print(f"  ✓ 数据形状: {v.shape}")
print(f"  ✓ 采样点: {v.shape[0]:,}")
print(f"  ✓ 时间长度: {v.shape[0] / 250e6:.3f} 秒")
print(f"  ✓ 天线数: {v.shape[2]}")
print(f"  ✓ 平均功率: {np.abs(v).mean():.6f}")
EOF

echo ""

# 步骤2：检查EPIC格式数据
echo "步骤2：检查EPIC格式数据..."
if [ ! -f "data/epic_ready/9197_epic_512chan.npz" ]; then
    echo "错误：EPIC数据不存在，正在转换..."
    python scripts/convert_to_epic_format.py \
        data/processed/9197_file0_complete.npz \
        -o data/epic_ready/9197_epic_512chan.npz \
        --nchan 512 \
        --ntime-samples 25000
else
    echo "  ✓ EPIC数据已存在"
    ls -lh data/epic_ready/9197_epic_512chan.npz
fi

echo ""

# 步骤3：检查EPIC是否已编译
echo "步骤3：检查EPIC..."
if [ ! -f "build/epic++" ]; then
    echo "错误：EPIC未编译！"
    echo "请运行："
    echo "  cd build && cmake .. && make"
    exit 1
fi

echo "  ✓ EPIC已编译"
echo ""

# 步骤4：运行EPIC
echo "步骤4：运行EPIC成像..."
mkdir -p results/images

cd build

echo "  注意：由于EPIC的gulp_size=nts×0.04限制"
echo "       使用 nts=1000 (约4.3秒数据中的前100ms)"
echo ""

./epic++ --offline \
    --npytbnfile ../data/epic_ready/9197_epic_512chan.npz \
    --out_dir ../results/images \
    --disable_metrics \
    --imagesize 64 \
    --imageres 5.0 \
    --channels 512 \
    --chan_nbin 4 \
    --seq_accum 100 \
    --nts 1000 \
    --support 2 \
    --aeff 10 \
    --kernel_oversample 2 \
    --nstreams 4 \
    --ngpus 1 \
    --gpu_ids=0

cd ..

cd ..

echo ""
echo "==========================================="
echo "  成像完成！"
echo "==========================================="
echo ""
echo "输出文件:"
ls -lh results/images/EPIC_*.fits 2>/dev/null || echo "  (未生成FITS文件)"
ls -lh results/images/EPIC_*.png 2>/dev/null || echo "  (未生成PNG文件)"
echo ""
echo "查看图像:"
echo "  ds9 results/images/EPIC_*.fits"
echo ""
